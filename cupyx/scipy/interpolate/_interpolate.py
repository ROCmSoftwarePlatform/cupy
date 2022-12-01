
import cupy
from cupy._core import internal  # NOQA
from cupy._core._scalar import get_typename  # NOQA

import numpy as np

TYPES = ['double', 'thrust::complex<double>']
INT_TYPES = ['int', 'long long']

INTERVAL_KERNEL = r'''
#include <cupy/complex.cuh>

#define le_or_ge(x, y, r) ((r) ? ((x) < (y)) : ((x) > (y)))
#define ge_or_le(x, y, r) ((r) ? ((x) > (y)) : ((x) < (y)))
#define geq_or_leq(x, y, r) ((r) ? ((x) >= (y)) : ((x) <= (y)))

extern "C" {
__global__ void find_breakpoint_position(
        const double* breakpoints, const double* x, long long* out,
        bool extrapolate, int total_x, int total_breakpoints, bool asc) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= total_x) {
        return;
    }

    double xp = *&x[idx];
    double a = *&breakpoints[0];
    double b = *&breakpoints[total_breakpoints - 1];

    if(isnan(xp)) {
        out[idx] = -1;
        return;
    }

    if(le_or_ge(xp, a, asc) || ge_or_le(xp, b, asc)) {
        if(!extrapolate) {
            out[idx] = -1;
        } else if(le_or_ge(xp, a, asc)) {
            out[idx] = 0;
        } else if(ge_or_le(xp, b, asc)) {
            out[idx] = total_breakpoints - 2;
        }
        return;
    } else if (xp == b) {
        out[idx] = total_breakpoints - 2;
        return;
    }

    int left = 0;
    int right = total_breakpoints - 2;
    int mid;

    if(le_or_ge(xp, *&breakpoints[left + 1], asc)) {
        right = left;
    }

    bool found = false;

    while(left < right && !found) {
        mid = ((right + left) / 2);
        if(le_or_ge(xp, *&breakpoints[mid], asc)) {
            right = mid;
        } else if (geq_or_leq(xp, *&breakpoints[mid + 1], asc)) {
            left = mid + 1;
        } else {
            found = true;
            left = mid;
        }
    }

    out[idx] = left;
}
}
'''

INTERVAL_MODULE = cupy.RawModule(
    code=INTERVAL_KERNEL, options=('-std=c++11',),)

PPOLY_KERNEL = r"""
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cupy/complex.cuh>
#include <cupy/math_constants.h>

template<typename T>
__device__ T eval_poly_1(
        const double s, const T* coef, long long ci, int cj, int dx,
        const long long* c_dims, const long long stride_0,
        const long long stride_1) {
    int kp, k;
    T res, z;
    double prefactor;

    res = 0.0;
    z = 1.0;

    if(dx < 0) {
        for(int i = 0; i < -dx; i++) {
            z *= s;
        }
    }

    int c_dim_0 = (int) *&c_dims[0];

    for(kp = 0; kp < c_dim_0; kp++) {
        if(dx == 0) {
            prefactor = 1.0;
        } else if(dx > 0) {
            if(kp < dx) {
                continue;
            } else {
                prefactor = 1.0;
                for(k = kp; k > kp - dx; k--) {
                    prefactor *= k;
                }
            }
        } else {
            prefactor = 1.0;
            for(k = kp; k < kp - dx; k++) {
                prefactor /= k + 1;
            }
        }

        int off = stride_0 * (c_dim_0 - kp - 1) + stride_1 * ci + cj;
        T cur_coef = *&coef[off];
        res += cur_coef * z * ((T) prefactor);

        if((kp < c_dim_0 - 1) && kp >= dx) {
            z *= s;
        }

    }

    return s;

}

template<typename T>
__global__ void eval_ppoly(
        const T* coef, const double* breakpoints, const double* x,
        const long long* intervals, int dx, const long long* c_dims,
        const long long* c_strides, int num_x, T* out) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if(idx >= num_x) {
        return;
    }

    double xp = *&x[idx];
    long long interval = *&intervals[idx];
    double breakpoint = *&breakpoints[interval];

    const long long num_c = *&c_strides[0];
    const long long stride_1 = *&c_strides[1];

    if(interval < 0) {
        for(int j = 0; j < num_c; j++) {
            out[num_c * idx + j] = CUDART_NAN;
        }
        return;
    }

    for(int j = 0; j < num_c; j++) {
        out[num_c * idx + j] = eval_poly_1<T>(
            xp - breakpoint, coef, interval, ((long long) (j)), dx,
            c_dims, num_c, stride_1);
    }
}
"""

PPOLY_MODULE = cupy.RawModule(
    code=PPOLY_KERNEL, options=('-std=c++11',),
    name_expressions=[f'eval_ppoly<{type_name}>' for type_name in TYPES])


def _get_module_func(module, func_name, *template_args):
    def _get_typename(dtype):
        typename = get_typename(dtype)
        if dtype.kind == 'c':
            typename = 'thrust::' + typename
        return typename
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel


def _ppoly_evaluate(c, x, xp, dx, extrapolate, out):
    """
    Evaluate a piecewise polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, n)
        Coefficients local polynomials of order `k-1` in `m` intervals.
        There are `n` polynomials in each interval.
        Coefficient of highest order-term comes first.
    x : ndarray, shape (m+1,)
        Breakpoints of polynomials.
    xp : ndarray, shape (r,)
        Points to evaluate the piecewise polynomial at.
    dx : int
        Order of derivative to evaluate.  The derivative is evaluated
        piecewise and may have discontinuities.
    extrapolate : bint
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs.
    out : ndarray, shape (r, n)
        Value of each polynomial at each of the input points.
        This argument is modified in-place.
    """
    # Determine if the breakpoints are in ascending order or descending one
    ascending = x[x.shape[0] - 1] >= x[0]

    intervals = cupy.empty(xp.shape[0], dtype=cupy.int_)
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position')
    interval_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,),
                    (x, xp, intervals, extrapolate, xp.shape[0], x.shape[0],
                     ascending))

    # Compute coefficient displacement stride (in elements)
    c_shape = cupy.asarray(c.shape, dtype=cupy.int_)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int_) // c.itemsize

    ppoly_kernel = _get_module_func(PPOLY_MODULE, 'eval_ppoly', c)
    ppoly_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,),
                 (c, x, xp, intervals, dx, c_shape, c_strides,
                  xp.shape[0], out))


class _PPolyBase:
    """Base class for piecewise polynomials."""
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        self.c = cupy.asarray(c)
        self.x = cupy.ascontiguousarray(x, dtype=cupy.float64)

        if extrapolate is None:
            extrapolate = True
        elif extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        self.extrapolate = extrapolate

        if self.c.ndim < 2:
            raise ValueError("Coefficients array must be at least "
                             "2-dimensional.")

        if not (0 <= axis < self.c.ndim - 1):
            raise ValueError("axis=%s must be between 0 and %s" %
                             (axis, self.c.ndim-1))

        self.axis = axis
        if axis != 0:
            # move the interpolation axis to be the first one in self.c
            # More specifically, the target shape for self.c is (k, m, ...),
            # and axis !=0 means that we have c.shape (..., k, m, ...)
            #                                               ^
            #                                              axis
            # So we roll two of them.
            self.c = cupy.moveaxis(self.c, axis+1, 0)
            self.c = cupy.moveaxis(self.c, axis+1, 0)

        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.size < 2:
            raise ValueError("at least 2 breakpoints are needed")
        if self.c.ndim < 2:
            raise ValueError("c must have at least 2 dimensions")
        if self.c.shape[0] == 0:
            raise ValueError("polynomial must be at least of order 0")
        if self.c.shape[1] != self.x.size-1:
            raise ValueError("number of coefficients != len(x)-1")
        dx = cupy.diff(self.x)
        if not (cupy.all(dx >= 0) or cupy.all(dx <= 0)):
            raise ValueError("`x` must be strictly increasing or decreasing.")

        dtype = self._get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if (cupy.issubdtype(dtype, cupy.complexfloating)
                or cupy.issubdtype(self.c.dtype, cupy.complexfloating)):
            return cupy.complex_
        else:
            return cupy.float_

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        """
        Construct the piecewise polynomial without making checks.
        Takes the same parameters as the constructor. Input arguments
        ``c`` and ``x`` must be arrays of the correct shape and type. The
        ``c`` array can only be of dtypes float and complex, and ``x``
        array must have dtype float.
        """
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _ensure_c_contiguous(self):
        """
        c and x may be modified by the user. The Cython code expects
        that they are C contiguous.
        """
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x):
        """
        Add additional breakpoints and coefficients to the polynomial.
        Parameters
        ----------
        c : ndarray, size (k, m, ...)
            Additional coefficients for polynomials in intervals. Note that
            the first additional interval will be formed using one of the
            ``self.x`` end points.
        x : ndarray, size (m,)
            Additional breakpoints. Must be sorted in the same order as
            ``self.x`` and either to the right or to the left of the current
            breakpoints.
        """

        c = cupy.asarray(c)
        x = cupy.asarray(x)

        if c.ndim < 2:
            raise ValueError("invalid dimensions for c")
        if x.ndim != 1:
            raise ValueError("invalid dimensions for x")
        if x.shape[0] != c.shape[1]:
            raise ValueError("Shapes of x {} and c {} are incompatible"
                             .format(x.shape, c.shape))
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError("Shapes of c {} and self.c {} are incompatible"
                             .format(c.shape, self.c.shape))

        if c.size == 0:
            return

        dx = cupy.diff(x)
        if not (cupy.all(dx >= 0) or cupy.all(dx <= 0)):
            raise ValueError("`x` is not sorted.")

        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")

            if x[0] >= self.x[-1]:
                action = 'append'
            elif x[-1] <= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")
        else:
            if not x[-1] <= x[0]:
                raise ValueError("`x` is in the different order "
                                 "than `self.x`.")

            if x[0] <= self.x[-1]:
                action = 'append'
            elif x[-1] >= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError("`x` is neither on the left or on the right "
                                 "from `self.x`.")

        dtype = self._get_dtype(c.dtype)

        k2 = max(c.shape[0], self.c.shape[0])
        c2 = cupy.zeros(
            (k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:],
            dtype=dtype)

        if action == 'append':
            c2[k2 - self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2 - c.shape[0]:, self.c.shape[1]:] = c
            self.x = cupy.r_[self.x, x]
        elif action == 'prepend':
            c2[k2 - self.c.shape[0]:, :c.shape[1]] = c
            c2[k2 - c.shape[0]:, c.shape[1]:] = self.c
            self.x = cupy.r_[x, self.x]

        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        """
        Evaluate the piecewise polynomial or its derivative.
        Parameters
        ----------
        x : array_like
            Points to evaluate the interpolant at.
        nu : int, optional
            Order of derivative to evaluate. Must be non-negative.
        extrapolate : {bool, 'periodic', None}, optional
            If bool, determines whether to extrapolate to out-of-bounds points
            based on first and last intervals, or to return NaNs.
            If 'periodic', periodic extrapolation is used.
            If None (default), use `self.extrapolate`.
        Returns
        -------
        y : array_like
            Interpolated values. Shape is determined by replacing
            the interpolation axis in the original array with the shape of x.
        Notes
        -----
        Derivatives are evaluated piecewise for each polynomial
        segment, even if the polynomial is not differentiable at the
        breakpoints. The polynomial intervals are considered half-open,
        ``[a, b)``, except for the last interval which is closed
        ``[a, b]``.
        """
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = cupy.asarray(x)
        x_shape, x_ndim = x.shape, x.ndim
        x = cupy.ascontiguousarray(x.ravel(), dtype=cupy.float_)

        # With periodic extrapolation we map x to the segment
        # [self.x[0], self.x[-1]].
        if extrapolate == 'periodic':
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False

        out = cupy.empty((len(x), int(np.prod(self.c.shape[2:]))),
                         dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            # transpose to move the calculated values to the interpolation axis
            dims = list(range(out.ndim))
            dims = (dims[x_ndim:x_ndim + self.axis] + dims[:x_ndim] +
                    dims[x_ndim + self.axis:])
            out = out.transpose(dims)
        return out


class PPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints
    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i]) ** (k - m) for m in range(k + 1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    See also
    --------
    BPoly : piecewise polynomials in the Bernstein basis

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.

    .. seealso:: :class:`scipy.interpolate.BSpline`
    """

    def _evaluate(self, x, nu, extrapolate, out):
        _ppoly_evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1),
                        self.x, x, nu, bool(extrapolate), out)
