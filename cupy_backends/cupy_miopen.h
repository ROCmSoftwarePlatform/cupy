// This file is a stub header file of cudnn for Read the Docs.


#ifndef INCLUDE_GUARD_CUPY_CUDNN_H
#define INCLUDE_GUARD_CUPY_CUDNN_H
#if CUPY_USE_HIP

#include <miopen/miopen.h>

#elif !defined(CUPY_NO_CUDA)

#include <cudnn.h>

#elif defined(CUPY_NO_CUDA)

#include "stub/cupy_cuda_common.h"
#include "stub/cupy_cudnn.h"


#endif // #ifdef CUPY_NO_CUDA
#endif // #ifndef INCLUDE_GUARD_CUPY_CUDNN_H
