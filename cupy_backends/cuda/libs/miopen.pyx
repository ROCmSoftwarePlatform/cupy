# distutils: language = c++

"""Thin wrapper of cuDNN."""
# NOTE: This wrapper does not cover all APIs of cuDNN v4.
cimport cython  # NOQA
from libcpp cimport vector

from cupy_backends.cuda.api cimport driver
from cupy_backends.cuda.api cimport runtime
from cupy_backends.cuda cimport stream as stream_module

###############################################################################
# Extern
###############################################################################

cdef extern from '../../cupy_cudnn.h' nogil:
    # Types
    ctypedef int ActivationMode 'miopenActivationMode_t'
    ctypedef int AddMode 'cudnnAddMode_t'
    ctypedef int BatchNormMode 'miopenBatchNormMode_t'
    ctypedef int BatchNormOps 'cudnnBatchNormOps_t'
    ctypedef int ConvolutionBwdDataAlgo 'miopenBwdDataAlgorithm_t'
    ctypedef int ConvolutionBwdDataPreference \
        'cudnnConvolutionBwdDataPreference_t'
    ctypedef struct ConvolutionBwdDataAlgoPerf \
        'cudnnConvolutionBwdDataAlgoPerf_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
    ctypedef struct ConvolutionBwdDataAlgoPerf_v7 \
        'cudnnConvolutionBwdDataAlgoPerf_v7_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
        int determinism
        int mathType
    ctypedef int ConvolutionBwdFilterAlgo 'miopenConvBwdWeightsAlgorithm_t'
    ctypedef int ConvolutionBwdFilterPreference \
        'cudnnConvolutionBwdFilterPreference_t'
    ctypedef struct ConvolutionBwdFilterAlgoPerf \
        'cudnnConvolutionBwdFilterAlgoPerf_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
    ctypedef struct ConvolutionBwdFilterAlgoPerf_v7 \
        'cudnnConvolutionBwdFilterAlgoPerf_v7_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
        int determinism
        int mathType
    ctypedef int ConvolutionFwdAlgo 'miopenConvolutionFwdAlgorithm_t'
    ctypedef int ConvolutionFwdPreference 'cudnnConvolutionFwdPreference_t'
    ctypedef struct ConvolutionFwdAlgoPerf 'cudnnConvolutionFwdAlgoPerf_t':
        int algo
        int status
        float time
        size_t memory
    ctypedef struct ConvolutionFwdAlgoPerf_v7 \
        'cudnnConvolutionFwdAlgoPerf_v7_t':  # NOQA: E125
        int algo
        int status
        float time
        size_t memory
        int determinism
        int mathType
    ctypedef int ConvolutionMode 'miopenConvolutionMode_t'
    ctypedef int DataType 'miopenDataType_t'
    ctypedef int MathType 'cudnnMathType_t'
    ctypedef int DirectionMode 'miopenRNNDirectionMode_t'
    ctypedef int NanPropagation 'miopenNanPropagation_t'
    ctypedef int PoolingMode 'miopenPoolingMode_t'
    ctypedef int RNNInputMode 'miopenRNNInputMode_t'
    ctypedef int CTCLossAlgo 'miopenCTCLossAlgo_t'
    ctypedef int RNNMode 'miopenRNNMode_t'
    ctypedef int RNNAlgo 'miopenRNNAlgo_t'
    ctypedef int RNNDataLayout 'cudnnRNNDataLayout_t'
    ctypedef int RNNPaddingMode 'cudnnRNNPaddingMode_t'
    ctypedef int SoftmaxAlgorithm 'miopenSoftmaxAlgorithm_t'
    ctypedef int SoftmaxMode 'miopenSoftmaxMode_t'
    ctypedef int Status 'miopenStatus_t'
    ctypedef int TensorFormat 'cudnnTensorFormat_t'
    ctypedef int OpTensorOp 'miopenTensorOp_t'
	
    ctypedef int ReduceTensorOp 'miopenReduceTensorOp_t'
    ctypedef int ReduceTensorIndices 'miopenReduceTensorIndices_t'
    ctypedef int IndicesType 'miopenIndicesType_t'
    ctypedef int ErrQueryMode 'cudnnErrQueryMode_t'
    ctypedef int FusedOps 'cudnnFusedOps_t'
    ctypedef int FusedOpsConstParamLabel 'cudnnFusedOpsConstParamLabel_t'
    ctypedef int FusedOpsPointerPlaceHolder 'cudnnFusedOpsPointerPlaceHolder_t'
    ctypedef int FusedOpsVariantParamLabel 'cudnnFusedOpsVariantParamLabel_t'
    ctypedef struct RuntimeTag 'cudnnRuntimeTag_t'

    ctypedef void* ActivationDescriptor 'miopenActivationDescriptor_t'
    ctypedef void* ConvolutionDescriptor 'miopenConvolutionDescriptor_t'
    ctypedef void* DropoutDescriptor 'miopenDropoutDescriptor_t'
    ctypedef void* FilterDescriptor 'cudnnFilterDescriptor_t'
    ctypedef void* Handle 'miopenHandle_t'
    ctypedef void* PoolingDescriptor 'miopenPoolingDescriptor_t'
    ctypedef void* CTCLossDescriptor 'miopenCTCLossDescriptor_t'
    ctypedef void* RNNDescriptor 'miopenRNNDescriptor_t'
    ctypedef void* RNNDataDescriptor 'miopenRNNDataDescriptor_t'
    ctypedef void* PersistentRNNPlan 'cudnnPersistentRNNPlan_t'
    ctypedef void* TensorDescriptor 'miopenTensorDescriptor_t'
    ctypedef void* OpTensorDescriptor 'miopenTensorDescriptor_t'
    ctypedef void* ReduceTensorDescriptor 'miopenReduceTensorDescriptor_t'
    ctypedef void* SpatialTransformerDescriptor \
        'cudnnSpatialTransformerDescriptor_t'
    ctypedef void* SamplerType 'cudnnSamplerType_t'
    ctypedef void* FusedOpsConstParamPack 'cudnnFusedOpsConstParamPack_t'
    ctypedef void* FusedOpsVariantParamPack 'cudnnFusedOpsVariantParamPack_t'
    ctypedef void* FusedOpsPlan 'cudnnFusedOpsPlan_t'

    # Error handling
    const char* miopenGetErrorString(Status status)

    # Version
    size_t miopenGetVersion()

    # Runtime error checking
    int cudnnQueryRuntimeError(Handle handle, Status *rstatus,
                               ErrQueryMode mode, RuntimeTag *tag)

    # Initialization and CUDA cooperation
    int miopenCreate(Handle* handle)
    int miopenDestroy(Handle handle)
    int miopenSetStream(Handle handle, driver.Stream stream)
    int miopenGetStream(Handle handle, driver.Stream* stream)

    # Tensor manipulation
    int miopenCreateTensorDescriptor(TensorDescriptor* descriptor)
    int miopenSet4dTensorDescriptor(
        TensorDescriptor tensorDesc, 
        DataType dataType, int n, int c, int h, int w)
    int miopenSet4dTensorDescriptorEx(
        TensorDescriptor tensorDesc, DataType dataType,
        int n, int c, int h, int w,
        int nStride, int cStride, int hStride, int wStride)
    int miopenGet4dTensorDescriptor(
        TensorDescriptor tensorDesc, DataType* dataType,
        int* n, int* c, int* h, int* w,
        int* nStride, int* cStride, int* hStride, int* wStride)
    int cudnnSetTensorNdDescriptor(
        TensorDescriptor tensorDesc, DataType dataType, int nbDims,
        int* dimA, int* strideA)
    int miopenDestroyTensorDescriptor(TensorDescriptor tensorDesc)
    int cudnnAddTensor_v3(
        Handle handle, void* alpha, TensorDescriptor bDesc,
        void* b, void* beta, TensorDescriptor yDesc, void* y)

    # Tensor operations
    int cudnnCreateOpTensorDescriptor(OpTensorDescriptor* opTensorDesc)
    int cudnnSetOpTensorDescriptor(
        OpTensorDescriptor opTensorDesc, OpTensorOp opTensorOp,
        DataType opTensorCompType, NanPropagation opTensorNanOpt)
    int cudnnGetOpTensorDescriptor(
        OpTensorDescriptor opTensorDesc, OpTensorOp* opTensorOp,
        DataType* opTensorCompType, NanPropagation* opTensorNanOpt)
    int cudnnDestroyOpTensorDescriptor(OpTensorDescriptor opTensorDesc)
    int miopenOpTensor(
        Handle handle, OpTensorDescriptor opTensorDesc, void* alpha1,
        TensorDescriptor aDesc, void* A, void* alpha2,
        TensorDescriptor bDesc, void* B, void* beta,
        TensorDescriptor cDesc, void* C)

    # Tensor reductions
    int miopenCreateReduceTensorDescriptor(
        ReduceTensorDescriptor* reduceTensorDesc)
    int miopenSetReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc, ReduceTensorOp reduceTensorOp,
        DataType reduceTensorCompType, NanPropagation reduceTensorNanOpt,
        ReduceTensorIndices reduceTensorIndices,
        IndicesType reduceTensorIndicesType)
    int miopenGetReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc,
        ReduceTensorOp* reduceTensorOp, DataType* reduceTensorCompType,
        NanPropagation* reduceTensorNanOpt,
        ReduceTensorIndices* reduceTensorIndices,
        IndicesType* reduceTensorIndicesType)
    int miopenDestroyReduceTensorDescriptor(
        ReduceTensorDescriptor reduceTensorDesc)
    int miopenGetReductionIndicesSize(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc,
        TensorDescriptor aDesc, TensorDescriptor cDesc, size_t* sizeInBytes)
    int miopenGetReductionWorkspaceSize(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc,
        TensorDescriptor aDesc, TensorDescriptor cDesc, size_t* sizeInBytes)
    int miopenReduceTensor(
        Handle handle, ReduceTensorDescriptor reduceTensorDesc, void* indices,
        size_t indicesSizeInBytes, void* workspace,
        size_t workspaceSizeInBytes, void* alpha, TensorDescriptor aDesc,
        void* A, void* beta, TensorDescriptor cDesc, void* c)
    int miopenSetTensor(
        Handle handle, TensorDescriptor yDesc, void* y, void* valuePtr)
    int miopenScaleTensor(
        Handle handle, TensorDescriptor yDesc, void* y, void* alpha)

    # Filter manipulation
    int cudnnCreateFilterDescriptor(FilterDescriptor* filterDesc)
    int cudnnSetFilter4dDescriptor_v4(
        FilterDescriptor filterDesc, DataType dataType,
        TensorFormat format, int k, int c, int h, int w)
    int cudnnSetFilterNdDescriptor_v4(
        FilterDescriptor filterDesc, DataType dataType,
        TensorFormat format, int nbDims, const int filterDimA[])
    int cudnnGetFilterNdDescriptor_v4(
        FilterDescriptor wDesc, int nbDimsRequested, DataType* dataType,
        TensorFormat* format, int* nbDims, int filterDimA[])
    int cudnnDestroyFilterDescriptor(FilterDescriptor filterDesc)

    # Convolution
    int miopenCreateConvolutionDescriptor(ConvolutionDescriptor* convDesc)
    int cudnnSetConvolutionMathType(
        ConvolutionDescriptor convDesc, MathType mathType)
    int cudnnGetConvolutionMathType(
        ConvolutionDescriptor convDesc, MathType *mathType)
    int miopenSetConvolutionGroupCount(
        ConvolutionDescriptor convDesc, int groupCount)
    int miopenGetConvolutionGroupCount(
        ConvolutionDescriptor convDesc, int *groupCount)
    int cudnnSetConvolution2dDescriptor_v4(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int dilation_h, int dilation_w, ConvolutionMode mode)
    int cudnnSetConvolution2dDescriptor_v5(
        ConvolutionDescriptor convDesc, int pad_h, int pad_w, int u,
        int v, int dilation_h, int dilation_w, ConvolutionMode mode,
        DataType computeType)
    int cudnnSetConvolutionNdDescriptor_v3(
        ConvolutionDescriptor convDesc, int arrayLength, int* padA,
        int* filterStrideA, int* dilationA, ConvolutionMode mode,
        DataType dataType)
    int miopenDestroyConvolutionDescriptor(ConvolutionDescriptor conDesc)
    int cudnnFindConvolutionForwardAlgorithm(
        Handle handle, TensorDescriptor xDesc, FilterDescriptor wDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor yDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionFwdAlgoPerf* perfResults)
    int cudnnFindConvolutionForwardAlgorithmEx(
        Handle handle, TensorDescriptor xDesc, void* x,
        FilterDescriptor wDesc, void* w, ConvolutionDescriptor convDesc,
        TensorDescriptor yDesc, void* y, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionFwdAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnFindConvolutionForwardAlgorithmEx_v7(
        Handle handle, TensorDescriptor xDesc, void* x,
        FilterDescriptor wDesc, void* w, ConvolutionDescriptor convDesc,
        TensorDescriptor yDesc, void* y, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionFwdAlgoPerf_v7* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnGetConvolutionForwardAlgorithm_v6(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, ConvolutionFwdPreference preference,
        size_t memoryLimitInbytes, ConvolutionFwdAlgo* algo)
    int cudnnGetConvolutionForwardAlgorithm_v7(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionFwdAlgoPerf_v7* perfResults)
    int miopenConvolutionForwardGetWorkSpaceSize(
        Handle handle, TensorDescriptor srcDesc,
        FilterDescriptor filterDesc, ConvolutionDescriptor convDesc,
        TensorDescriptor destDesc,
        size_t* sizeInBytes)
    int cudnnConvolutionForward(
        Handle handle, void* alpha, TensorDescriptor srcDesc,
        void* srcData, FilterDescriptor filterDesc, void* filterData,
        ConvolutionDescriptor convDesc, ConvolutionFwdAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor destDesc, void* destData)
    int cudnnConvolutionBackwardBias(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor destDesc, void* destData)
    int cudnnFindConvolutionBackwardFilterAlgorithm(
        Handle handle, TensorDescriptor xDesc, TensorDescriptor dyDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor dwDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdFilterAlgoPerf* perfResults)
    int cudnnFindConvolutionBackwardFilterAlgorithmEx(
        Handle handle, TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        FilterDescriptor dwDesc, void* dw, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdFilterAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnFindConvolutionBackwardFilterAlgorithmEx_v7(
        Handle handle, TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        FilterDescriptor dwDesc, void* dw, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdFilterAlgoPerf_v7* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnGetConvolutionBackwardFilterAlgorithm_v6(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor filterDesc,
        ConvolutionBwdFilterPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdFilterAlgo* algo)
    int cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor gradDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdFilterAlgoPerf_v7* perfResults)
    int cudnnGetConvolutionBackwardFilterWorkspaceSize(
        Handle handle, TensorDescriptor srcDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor filterDesc,
        ConvolutionBwdFilterAlgo algo, size_t* sizeInBytes)
    int cudnnConvolutionBackwardFilter_v3(
        Handle handle, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, ConvolutionBwdFilterAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        FilterDescriptor gradDesc, void* gradData)
    int cudnnGetConvolutionBackwardDataAlgorithm_v6(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        ConvolutionBwdDataPreference preference,
        size_t memoryLimitInbytes, ConvolutionBwdDataAlgo* algo)
    int cudnnGetConvolutionBackwardDataAlgorithm_v7(
        Handle handle, TensorDescriptor filterDesc, TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor gradDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdDataAlgoPerf_v7* perfResults)
    int cudnnFindConvolutionBackwardDataAlgorithm(
        Handle handle, TensorDescriptor wDesc, TensorDescriptor dyDesc,
        ConvolutionDescriptor convDesc, FilterDescriptor dxDesc,
        int requestedAlgoCount, int* returnedAlgoCount,
        ConvolutionBwdDataAlgoPerf* perfResults)
    int cudnnFindConvolutionBackwardDataAlgorithmEx(
        Handle handle, FilterDescriptor wDesc, void* w,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        TensorDescriptor dxDesc, void* dx, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdDataAlgoPerf* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnFindConvolutionBackwardDataAlgorithmEx_v7(
        Handle handle, FilterDescriptor wDesc, void* w,
        TensorDescriptor dyDesc, void* dy, ConvolutionDescriptor convDesc,
        TensorDescriptor dxDesc, void* dx, int requestedAlgoCount,
        int* returnedAlgoCount, ConvolutionBwdDataAlgoPerf_v7* perfResults,
        void* workSpace, size_t workSpaceSizeInBytes)
    int miopenConvolutionBackwardDataGetWorkSpaceSize(
        Handle handle, FilterDescriptor filterDesc,
        TensorDescriptor diffDesc,
        ConvolutionDescriptor convDesc, TensorDescriptor gradDesc,
        size_t* sizeInBytes)
    int cudnnConvolutionBackwardData_v3(
        Handle handle, void* alpha,
        FilterDescriptor filterDesc, void* filterData,
        TensorDescriptor diffDesc, void* diffData,
        ConvolutionDescriptor convDesc, ConvolutionBwdDataAlgo algo,
        void* workSpace, size_t workSpaceSizeInBytes, void* beta,
        TensorDescriptor gradDesc, void* gradData)

    # Pooling
    int miopenCreatePoolingDescriptor(PoolingDescriptor* desc)
    int cudnnSetPooling2dDescriptor_v4(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        NanPropagation maxpoolingNanOpt, int windowHeight, int windowWidth,
        int verticalPadding, int horizontalPadding, int verticalStride,
        int horizontalStride)
    int cudnnSetPoolingNdDescriptor_v4(
        PoolingDescriptor poolingDesc, PoolingMode mode,
        NanPropagation maxpoolingNanOpt, int nbDims,
        int* windowDimA, int* paddingA, int* strideA)
    int miopenDestroyPoolingDescriptor(PoolingDescriptor poolingDesc)
    int cudnnPoolingForward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int cudnnPoolingBackward(
        Handle handle, PoolingDescriptor poolingDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)

    # Batch Normalization
    int miopenDeriveBNTensorDescriptor(
        TensorDescriptor derivedBnDesc, TensorDescriptor xDesc,
        BatchNormMode mode)
    int miopenBatchNormalizationForwardTraining(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, double exponentialAverageFactor,
        void* resultRunningMean, void* resultRunningVariance,
        double epsilon, void* resultSaveMean,
        void* resultSaveInvVariance)
    int miopenBatchNormalizationForwardInference(
        Handle handle, BatchNormMode mode,
        void* alpha, void* beta, TensorDescriptor xDesc,
        void* x, TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc, void* bnScale,
        void* bnBias, void* estimatedMean, void* estimatedVariance,
        double epsilon)
    int miopenBatchNormalizationBackward(
        Handle handle, BatchNormMode mode,
        void* alphaDataDiff, void* betaDataDiff,
        void* alphaParamDiff, void* betaParamDiff,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor dyDesc, void* dy,
        TensorDescriptor dxDesc, void* dx,
        TensorDescriptor dBnScaleBiasDesc, void* bnScale,
        void* dBnScaleResult, void* dBnBiasResult,
        double epsilon, void* savedMean, void* savedInvVariance)

    int cudnnBatchNormalizationForwardTrainingEx(
        Handle handle,
        BatchNormMode mode, BatchNormOps bnOps,
        void* alpha, void* beta,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor zDesc, void* z,
        TensorDescriptor yDesc, void* y,
        TensorDescriptor bnScaleBiasMeanVarDesc,
        void* bnScale, void* bnBias,
        double exponentialAverageFactor,
        void* resultRunningMean, void* resultRunningVariance,
        double epsilon,
        void* resultSaveMean, void* resultSaveInvVariance,
        ActivationDescriptor activationDesc,
        void* workspace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
        Handle handle,
        BatchNormMode mode, BatchNormOps bnOps,
        TensorDescriptor xDesc,
        TensorDescriptor zDesc,
        TensorDescriptor yDesc,
        TensorDescriptor bnScaleBiasMeanVarDesc,
        ActivationDescriptor activationDesc,
        size_t* sizeInBytes)
    int cudnnBatchNormalizationBackwardEx(
        Handle handle,
        BatchNormMode mode, BatchNormOps bnops,
        void* alphaDataDiff, void* betaDataDiff,
        void* alphaParamDiff, void* betaParamDiff,
        TensorDescriptor xDesc, void* x,
        TensorDescriptor yDesc, void* y,
        TensorDescriptor dyDesc, void* dy,
        TensorDescriptor dzDesc, void* dz,
        TensorDescriptor dxDesc, void* dx,
        TensorDescriptor dBnScaleBiasDesc,
        void* bnScaleData, void* bnBiasData,
        void* dBnScaleData, void* dBnBiasData,
        double epsilon,
        void* savedMean, void* savedInvVariance,
        ActivationDescriptor activationDesc,
        void* workspace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnGetBatchNormalizationBackwardExWorkspaceSize(
        Handle handle,
        BatchNormMode mode,
        BatchNormOps bnOps,
        TensorDescriptor xDesc,
        TensorDescriptor yDesc,
        TensorDescriptor dyDesc,
        TensorDescriptor dzDesc,
        TensorDescriptor dxDesc,
        TensorDescriptor dBnScaleBiasDesc,
        ActivationDescriptor activationDesc,
        size_t* sizeInBytes)
    int cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
        Handle handle,
        BatchNormMode mode,
        BatchNormOps bnOps,
        ActivationDescriptor activationDesc,
        TensorDescriptor xDesc,
        size_t* sizeInBytes)

    # Activation
    int miopenCreateActivationDescriptor(
        ActivationDescriptor* activationDesc)
    int cudnnSetActivationDescriptor(
        ActivationDescriptor activationDesc, ActivationMode mode,
        NanPropagation reluNanOpt, double reluCeiling)
    int miopenDestroyActivationDescriptor(
        ActivationDescriptor activationDesc)
    int miopenSoftmaxForward(
        Handle handle, 
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        void* beta, TensorDescriptor dstDesc, void* dstData)
    int miopenSoftmaxBackward(
        Handle handle, 
        void* alpha, TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)
    int cudnnActivationForward_v4(
        Handle handle, ActivationDescriptor activationDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData, void* beta,
        TensorDescriptor dstDesc, void* dstData)
    int cudnnActivationBackward_v4(
        Handle handle, ActivationDescriptor activationDesc, void* alpha,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor srcDiffDesc, void* srcDiffData,
        TensorDescriptor destDesc, void* destData, void* beta,
        TensorDescriptor destDiffDesc, void* destDiffData)

    # Dropout
    int miopenCreateDropoutDescriptor(DropoutDescriptor* desc)
    int miopenDestroyDropoutDescriptor(DropoutDescriptor dropoutDesc)
    int miopenDropoutGetStatesSize(Handle handle, size_t* sizeInBytes)
    int miopenDropoutGetReserveSpaceSize(
        TensorDescriptor xDesc, size_t* sizeInBytes)
    int cudnnSetDropoutDescriptor(
        DropoutDescriptor dropoutDesc, Handle handle, float dropout,
        void* states, size_t stateSizeInBytes, unsigned long long seed)
    int cudnnDropoutForward(
        Handle handle, DropoutDescriptor dropoutDesc,
        TensorDescriptor srcDesc, void* srcData,
        TensorDescriptor dstDesc, void* dstData,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnDropoutBackward(
        Handle handle, DropoutDescriptor dropoutDesc,
        TensorDescriptor dydesc, void* dy, TensorDescriptor dxdesc,
        void* dx, void* reserveSpace, size_t reserveSpaceSizeInBytes)

    # CTC
    int miopenCreateCTCLossDescriptor(CTCLossDescriptor* ctcLossDesc)
    int miopenDestroyCTCLossDescriptor(CTCLossDescriptor ctcLossDesc)
    int cudnnSetCTCLossDescriptor(
        CTCLossDescriptor ctcLossDesc, DataType dataType)
    int cudnnGetCTCLossDescriptor(
        CTCLossDescriptor ctcLossDesc, DataType* dataType)
    int miopenGetCTCLossWorkspaceSize(
        Handle handle, TensorDescriptor probsDesc,
        TensorDescriptor gradientsDesc, int* labels,
        int* labelLengths, int* inputLengths, CTCLossAlgo algo,
        CTCLossDescriptor ctcLossDesc, size_t* sizeInBytes)
    int miopenCTCLoss(
        Handle handle, TensorDescriptor probsDesc,
        void* probs, int* labels, int* labelLengths, int* inputLengths,
        void* costs, TensorDescriptor gradientsDesc, void* gradients,
        CTCLossAlgo algo, CTCLossDescriptor ctcLossDesc,
        void* workspace, size_t workSpaceSizeInBytes)
    # RNN
    int miopenCreateRNNDescriptor(RNNDescriptor* rnnDesc)
    int miopenDestroyRNNDescriptor(RNNDescriptor rnnDesc)
    int cudnnCreatePersistentRNNPlan(
        RNNDescriptor rnnDesc,
        const int minibatch, DataType dataType,
        PersistentRNNPlan* plan)
    int cudnnSetPersistentRNNPlan(
        RNNDescriptor rnnDesc, PersistentRNNPlan plan)
    int cudnnDestroyPersistentRNNPlan(PersistentRNNPlan plan)
    int cudnnSetRNNDescriptor_v5(
        RNNDescriptor rnnDesc, int hiddenSize,
        int numLayers, DropoutDescriptor dropoutDesc, RNNInputMode inputMode,
        DirectionMode direction, RNNMode mode, DataType dataType)
    int cudnnSetRNNDescriptor_v6(
        Handle handle, RNNDescriptor rnnDesc, int hiddenSize,
        int numLayers, DropoutDescriptor dropoutDesc, RNNInputMode inputMode,
        DirectionMode direction, RNNMode mode, RNNAlgo algo, DataType dataType)
    int cudnnSetRNNPaddingMode(
        RNNDescriptor rnnDesc, RNNPaddingMode paddingMode)
    int cudnnGetRNNPaddingMode(
        RNNDescriptor rnnDesc, RNNPaddingMode* paddingMode)
    int cudnnCreateRNNDataDescriptor(RNNDataDescriptor* RNNDataDesc)
    int cudnnDestroyRNNDataDescriptor(RNNDataDescriptor RNNDataDesc)
    int cudnnSetRNNDataDescriptor(
        RNNDataDescriptor RNNDataDesc, DataType dataType, RNNDataLayout layout,
        int maxSeqLength, int batchSize, int vectorSize,
        const int seqLengthArray[], void *paddingFill)
    int cudnnGetRNNDataDescriptor(
        RNNDataDescriptor RNNDataDesc, DataType* dataType,
        RNNDataLayout* layout, int* maxSeqLength, int* batchSize,
        int* vectorSize, int arrayLengthRequested, int seqLengthArray[],
        void* paddingFill)
    int miopenGetRNNWorkspaceSize(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, size_t* sizeInBytes)
    int miopenGetRNNTrainingReserveSize(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, size_t* sizeInBytes)
    int miopenGetRNNParamsSize(
        Handle handle, RNNDescriptor rnnDesc, TensorDescriptor xDesc,
        size_t* sizeInBytes, DataType dataType)
    int cudnnGetRNNLinLayerMatrixParams(
        Handle handle, RNNDescriptor rnnDesc, int layer,
        TensorDescriptor xDesc, FilterDescriptor wDesc, void* w,
        int linLayerID, FilterDescriptor linLayerMatDesc,
        void** linLayerMat)
    int cudnnGetRNNLinLayerBiasParams(
        Handle handle, RNNDescriptor rnnDesc, int layer,
        TensorDescriptor xDesc, FilterDescriptor wDesc, void* w,
        int linLayerID, FilterDescriptor linLayerBiasDesc,
        void** linLayerBias)
    int miopenRNNForwardInference(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc,
        void* x, TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc,
        void* cx, FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc,
        void* y, TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc,
        void* cy, void* workspace, size_t workSpaceSizeInBytes)
    int miopenRNNForwardTraining(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, void* x,
        TensorDescriptor hxDesc, void* hx, TensorDescriptor cxDesc, void* cx,
        FilterDescriptor wDesc, void* w, TensorDescriptor* yDesc, void* y,
        TensorDescriptor hyDesc, void* hy, TensorDescriptor cyDesc, void* cy,
        void* workspace, size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardData(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* yDesc, void* y,
        TensorDescriptor* dyDesc, void* dy,
        TensorDescriptor dhyDesc, void* dhy,
        TensorDescriptor dcyDesc, void* dcy,
        FilterDescriptor wDesc, void* w,
        TensorDescriptor hxDesc, void* hx,
        TensorDescriptor cxDesc, void* cx,
        TensorDescriptor* dxDesc, void* dx,
        TensorDescriptor dhxDesc, void* dhx,
        TensorDescriptor dcxDesc, void* dcx, void* workspace,
        size_t workSpaceSizeInBytes, void* reserveSpace,
        size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardWeights(
        Handle handle, RNNDescriptor rnnDesc, int seqLength,
        TensorDescriptor* xDesc, void* x, TensorDescriptor hxDesc, void* hx,
        TensorDescriptor* yDesc, void* y,
        void* workspace, size_t workSpaceSizeInBytes, FilterDescriptor dwDesc,
        void* dw, void* reserveSpace, size_t reserveSpaceSizeInBytes)

    int cudnnRNNForwardInferenceEx(
        Handle handle, RNNDescriptor rnnDesc,
        RNNDataDescriptor xDesc, const void* x,
        TensorDescriptor hxDesc, const void* hx,
        TensorDescriptor cxDesc, const void* cx,
        FilterDescriptor wDesc, const void* w,
        RNNDataDescriptor yDesc, void* y,
        TensorDescriptor hyDesc, void* hy,
        TensorDescriptor cyDesc, void* cy,
        RNNDataDescriptor kDesc, const void* keys,
        RNNDataDescriptor cDesc, void* cAttn,
        RNNDataDescriptor iDesc, void* iAttn,
        RNNDataDescriptor qDesc, void* queries,
        void* workSpace, size_t workSpaceSizeInBytes)
    int cudnnRNNForwardTrainingEx(
        Handle handle, RNNDescriptor rnnDesc,
        RNNDataDescriptor xDesc, const void* x,
        TensorDescriptor hxDesc, const void* hx,
        TensorDescriptor cxDesc, const void* cx,
        FilterDescriptor wDesc, const void* w,
        RNNDataDescriptor yDesc, void* y,
        TensorDescriptor hyDesc, void* hy,
        TensorDescriptor cyDesc, void* cy,
        RNNDataDescriptor kDesc, const void* keys,
        RNNDataDescriptor cDesc, void* cAttn,
        RNNDataDescriptor iDesc, void* iAttn,
        RNNDataDescriptor qDesc, void* queries,
        void* workSpace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardDataEx(
        Handle handle, RNNDescriptor rnnDesc,
        RNNDataDescriptor yDesc, const void* y,
        RNNDataDescriptor dyDesc, const void* dy,
        RNNDataDescriptor dcDesc, const void* dcAttn,
        TensorDescriptor dhyDesc, const void* dhy,
        TensorDescriptor dcyDesc, const void* dcy,
        FilterDescriptor wDesc, const void* w,
        TensorDescriptor hxDesc, const void* hx,
        TensorDescriptor cxDesc, const void* cx,
        RNNDataDescriptor dxDesc, void* dx,
        TensorDescriptor dhxDesc, void* dhx,
        TensorDescriptor dcxDesc, void* dcx,
        RNNDataDescriptor dkDesc, void* dkeys,
        void* workSpace, size_t workSpaceSizeInBytes,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)
    int cudnnRNNBackwardWeightsEx(
        Handle handle, RNNDescriptor rnnDesc,
        RNNDataDescriptor xDesc, const void* x,
        TensorDescriptor hxDesc, const void* hx,
        RNNDataDescriptor yDesc, const void* y,
        void* workSpace, size_t workSpaceSizeInBytes,
        FilterDescriptor dwDesc, void* dw,
        void* reserveSpace, size_t reserveSpaceSizeInBytes)

    # Spatial Transformer
    int cudnnCreateSpatialTransformerDescriptor(
        SpatialTransformerDescriptor* stDesc)
    int cudnnDestroySpatialTransformerDescriptor(
        SpatialTransformerDescriptor stDesc)
    int cudnnSetSpatialTransformerNdDescriptor(
        SpatialTransformerDescriptor stDesc, SamplerType samplerType,
        DataType dataType, int nbDims, int dimA[])
    int cudnnSpatialTfGridGeneratorForward(
        Handle handle, SpatialTransformerDescriptor stDesc,
        void* theta, void* grid)
    int cudnnSpatialTfGridGeneratorBackward(
        Handle handle, SpatialTransformerDescriptor stDesc,
        void* dgrid, void* dtheta)
    int cudnnSpatialTfSamplerForward(
        Handle handle, SpatialTransformerDescriptor stDesc,
        void* alpha, TensorDescriptor xDesc, void* x,
        void* grid, void* beta, TensorDescriptor yDesc, void* y)
    int cudnnSpatialTfSamplerBackward(
        Handle handle, SpatialTransformerDescriptor stDesc,
        void* alpha, TensorDescriptor xDesc, void* x, void* beta,
        TensorDescriptor dxDesc, void* dx, void* alphaDgrid,
        TensorDescriptor dyDesc, void* dy, void* grid,
        void* betaDgrid, void* dgrid)

    # Fused Ops
    int cudnnCreateFusedOpsConstParamPack(
        FusedOpsConstParamPack* constPack, int ops)
    int cudnnDestroyFusedOpsConstParamPack(FusedOpsConstParamPack constPack)
    int cudnnSetFusedOpsConstParamPackAttribute(
        FusedOpsConstParamPack constPack, FusedOpsConstParamLabel paramLabel,
        const void *param)
    int cudnnGetFusedOpsConstParamPackAttribute(
        const FusedOpsConstParamPack constPack,
        FusedOpsConstParamLabel paramLabel, void *param, int *isNULL)
    int cudnnCreateFusedOpsVariantParamPack(
        FusedOpsVariantParamPack *varPack, FusedOps ops)
    int cudnnDestroyFusedOpsVariantParamPack(FusedOpsVariantParamPack varPack)
    int cudnnSetFusedOpsVariantParamPackAttribute(
        FusedOpsVariantParamPack varPack, FusedOpsVariantParamLabel paramLabel,
        void *ptr)
    int cudnnGetFusedOpsVariantParamPackAttribute(
        const FusedOpsVariantParamPack varPack,
        FusedOpsVariantParamLabel paramLabel, void *ptr)
    int cudnnCreateFusedOpsPlan(FusedOpsPlan *plan, FusedOps ops)
    int cudnnDestroyFusedOpsPlan(FusedOpsPlan plan)
    int cudnnMakeFusedOpsPlan(
        Handle handle, FusedOpsPlan plan,
        const FusedOpsConstParamPack constPack, size_t *workspaceSizeInBytes)
    int cudnnFusedOpsExecute(
        Handle handle, const FusedOpsPlan plan,
        FusedOpsVariantParamPack varPack)

    # Build-time version
    int CUDNN_VERSION

    # Constants
    double _CUDNN_BN_MIN_EPSILON 'CUDNN_BN_MIN_EPSILON'

