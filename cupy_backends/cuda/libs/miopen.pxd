from libc.stdint cimport intptr_t


###############################################################################
# Enum
###############################################################################
IF CUPY_HIP_VERSION != 0:
    cpdef enum:
        miopenFloat = 1
        miopenDouble = 6
        miopenHalf  = 0
		
        miopenConvolutionFwdAlgoGEMM = 0
        miopenConvolutionFwdAlgoDirect = 1
        miopenConvolutionFwdAlgoFFT = 2
        miopenConvolutionFwdAlgoWinograd = 3
        miopenConvolutionFwdAlgoImplicitGEMM = 5

        miopenPoolingMax = 0
        miopenPoolingAverage = 1
        miopenPoolingAverageInclusive = 2

        miopenActivationPASTHRU = 0
        miopenActivationTANH = 2
        miopenActivationRELU = 3
        miopenActivationCLIPPEDRELU = 7
        miopenActivationELU = 9
		
        miopenRNNDataSeqMajorNotPadded = 1
        miopenRNNDataSeqMajorPadded = 2
        miopenRNNDataBatchMajorPadded = 3
		
        MIOPEN_NOT_PROPAGATE_NAN = 0
        MIOPEN_PROPAGATE_NAN = 1

        miopenTensorNCHW = 0
        miopenTensorNHWC = 1

        miopenTensorOpAdd = 0
        miopenTensorOpMul = 1
        miopenTensorOpMin = 2
        miopenTensorOpMax = 3
	    
        MIOPEN_REDUCE_TENSOR_ADD = 0
        MIOPEN_REDUCE_TENSOR_MUL = 1
        MIOPEN_REDUCE_TENSOR_MIN = 2
        MIOPEN_REDUCE_TENSOR_MAX = 3
        MIOPEN_REDUCE_TENSOR_AMAX = 4
        MIOPEN_REDUCE_TENSOR_AVG = 5
        MIOPEN_REDUCE_TENSOR_NORM1 = 6
        MIOPEN_REDUCE_TENSOR_NORM2 = 7
	    
        MIOPEN_REDUCE_TENSOR_NO_INDICES = 0
        MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES = 1 
	    
        MIOPEN_32BIT_INDICES = 0
        MIOPEN_64BIT_INDICES = 1
        MIOPEN_16BIT_INDICES = 2
        MIOPEN_8BIT_INDICES  = 3
	    
        miopenConvolution = 0
        miopenTranspose   = 1
	    
        MIOPEN_SOFTMAX_FAST = 0
        MIOPEN_SOFTMAX_ACCURATE = 1
        MIOPEN_SOFTMAX_LOG = 2
	    
        MIOPEN_SOFTMAX_MODE_INSTANCE = 0
        MIOPEN_SOFTMAX_MODE_CHANNEL = 1
	    
        miopenBNPerActivation = 0
        miopenBNSpatial       = 1
	    
        MIOPEN_CTC_LOSS_ALGO_DETERMINISTIC = 0
	    
        miopenRNNRELU = 0
        miopenRNNTANH = 1
        miopenLSTM    = 2
        miopenGRU     = 3
	    
        miopenRNNunidirection = 0
        miopenRNNbidirection  = 1
	    
        miopenRNNIONotPadded = 0
        miopenRNNIOWithPadding = 1
	    
        miopenRNNlinear = 0
        miopenRNNskip = 1
	    
        miopenStatusSuccess  = 0

        MIOPEN_RNG_PSEUDO_XORWOW = 0

IF CUPY_HIP_VERSION == 0:
    ###############################################################################
    # Class
    ###############################################################################
    
    cdef class CuDNNAlgoPerf:
        cdef:
            int algo
            int status
            float time
            size_t memory
            int determinism
            int mathType
    ###############################################################################
    # Version
    ###############################################################################
    
    cpdef size_t getVersion() except? 0
    
    ###############################################################################
    # Runtime error checking
    ###############################################################################
    cpdef queryRuntimeError(intptr_t handle, int mode)
    
    ###############################################################################
    # Initialization and CUDA cooperation
    ###############################################################################
    
    cpdef intptr_t create() except? 0
    cpdef destroy(intptr_t handle)
    cpdef setStream(intptr_t handle, size_t stream)
    cpdef size_t getStream(intptr_t handle) except? 0
    
    
    ###############################################################################
    # Tensor manipulation
    ###############################################################################
    
    cpdef size_t createTensorDescriptor() except? 0
    cpdef setTensor4dDescriptor(size_t tensorDesc, int format, int dataType,
                                int n, int c, int h, int w)
    cpdef setTensor4dDescriptorEx(size_t tensorDesc, int dataType,
                                  int n, int c, int h, int w, int nStride,
                                  int cStride, int hStride, int wStride)
    cpdef tuple getTensor4dDescriptor(size_t tensorDesc)
    cpdef setTensorNdDescriptor(size_t tensorDesc, int dataType, int nbDims,
                                size_t dimA, size_t strideA)
    cpdef destroyTensorDescriptor(size_t tensorDesc)
    cpdef addTensor_v3(intptr_t handle, size_t alpha, size_t bDesc,
                       size_t b, size_t beta, size_t yDesc, size_t y)
    
    
    ###############################################################################
    # Tensor operations
    ###############################################################################
    
    cpdef size_t createOpTensorDescriptor() except? 0
    cpdef setOpTensorDescriptor(size_t opTensorDesc, int opTensorOp,
                                int opTensorCompType, int opTensorNanOpt)
    cpdef getOpTensorDescriptor(size_t opTensorDesc)
    cpdef destroyOpTensorDescriptor(size_t opTensorDesc)
    cpdef opTensor(intptr_t handle, size_t opTensorDesc, size_t alpha1,
                   size_t aDesc, size_t A, size_t alpha2, size_t bDesc,
                   size_t B, size_t beta, size_t cDesc, size_t C)
    
    
    ###############################################################################
    # Tensor reductions
    ###############################################################################
    
    cpdef size_t createReduceTensorDescriptor() except? 0
    cpdef setReduceTensorDescriptor(
        size_t reduceTensorDesc, int reduceTensorOp,
        int reduceTensorCompType, int reduceTensorNanOpt,
        int reduceTensorIndices, int reduceTensorIndicesType)
    cpdef getReduceTensorDescriptor(size_t reduceTensorDesc)
    cpdef destroyReduceTensorDescriptor(size_t reduceTensorDesc)
    cpdef size_t getReductionIndicesSize(
        intptr_t handle, size_t reduceTensorDesc, size_t aDesc,
        size_t cDesc) except? 0
    cpdef size_t getReductionWorkspaceSize(
        intptr_t handle, size_t reduceTensorDesc, size_t aDesc,
        size_t cDesc) except? 0
    cpdef reduceTensor(
        intptr_t handle, size_t reduceTensorDesc, size_t indices,
        size_t indicesSizeInBytes, size_t workspace,
        size_t workspaceSizeInBytes, size_t alpha, size_t aDesc,
        size_t A, size_t beta, size_t cDesc, size_t C)
    cpdef setTensor(intptr_t handle, size_t yDesc, size_t y, size_t valuePtr)
    cpdef scaleTensor(intptr_t handle, size_t yDesc, size_t y, size_t alpha)
    
    
    ###############################################################################
    # Filter manipulation
    ###############################################################################
    
    cpdef size_t createFilterDescriptor() except? 0
    cpdef setFilter4dDescriptor_v4(
        size_t filterDesc, int dataType, int format, int k, int c, int h, int w)
    cpdef setFilterNdDescriptor_v4(
        size_t filterDesc, int dataType, int format, int nbDims, size_t filterDimA)
    cpdef getFilterNdDescriptor(size_t wDesc, int nbDimsRequested)
    cpdef destroyFilterDescriptor(size_t filterDesc)
    
    
    ###############################################################################
    # Convolution
    ###############################################################################
    
    cpdef size_t createConvolutionDescriptor() except? 0
    cpdef setConvolutionMathType(
        size_t convDesc, size_t mathType)
    cpdef size_t getConvolutionMathType(size_t convDesc) except? 0
    cpdef setConvolutionGroupCount(
        size_t convDesc, int groupCount)
    cpdef int getConvolutionGroupCount(size_t convDesc) except? -1
    cpdef setConvolution2dDescriptor_v4(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h,
        int dilation_w, int mode)
    cpdef setConvolution2dDescriptor_v5(
        size_t convDesc, int pad_h, int pad_w, int u, int v, int dilation_h,
        int dilation_w, int mode, size_t computeType)
    cpdef setConvolutionNdDescriptor_v3(
        size_t convDesc, int arrayLength, size_t padA, size_t filterStrideA,
        size_t dilationA, int mode, int dataType)
    cpdef destroyConvolutionDescriptor(size_t convDesc)
    cpdef findConvolutionForwardAlgorithm(
        intptr_t handle, size_t xDesc, size_t wDesc, size_t convDesc, size_t yDesc,
        int requestedAlgoCount)
    cpdef list findConvolutionForwardAlgorithmEx(
        intptr_t handle, size_t xDesc, size_t x, size_t wDesc, size_t w,
        size_t convDesc, size_t yDesc, size_t y, int requestedAlgoCount,
        size_t workSpace, size_t workSpaceSizeInBytes)
    cpdef list findConvolutionForwardAlgorithmEx_v7(
        intptr_t handle, size_t xDesc, size_t x, size_t wDesc, size_t w,
        size_t convDesc, size_t yDesc, size_t y, int requestedAlgoCount,
        size_t workSpace, size_t workSpaceSizeInBytes)
    cpdef int getConvolutionForwardAlgorithm_v6(
        intptr_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int preference, size_t memoryLimitInbytes) except? -1
    cpdef list getConvolutionForwardAlgorithm_v7(
        intptr_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int requestedAlgoCount)
    cpdef Py_ssize_t getConvolutionForwardWorkspaceSize(
        intptr_t handle, size_t srcDesc, size_t filterDesc, size_t convDesc,
        size_t destDesc, int algo) except? -1
    cpdef convolutionForward(
        intptr_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t filterDesc, size_t filterData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t destDesc, size_t destData)
    cpdef convolutionBackwardBias(
        intptr_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t beta, size_t destDesc, size_t destData)
    cpdef findConvolutionBackwardFilterAlgorithm(
        intptr_t handle, size_t xDesc, size_t dyDesc, size_t convDesc,
        size_t dwDesc, int requestedAlgoCount)
    cpdef list findConvolutionBackwardFilterAlgorithmEx(
        intptr_t handle, size_t xDesc, size_t x, size_t dyDesc, size_t dy,
        size_t convDesc, size_t dwDesc, size_t dw, int requestedAlgoCount,
        size_t workSpace, size_t workSpaceSizeInBytes)
    cpdef list findConvolutionBackwardFilterAlgorithmEx_v7(
        intptr_t handle, size_t xDesc, size_t x, size_t dyDesc, size_t dy,
        size_t convDesc, size_t dwDesc, size_t dw, int requestedAlgoCount,
        size_t workSpace, size_t workSpaceSizeInBytes)
    cpdef int getConvolutionBackwardFilterAlgorithm_v6(
        intptr_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t filterDesc, int preference, size_t memoryLimitInbytes) except? -1
    cpdef list getConvolutionBackwardFilterAlgorithm_v7(
        intptr_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t gradDesc, int requestedAlgoCount)
    cpdef Py_ssize_t getConvolutionBackwardFilterWorkspaceSize(
        intptr_t handle, size_t srcDesc, size_t diffDesc, size_t convDesc,
        size_t filterDesc, int algo) except? -1
    cpdef convolutionBackwardFilter_v3(
        intptr_t handle, size_t alpha, size_t srcDesc, size_t srcData,
        size_t diffDesc, size_t diffData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t gradDesc, size_t gradData)
    cpdef findConvolutionBackwardDataAlgorithm(
        intptr_t handle, size_t wDesc, size_t dyDesc, size_t convDesc,
        size_t dxDesc, int requestedAlgoCount)
    cpdef list findConvolutionBackwardDataAlgorithmEx(
        intptr_t handle, size_t wDesc, size_t w, size_t dyDesc, size_t dy,
        size_t convDesc, size_t dxDesc, size_t dx,
        int requestedAlgoCount, size_t workSpace, size_t workSpaceSizeInBytes)
    cpdef list findConvolutionBackwardDataAlgorithmEx_v7(
        intptr_t handle, size_t wDesc, size_t w, size_t dyDesc, size_t dy,
        size_t convDesc, size_t dxDesc, size_t dx,
        int requestedAlgoCount, size_t workSpace, size_t workSpaceSizeInBytes)
    cpdef int getConvolutionBackwardDataAlgorithm_v6(
        intptr_t handle, size_t filterDesc, size_t diffDesc, size_t convDesc,
        size_t gradDesc, size_t preference,
        size_t memoryLimitInbytes) except? -1
    cpdef list getConvolutionBackwardDataAlgorithm_v7(
        intptr_t handle, size_t filterDesc, size_t diffDesc, size_t convDesc,
        size_t gradDesc, int requestedAlgoCount)
    cpdef Py_ssize_t getConvolutionBackwardDataWorkspaceSize(
        intptr_t handle, size_t filterDesc, size_t diffDesc, size_t convDesc,
        size_t gradDesc, int algo) except? -1
    cpdef convolutionBackwardData_v3(
        intptr_t handle, size_t alpha, size_t filterDesc, size_t filterData,
        size_t diffDesc, size_t diffData, size_t convDesc, int algo,
        size_t workSpace, size_t workSpaceSizeInBytes, size_t beta,
        size_t gradDesc, size_t gradData)
    
    
    ###############################################################################
    # Pooling
    ###############################################################################
    
    cpdef size_t createPoolingDescriptor() except? 0
    cpdef setPooling2dDescriptor_v4(
        size_t poolingDesc, int mode, int maxpoolingNanOpt, int windowHeight,
        int windowWidth, int verticalPadding, int horizontalPadding,
        int verticalStride, int horizontalStride)
    cpdef setPoolingNdDescriptor_v4(
        size_t poolingDesc, int mode, int maxpoolingNanOpt, int nbDims,
        size_t windowDimA, size_t paddingA, size_t strideA)
    cpdef destroyPoolingDescriptor(size_t poolingDesc)
    cpdef poolingForward(
        intptr_t handle, size_t poolingDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData)
    cpdef poolingBackward(
        intptr_t handle, size_t poolingDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData,
        size_t destDesc, size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData)
    
    ###############################################################################
    # Batch Normalization
    ###############################################################################
    
    cpdef deriveBNTensorDescriptor(
        size_t derivedBnDesc, size_t xDesc, int mode)
    
    cpdef batchNormalizationForwardTraining(
        intptr_t handle, int mode,
        size_t alpha, size_t beta, size_t xDesc,
        size_t x, size_t yDesc, size_t y,
        size_t bnScaleBiasMeanVarDesc, size_t bnScale,
        size_t bnBias, double exponentialAverageFactor,
        size_t resultRunningMean, size_t resultRunningVariance,
        double epsilon, size_t resultSaveMean, size_t resultSaveInvVariance)
    
    cpdef batchNormalizationForwardInference(
        intptr_t handle, int mode,
        size_t alpha, size_t beta, size_t xDesc,
        size_t x, size_t yDesc, size_t y,
        size_t bnScaleBiasMeanVarDesc, size_t bnScale,
        size_t bnBias, size_t estimatedMean, size_t estimatedVariance,
        double epsilon)
    
    cpdef batchNormalizationBackward(
        intptr_t handle, int mode,
        size_t alphaDataDiff, size_t betaDataDiff,
        size_t alphaParamDiff, size_t betaParamDiff,
        size_t xDesc, size_t x, size_t dyDesc,
        size_t dy, size_t dxDesc, size_t dx,
        size_t dBnScaleBiasDesc, size_t bnScale,
        size_t dBnScaleResult, size_t dBnBiasResult,
        double epsilon, size_t savedMean, size_t savedInvVariance)
    
    cpdef batchNormalizationForwardTrainingEx(
        intptr_t handle, int mode, int bnOps,
        size_t alpha, size_t beta,
        size_t xDesc, size_t x,
        size_t zDesc, size_t z,
        size_t yDesc, size_t y,
        size_t bnScaleBiasMeanVarDesc,
        size_t bnScale, size_t bnBias,
        double exponentialAverageFactor,
        size_t resultRunningMean, size_t resultRunningVariance,
        double epsilon, size_t resultSaveMean, size_t resultSaveInvVariance,
        size_t activationDesc,
        size_t workSpace, size_t workSpaceSizeInBytes,
        size_t reserveSpace, size_t reserveSpaceSizeInBytes)
    
    cpdef size_t getBatchNormalizationForwardTrainingExWorkspaceSize(
        intptr_t handle, int mode, int bnOps,
        size_t xDesc,
        size_t zDesc,
        size_t yDesc,
        size_t bnScaleBiasMeanVarDesc,
        size_t activationDesc) except? 0
    
    cpdef batchNormalizationBackwardEx(
        intptr_t handle, int mode, int bnops,
        size_t alphaDataDiff, size_t betaDataDiff,
        size_t alphaParamDiff, size_t betaParamDiff,
        size_t xDesc, size_t x,
        size_t yDesc, size_t y,
        size_t dyDesc, size_t dy,
        size_t dzDesc, size_t dz,
        size_t dxDesc, size_t dx,
        size_t dBnScaleBiasDesc,
        size_t bnScaleData, size_t bnBiasData,
        size_t dBnScaleData, size_t dBnBiasData,
        double epsilon,
        size_t savedMean, size_t savedInvVariance,
        size_t activationDesc,
        size_t workSpace, size_t workSpaceSizeInBytes,
        size_t reserveSpace, size_t reserveSpaceSizeInBytes)
    
    cpdef size_t getBatchNormalizationBackwardExWorkspaceSize(
        intptr_t handle, int mode, int bnOps,
        size_t xDesc,
        size_t yDesc,
        size_t dyDesc,
        size_t dzDesc,
        size_t dxDesc,
        size_t dBnScaleBiasDesc,
        size_t activationDesc) except? 0
    
    cpdef size_t getBatchNormalizationTrainingExReserveSpaceSize(
        intptr_t handle, int mode, int bnOps,
        size_t activationDesc,
        size_t xDesc) except? 0
    
    
    ###############################################################################
    # Activation
    ###############################################################################
    
    cpdef size_t createActivationDescriptor() except? 0
    cpdef setActivationDescriptor(
        size_t activationDesc, int mode, int reluNanOpt, double reluCeiling)
    cpdef destroyActivationDescriptor(size_t activationDesc)
    cpdef softmaxForward(
        intptr_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData)
    cpdef softmaxBackward(
        intptr_t handle, int algorithm, int mode, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData, size_t beta,
        size_t destDiffDesc, size_t destDiffData)
    cpdef activationForward_v4(
        intptr_t handle, size_t activationDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t beta, size_t dstDesc, size_t dstData)
    cpdef activationBackward_v4(
        intptr_t handle, size_t activationDesc, size_t alpha, size_t srcDesc,
        size_t srcData, size_t srcDiffDesc, size_t srcDiffData,
        size_t destDesc, size_t destData, size_t beta, size_t destDiffDesc,
        size_t destDiffData)
    
    
    ###############################################################################
    # Dropout
    ###############################################################################
    cpdef size_t createDropoutDescriptor() except? 0
    cpdef destroyDropoutDescriptor(size_t dropoutDesc)
    cpdef Py_ssize_t dropoutGetStatesSize(intptr_t handle) except? -1
    cpdef setDropoutDescriptor(
        size_t dropoutDesc, intptr_t handle, float dropout,
        size_t states, size_t stateSizeInBytes, unsigned long long seed)
    cpdef size_t getDropoutReserveSpaceSize(size_t xDesc) except? 0
    cpdef dropoutForward(
        intptr_t handle, size_t dropoutDesc,
        size_t srcDesc, size_t srcData,
        size_t dstDesc, size_t dstData,
        size_t reserveSpace, size_t reserveSpaceSizeInBytes)
    cpdef dropoutBackward(
        intptr_t handle, size_t dropoutDesc,
        size_t dyDesc, size_t dyData,
        size_t dxtDesc, size_t dxData,
        size_t reserveSpace, size_t reserveSpaceSizeInBytes)
    
    
    ###############################################################################
    # CTC
    ###############################################################################
    
    cpdef size_t createCTCLossDescriptor() except? 0
    cpdef destroyCTCLossDescriptor(size_t ctcLossDesc)
    cpdef setCTCLossDescriptor(size_t ctcLossDesc, int dataType)
    cpdef getCTCLossDescriptor(size_t ctcLossDesc)
    cpdef size_t getCTCLossWorkspaceSize(
        intptr_t handle, size_t probsDesc, size_t gradientsDesc,
        size_t labels, size_t labelLengths, size_t inputLengths,
        int algo, size_t ctcLossDesc) except? 0
    cpdef CTCLoss(
        intptr_t handle, size_t probsDesc,
        size_t probs, size_t labels, size_t labelLengths, size_t inputLengths,
        size_t costs, size_t gradientsDesc, size_t gradients, int algo,
        size_t ctcLossDesc, size_t workspace, size_t workSpaceSizeInBytes)
    
    
    ###############################################################################
    # RNN
    ###############################################################################
    
    cpdef size_t createRNNDescriptor() except? 0
    cpdef destroyRNNDescriptor(size_t rnnDesc)
    cpdef size_t createPersistentRNNPlan(
        size_t rnnDesc, int minibatch, int dataType) except? 0
    cpdef setPersistentRNNPlan(size_t rnnDesc, size_t plan)
    cpdef destroyPersistentRNNPlan(size_t plan)
    cpdef setRNNDescriptor_v5(
        size_t rnnDesc, int hiddenSize, int numLayers,
        size_t dropoutDesc, int inputMode, int direction, int mode,
        int dataType)
    cpdef setRNNDescriptor_v6(
        intptr_t handle, size_t rnnDesc, int hiddenSize, int numLayers,
        size_t dropoutDesc, int inputMode, int direction, int mode,
        int algo, int dataType)
    cpdef setRNNPaddingMode(size_t rnnDesc, int paddingMode)
    cpdef getRNNPaddingMode(size_t rnnDesc)
    cpdef size_t createRNNDataDescriptor() except? 0
    cpdef destroyRNNDataDescriptor(size_t RNNDataDesc)
    cpdef setRNNDataDescriptor(
        size_t RNNDataDesc, int dataType, size_t layout,
        int maxSeqLength, int batchSize, int vectorSize,
        size_t seqLengthArray, size_t paddingFill)
    cpdef getRNNDataDescriptor(
        size_t RNNDataDesc, size_t dataType,
        size_t layout, size_t maxSeqLength, size_t batchSize,
        size_t vectorSize, int arrayLengthRequested, size_t seqLengthArray,
        size_t paddingFill)
    cpdef getRNNWorkspaceSize(
        intptr_t handle, size_t rnnDesc, int seqLength, size_t xDesc)
    cpdef getRNNTrainingReserveSize(
        intptr_t handle, size_t rnnDesc, int seqLength, size_t xDesc)
    cpdef getRNNParamsSize(
        intptr_t handle, size_t rnnDesc, size_t xDesc, int dataType)
    cpdef getRNNLinLayerMatrixParams(
        intptr_t handle, size_t rnnDesc, int layer, size_t xDesc, size_t wDesc,
        size_t w, int linLayerID, size_t linLayerMatDesc, size_t linLayerMat)
    cpdef getRNNLinLayerBiasParams(
        intptr_t handle, size_t rnnDesc, int layer, size_t xDesc, size_t wDesc,
        size_t w, int linLayerID, size_t linLayerBiasDesc,
        size_t linLayerBias)
    cpdef RNNForwardInference(
        intptr_t handle, size_t rnnDesc, int seqLength, size_t xDesc,
        size_t x, size_t hxDesc, size_t hx, size_t cxDesc,
        size_t cx, size_t wDesc, size_t w, size_t yDesc,
        size_t y, size_t hyDesc, size_t hy, size_t cyDesc,
        size_t cy, size_t workspace, size_t workSpaceSizeInBytes)
    cpdef RNNForwardTraining(
        intptr_t handle, size_t rnnDesc, int seqLength, size_t xDesc, size_t x,
        size_t hxDesc, size_t hx, size_t cxDesc, size_t cx,
        size_t wDesc, size_t w, size_t yDesc, size_t y,
        size_t hyDesc, size_t hy, size_t cyDesc, size_t cy,
        size_t workspace, size_t workSpaceSizeInBytes, size_t reserveSpace,
        size_t reserveSpaceSizeInBytes)
    cpdef RNNBackwardData(
        intptr_t handle, size_t rnnDesc, int seqLength, size_t yDesc, size_t y,
        size_t dyDesc, size_t dy, size_t dhyDesc, size_t dhy,
        size_t dcyDesc, size_t dcy, size_t wDesc, size_t w,
        size_t hxDesc, size_t hx, size_t cxDesc, size_t cx,
        size_t dxDesc, size_t dx, size_t dhxDesc, size_t dhx,
        size_t dcxDesc, size_t dcx, size_t workspace,
        size_t workSpaceSizeInBytes, size_t reserveSpace,
        size_t reserveSpaceSizeInBytes)
    cpdef RNNBackwardWeights(
        intptr_t handle, size_t rnnDesc, int seqLength, size_t xDesc, size_t x,
        size_t hxDesc, size_t hx, size_t yDesc, size_t y,
        size_t workspace, size_t workSpaceSizeInBytes, size_t dwDesc,
        size_t dw, size_t reserveSpace, size_t reserveSpaceSizeInBytes)
    cpdef RNNForwardInferenceEx(
        intptr_t handle, size_t rnnDesc, size_t xDesc, size_t x, size_t hxDesc,
        size_t hx, size_t cxDesc, size_t cx, size_t wDesc, size_t w,
        size_t yDesc, size_t y, size_t hyDesc, size_t hy, size_t cyDesc,
        size_t cy, size_t kDesc, size_t keys, size_t cDesc, size_t cAttn,
        size_t iDesc, size_t iAttn, size_t qDesc, size_t queries,
        size_t workSpace, size_t workSpaceSizeInBytes)
    cpdef RNNForwardTrainingEx(
        intptr_t handle, size_t rnnDesc, size_t xDesc, size_t x, size_t hxDesc,
        size_t hx, size_t cxDesc, size_t cx, size_t wDesc, size_t w,
        size_t yDesc, size_t y, size_t hyDesc, size_t hy, size_t cyDesc,
        size_t cy, size_t kDesc, size_t keys, size_t cDesc, size_t cAttn,
        size_t iDesc, size_t iAttn, size_t qDesc, size_t queries,
        size_t workSpace, size_t workSpaceSizeInBytes,
        size_t reserveSpace, size_t reserveSpaceSizeInBytes)
    cpdef RNNBackwardDataEx(
        intptr_t handle, size_t rnnDesc, size_t yDesc, size_t y, size_t dyDesc,
        size_t dy, size_t dcDesc, size_t dcAttn, size_t dhyDesc, size_t dhy,
        size_t dcyDesc, size_t dcy, size_t wDesc, size_t w, size_t hxDesc,
        size_t hx, size_t cxDesc, size_t cx, size_t dxDesc, size_t dx,
        size_t dhxDesc, size_t dhx, size_t dcxDesc, size_t dcx,
        size_t dkDesc, size_t dkeys,
        size_t workSpace, size_t workSpaceSizeInBytes,
        size_t reserveSpace, size_t reserveSpaceSizeInBytes)
    cpdef RNNBackwardWeightsEx(
        intptr_t handle, size_t rnnDesc, size_t xDesc, size_t x,
        size_t hxDesc, size_t hx, size_t yDesc, size_t y,
        size_t workSpace, size_t workSpaceSizeInBytes,
        size_t dwDesc, size_t dw,
        size_t reserveSpace, size_t reserveSpaceSizeInBytes)
    
    
    ###############################################################################
    # Spatial Transformer
    ###############################################################################
    
    cpdef size_t createSpatialTransformerDescriptor() except? 0
    cpdef destroySpatialTransformerDescriptor(size_t stDesc)
    cpdef setSpatialTransformerDescriptor(
        size_t stDesc, size_t samplerType, int dataType,
        int nbDims, size_t dimA)
    cpdef spatialTfGridGeneratorForward(
        intptr_t handle, size_t stDesc, size_t theta, size_t grid)
    cpdef spatialTfGridGeneratorBackward(
        intptr_t handle, size_t stDesc, size_t dgrid, size_t dtheta)
    cpdef spatialTfSamplerForward(
        intptr_t handle, size_t stDesc, size_t alpha, size_t xDesc,
        size_t x, size_t grid, size_t beta, size_t yDesc, size_t y)
    cpdef spatialTfSamplerBackward(
        intptr_t handle, size_t stDesc, size_t alpha, size_t xDesc,
        size_t x, size_t beta, size_t dxDesc, size_t dx, size_t alphaDgrid,
        size_t dyDesc, size_t dy, size_t grid, size_t betaDgrid, size_t dgrid)
    
    ###############################################################################
    # Fused Ops
    ###############################################################################
    
    cpdef createFusedOpsConstParamPack(int ops)
    cpdef destroyFusedOpsConstParamPack(size_t constPack)
    cpdef setFusedOpsConstParamPackAttribute(size_t constPack, int paramLabel,
                                             size_t param)
    cpdef getFusedOpsConstParamPackAttribute(size_t constPack, int paramLabel,
                                             size_t param)
    cpdef createFusedOpsVariantParamPack(int ops)
    cpdef destroyFusedOpsVariantParamPack(size_t varPack)
    cpdef setFusedOpsVariantParamPackAttribute(size_t varPack, int paramLabel,
                                               size_t ptr)
    cpdef getFusedOpsVariantParamPackAttribute(size_t varPack, int paramLabel,
                                               size_t ptr)
    cpdef createFusedOpsPlan(int ops)
    cpdef destroyFusedOpsPlan(size_t plan)
    cpdef makeFusedOpsPlan(intptr_t handle, size_t plan, size_t constPack)
    cpdef fusedOpsExecute(intptr_t handle, size_t plan, size_t varPack)

