using Adapt
using CUDA.CUSPARSE, SparseArrays, KernelAbstractions, CUDAKernels

"""
    DeviceArray(x) 

Returns a `CuArray` when CUDA is used, otherwise a regular `Array`

"""
DeviceArray(x) = cuda_used[] ? adapt(CuArray,x) : adapt(Array,x)


"""
    DeviceArray(x) 

Returns a CUSPARSE Array when CUDA is used, otherwise a regular sparse Array

"""
DeviceSparseArray(x) = cuda_used[] ? CUDA.CUSPARSE.CuSparseMatrixCSC(x) : sparse(x)

"""
    gpuon()

Manually toggle GPU use on (if available)
"""
function gpuon() # manually toggle GPU use on and off
    cuda_used[] = CUDA.functional()
end

"""
    gpuoff()

Manually toggle GPU use off
"""
function gpuoff()
    cuda_used[] = false
end

"""
    device()

Return currently used device for KernelAbstractions, either `CPU` or `CUDADevice`
"""
device() = cuda_used[] ? CUDADevice : CPU
    