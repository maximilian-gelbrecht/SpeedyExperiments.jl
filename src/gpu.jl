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
    Device()

Return currently used device for KernelAbstractions, either `CPU` or `CUDADevice`
"""
Device() = cuda_used[] ? CUDADevice : CPU
    

"""
    DeviceSetup

Holds information about the device the model is running on and workgroup size
"""
struct DeviceSetup{S,T}
    device::S 
    n::T
end 

DeviceSetup(n::Integer) = DeviceSetup(device(), n)
function DeviceSetup() 
    current_device = Device()
    n = device isa GPU ? 32 : 4  

    DeviceSetup(current_device, n)
end 