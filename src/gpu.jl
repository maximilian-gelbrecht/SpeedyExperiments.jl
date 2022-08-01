using Adapt
using CUDA.CUSPARSE, SparseArrays, KernelAbstractions, CUDAKernels


abstract type AbstractDevice end 

struct CPUDevice <: AbstractDevice end 
struct GPUDevice <: AbstractDevice end 

"""
    Device_KA()

Return currently used device for KernelAbstractions, either `CPU` or `CUDADevice`
"""
Device_KA() = cuda_used[] ? CUDADevice : CPU

"""
    Device()

Return currently used device for internal purposes, either `CPUDevice` or `GPUDevice`
"""
Device() = cuda_used[] ? GPUDevice() : CPUDevice()
    
    

"""
    DeviceSetup

Holds information about the device the model is running on and workgroup size. 
"""
struct DeviceSetup{S<:AbstractDevice,T,U}
    device::S # for internal purposes
    device_KA::T # for KernelAbstractions
    n::U
end 

DeviceSetup(n::Integer) = DeviceSetup(Device(), Device_KA(), n)
function DeviceSetup() 
    current_device = Device_KA()
    n = device isa GPU ? 32 : 4  

    DeviceSetup(Device(), current_device, n)
end 


"""
    DeviceArray(x) 

Returns a `CuArray` when CUDA is used, otherwise a regular `Array`

"""
DeviceArray(x) = cuda_used[] ? adapt(CuArray,x) : adapt(Array,x)

"""
    DeviceArray(device::AbstractDevice, x) 

Returns a `CuArray` when `device<:GPUDevice` is used, otherwise a regular `Array`. Uses `adapt`, thus also can return SubArrays etc.

"""
DeviceArray(::GPUDevice, x) = adapt(CuArray, x)
DeviceArray(::CPUDevice, x) = adapt(Array, x)
DeviceArray(dev::DeviceSetup, x) = DeviceArray(dev.device, x)

"""
    DeviceArrayNotAdapt(device::AbstractDevice, x) 

Returns a `CuArray` when `device<:GPUDevice` is used, otherwise a regular `Array`. Doesn't uses `adapt`, therefore always returns CuArray/Array

"""
DeviceArrayNotAdapt(::GPUDevice, x) = CuArray(x)
DeviceArrayNotAdapt(::CPUDevice, x) = Array(x)
DeviceArrayNotAdapt(dev::DeviceSetup, x) = DeviceArrayNotAdapt(dev.device, x)



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
    launch_kernel!(device_setup::DeviceSetup, kernel!, ndrange, kernel_args...)

Launches the `kernel!` on the `device_setup` with `ndrange` computations over the kernel and arguments `kernel_args`. Returns an event.
"""
function launch_kernel!(device_setup::DeviceSetup, kernel!, ndrange, kernel_args...)
    device = device_setup.device_KA()
    n = device_setup.n 

    k! = kernel!(device, n)
    event = k!(kernel_args...; ndrange=ndrange)

    return event 
end