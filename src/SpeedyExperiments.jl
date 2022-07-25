module SpeedyExperiments

using CUDA 

global const cuda_used = Ref(false)

function __init__() # automatically called at runtime to set cuda_used
    cuda_used[] = CUDA.functional()
end

include("gpu.jl")
export DeviceArray, DeviceSparseArray

end
