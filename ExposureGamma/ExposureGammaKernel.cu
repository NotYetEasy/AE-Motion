#ifndef ExposureGamma
#define ExposureGamma

#include "PrGPU/KernelSupport/KernelCore.h" 
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif
GF_KERNEL_FUNCTION(ExposureGammaKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inExposure))
    ((float)(inGamma))
    ((float)(inOffset)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

        pixel.x += inOffset * pixel.w;
        pixel.y += inOffset * pixel.w;
        pixel.z += inOffset * pixel.w;

        if (inGamma != 0.0f) {
            pixel.x = pow(fmax(pixel.x, 0.0f), 1.0f / inGamma);
            pixel.y = pow(fmax(pixel.y, 0.0f), 1.0f / inGamma);
            pixel.z = pow(fmax(pixel.z, 0.0f), 1.0f / inGamma);
        }

        float exposureFactor = pow(2.0f, inExposure);
        pixel.x *= exposureFactor;
        pixel.y *= exposureFactor;
        pixel.z *= exposureFactor;

        pixel.x = fmax(fmin(pixel.x, 1.0f), 0.0f);
        pixel.y = fmax(fmin(pixel.y, 1.0f), 0.0f);
        pixel.z = fmax(fmin(pixel.z, 1.0f), 0.0f);
        pixel.w = fmax(fmin(pixel.w, 1.0f), 0.0f);

        WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
void ExposureGamma_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float inExposure,
    float inGamma,
    float inOffset)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    ExposureGammaKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, inExposure, inGamma, inOffset);

    cudaDeviceSynchronize();
}
#endif  
#endif
