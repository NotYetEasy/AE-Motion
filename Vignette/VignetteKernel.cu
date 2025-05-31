#ifndef Vignette
#define Vignette

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif
GF_KERNEL_FUNCTION(VignetteKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inScale))
    ((float)(inRoundness))
    ((float)(inFeather))
    ((float)(inStrength))
    ((float)(inTint))
    ((float)(inColorR))
    ((float)(inColorG))
    ((float)(inColorB))
    ((int)(inPunchout)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

        float width = (float)inWidth;
        float height = (float)inHeight;

        float normX = (float)inXY.x / width;
        float normY = (float)inXY.y / height;

        float stX = (normX - 0.5f) * 2.0f;
        float stY = (normY - 0.5f) * 2.0f;

        float scale = inScale + (inFeather / 4.0f);
        stX /= scale;
        stY /= scale;

        float n = inRoundness + 1.0f;
        float d = pow(abs(stX), n) + pow(abs(stY), n);

        float p = 0.0f;
        float edge0 = 1.0f - inFeather;
        float edge1 = 1.0f;
        float t = fmax(fmin((d - edge0) / (edge1 - edge0), 1.0f), 0.0f);
        p = t * t * (3.0f - 2.0f * t);

        float4 darkColor = pixel;

        float r = pixel.x * pixel.x;
        float g = pixel.y * pixel.y;
        float b = pixel.z * pixel.z;

        r = r * (1.0f - inTint) + inColorR * inTint;
        g = g * (1.0f - inTint) + inColorG * inTint;
        b = b * (1.0f - inTint) + inColorB * inTint;

        darkColor.x = r;
        darkColor.y = g;
        darkColor.z = b;

        float strength = inStrength;

        if (inPunchout)
        {
            pixel.x = pixel.x * (1.0f - p * strength);
            pixel.y = pixel.y * (1.0f - p * strength);
            pixel.z = pixel.z * (1.0f - p * strength);
            pixel.w = pixel.w * (1.0f - p * strength);
        }
        else
        {
            pixel.x = pixel.x * (1.0f - p * strength) + darkColor.x * (p * strength);
            pixel.y = pixel.y * (1.0f - p * strength) + darkColor.y * (p * strength);
            pixel.z = pixel.z * (1.0f - p * strength) + darkColor.z * (p * strength);
        }

        WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}

#endif
#if __NVCC__
void Vignette_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float scale,
    float roundness,
    float feather,
    float strength,
    float tint,
    float colorR,
    float colorG,
    float colorB,
    int punchout)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    VignetteKernel << < gridDim, blockDim, 0 >> > (
        (float4 const*)src,
        (float4*)dst,
        srcPitch,
        dstPitch,
        is16f,
        width,
        height,
        scale,
        roundness,
        feather,
        strength,
        tint,
        colorR,
        colorG,
        colorB,
        punchout);

    cudaDeviceSynchronize();
}
#endif  
#endif