#ifndef ZoomBlur
#define ZoomBlur

#include "PrGPU/KernelSupport/KernelCore.h" 
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif

GF_KERNEL_FUNCTION(ZoomBlurKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inStrength))
    ((float)(inCenterX))
    ((float)(inCenterY))
    ((int)(inAdaptive)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float4 srcPixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

        if (inStrength <= 0.001f) {
            WriteFloat4(srcPixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        float4 outColor = srcPixel;

        float uvX = (float)inXY.x / inWidth;
        float uvY = (float)inXY.y / inHeight;

        float centerX = inCenterX;
        float centerY = inCenterY;

        float vX = uvX - centerX;
        float vY = uvY - centerY;

        float dist = sqrt(vX * vX + vY * vY);

        float texelSizeX = 1.0f / inWidth;
        float texelSizeY = 1.0f / inHeight;
        float texelSize = fmin(texelSizeX, texelSizeY);

        float speed = inStrength / 2.0f / texelSize;

        if (inAdaptive) {
            speed *= dist;
        }
        else {
            float smoothEdge = texelSize * 5.0f;
            float t = fmax(0.0f, fmin(1.0f, dist / smoothEdge));
            speed *= t * t * (3.0f - 2.0f * t);   
        }

        int numSamples = (int)fmax(1.01f, fmin(100.01f, speed));
        if (numSamples <= 1) {
            WriteFloat4(srcPixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        float normX = 0.0f;
        float normY = 0.0f;

        if (dist > 0.0f) {
            normX = vX / dist;
            normY = vY / dist;
        }

        float aspectRatioX = 1.0f;
        float aspectRatioY = 1.0f;

        if (inHeight > inWidth) {
            aspectRatioX *= (float)inWidth / inHeight;
        }
        else {
            aspectRatioY *= (float)inHeight / inWidth;
        }

        float sumR = srcPixel.x;
        float sumG = srcPixel.y;
        float sumB = srcPixel.z;
        float sumA = srcPixel.w;
        int validSamples = 1;        

        for (int i = 1; i < numSamples; i++) {
            float sampleOffset = ((float)i / (float)(numSamples - 1) - 0.5f);

            float offsetX = normX * sampleOffset * speed * texelSize * aspectRatioX;
            float offsetY = normY * sampleOffset * speed * texelSize * aspectRatioY;

            float sampleUvX = uvX - offsetX;
            float sampleUvY = uvY - offsetY;

            int sampleX = (int)(sampleUvX * inWidth);
            int sampleY = (int)(sampleUvY * inHeight);

            if (sampleX >= 0 && sampleX < inWidth && sampleY >= 0 && sampleY < inHeight) {
                float4 samplePixel = ReadFloat4(inSrc, sampleY * inSrcPitch + sampleX, !!in16f);

                sumR += samplePixel.x;
                sumG += samplePixel.y;
                sumB += samplePixel.z;
                sumA += samplePixel.w;
                validSamples++;
            }
        }

        float invValidSamples = 1.0f / (float)validSamples;
        outColor.x = sumR * invValidSamples;
        outColor.y = sumG * invValidSamples;
        outColor.z = sumB * invValidSamples;
        outColor.w = sumA * invValidSamples;

        WriteFloat4(outColor, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}

#endif

#if __NVCC__
void ZoomBlur_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float strength,
    float centerX,
    float centerY,
    int adaptive)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    ZoomBlurKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst,
        srcPitch, dstPitch,
        is16f, width, height,
        strength, centerX, centerY, adaptive);

    cudaDeviceSynchronize();
}
#endif  
#endif
