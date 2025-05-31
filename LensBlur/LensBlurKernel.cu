#ifndef LensBlur
#define LensBlur

#include "PrGPU/KernelSupport/KernelCore.h"   
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#define __device__
#define fabs abs
#define fmodf fmod
#define fmaxf max
#define fminf min
#define sqrt sqrt
#endif

__device__ int min_int(int a, int b) {
    return a < b ? a : b;
}

__device__ int max_int(int a, int b) {
    return a > b ? a : b;
}

__device__ float fminf_custom(float a, float b) {
    return a < b ? a : b;
}

__device__ float fmaxf_custom(float a, float b) {
    return a > b ? a : b;
}

__device__ float smoothstepf(float edge0, float edge1, float x) {
    float t = fmaxf_custom(0.0f, fminf_custom(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

__device__ float mixf(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}


GF_KERNEL_FUNCTION(LensBlurKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inCenterX))
    ((float)(inCenterY))
    ((float)(inStrength))
    ((float)(inRadius)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float x = (float)inXY.x / (float)inWidth;
        float y = (float)inXY.y / (float)inHeight;

        float centerX = inCenterX;
        float centerY = inCenterY;

        float vx = x - centerX;
        float vy = y - centerY;

        float dist = sqrt(vx * vx + vy * vy);

        float blurStrength = 0.0f;
        if (dist <= inRadius) {
            blurStrength = 0.0f;
        }
        else if (dist >= 1.0f) {
            blurStrength = 1.0f;
        }
        else {
            float t = (dist - inRadius) / (1.0f - inRadius);
            blurStrength = t * t * (3.0f - 2.0f * t);
        }

        if (blurStrength < 0.01f) {
            float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
            WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        if (dist > 0.0001f) {
            vx /= dist;
            vy /= dist;
        }

        float texelSizeX = 1.0f / (float)inWidth;
        float texelSizeY = 1.0f / (float)inHeight;

        float speed = inStrength / 2.0f;
        speed /= texelSizeX;      
        speed *= blurStrength;

        int nSamples = (int)fmaxf_custom(1.01f, fminf_custom(speed, 100.01f));

        if (nSamples <= 1) {
            float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
            WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        vx *= texelSizeX * speed;
        vy *= texelSizeY * speed;

        float4 accum;
        accum.x = 0.0f;
        accum.y = 0.0f;
        accum.z = 0.0f;
        accum.w = 0.0f;

        for (int i = 0; i < nSamples; i++) {
            float t = (float)i / (float)(nSamples - 1) - 0.5f;

            float sampleX = (float)inXY.x - vx * inWidth * t;
            float sampleY = (float)inXY.y - vy * inHeight * t;

            int x0 = (int)floor(sampleX);
            int y0 = (int)floor(sampleY);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            x0 = max_int(0, min_int(inWidth - 1, x0));
            y0 = max_int(0, min_int(inHeight - 1, y0));
            x1 = max_int(0, min_int(inWidth - 1, x1));
            y1 = max_int(0, min_int(inHeight - 1, y1));

            float fx = sampleX - x0;
            float fy = sampleY - y0;

            float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
            float4 p10 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
            float4 p01 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
            float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

            float oneMinusFx = 1.0f - fx;
            float oneMinusFy = 1.0f - fy;

            float w00 = oneMinusFx * oneMinusFy;
            float w10 = fx * oneMinusFy;
            float w01 = oneMinusFx * fy;
            float w11 = fx * fy;

            float4 sample;
            sample.x = p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11;
            sample.y = p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11;
            sample.z = p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11;
            sample.w = p00.w * w00 + p10.w * w10 + p01.w * w01 + p11.w * w11;

            accum.x += sample.x;
            accum.y += sample.y;
            accum.z += sample.z;
            accum.w += sample.w;
        }

        float invSamples = 1.0f / (float)nSamples;
        accum.x *= invSamples;
        accum.y *= invSamples;
        accum.z *= invSamples;
        accum.w *= invSamples;

        WriteFloat4(accum, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}

#endif

#if __NVCC__
void LensBlur_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float centerX,
    float centerY,
    float strength,
    float radius)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    LensBlurKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, centerX, centerY, strength, radius);

    cudaDeviceSynchronize();
}
#endif  
#endif