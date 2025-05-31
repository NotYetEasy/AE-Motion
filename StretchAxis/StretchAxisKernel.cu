#ifndef StretchAxis
#define StretchAxis

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

GF_KERNEL_FUNCTION(StretchAxisKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inScale))
    ((float)(inAngle))
    ((int)(inContentOnly))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float centerX = inWidth * 0.5f;
        float centerY = inHeight * 0.5f;

        float x = inXY.x - centerX;
        float y = inXY.y - centerY;

        float rad = -inAngle * (3.14159265358979323846f / 180.0f);

        float cos_rad = cos(rad);
        float sin_rad = sin(rad);
        float cos_neg_rad = cos(-rad);
        float sin_neg_rad = sin(-rad);

        float x_rot = x * cos_rad - y * sin_rad;
        float y_rot = x * sin_rad + y * cos_rad;

        if (inScale != 0.0f) {
            x_rot /= inScale;
        }

        float x_final = x_rot * cos_neg_rad - y_rot * sin_neg_rad;
        float y_final = x_rot * sin_neg_rad + y_rot * cos_neg_rad;

        float sampleX = x_final + centerX;
        float sampleY = y_final + centerY;

        float uvX = sampleX / inWidth;
        float uvY = sampleY / inHeight;

        float4 currentPixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

        bool outsideBounds = false;

        if (inXTiles) {
            if (inMirror) {
                float fracPart = fmodf(fabs(uvX), 1.0f);
                int isOdd = (int)floor(fabs(uvX)) & 1;
                uvX = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                uvX = uvX - floor(uvX);
                if (uvX < 0.0f) uvX += 1.0f;
            }
        }
        else {
            if (uvX < 0.0f || uvX >= 1.0f) {
                outsideBounds = true;
            }
        }

        if (inYTiles) {
            if (inMirror) {
                float fracPart = fmodf(fabs(uvY), 1.0f);
                int isOdd = (int)floor(fabs(uvY)) & 1;
                uvY = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                uvY = uvY - floor(uvY);
                if (uvY < 0.0f) uvY += 1.0f;
            }
        }
        else {
            if (uvY < 0.0f || uvY >= 1.0f) {
                outsideBounds = true;
            }
        }

        float4 result;

        if (outsideBounds) {
            result.x = 0.0f;
            result.y = 0.0f;
            result.z = 0.0f;
            result.w = 0.0f;

            if (inContentOnly) {
                result.w = currentPixel.w;
            }
        }
        else {
            uvX = fmaxf_custom(0.0f, fminf_custom(0.9999f, uvX));
            uvY = fmaxf_custom(0.0f, fminf_custom(0.9999f, uvY));

            float x = uvX * inWidth;
            float y = uvY * inHeight;

            int x0 = (int)floor(x);
            int y0 = (int)floor(y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            x0 = max_int(0, min_int(inWidth - 1, x0));
            y0 = max_int(0, min_int(inHeight - 1, y0));
            x1 = max_int(0, min_int(inWidth - 1, x1));
            y1 = max_int(0, min_int(inHeight - 1, y1));

            float fx = x - x0;
            float fy = y - y0;

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

            result.x = p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11;
            result.y = p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11;
            result.z = p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11;
            result.w = p00.w * w00 + p10.w * w10 + p01.w * w01 + p11.w * w11;

            if (inContentOnly) {
                result.w = currentPixel.w;
            }
        }

        WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
void StretchAxis_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float scale,
    float angle,
    int contentOnly,
    int xTiles,
    int yTiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    StretchAxisKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, scale, angle, contentOnly, xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif  
#endif
