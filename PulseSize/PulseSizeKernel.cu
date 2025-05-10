#ifndef PULSESIZE
#define PULSESIZE

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

__device__ float fmodf_custom(float x, float y) {
    return x - y * floor(x / y);
}

__device__ float TriangleWaveDevice(float t) {
    t = fmodf_custom(t + 0.75f, 1.0f);

    if (t < 0)
        t += 1.0f;

    return (fabs(t - 0.5f) - 0.25f) * 4.0f;
}

GF_KERNEL_FUNCTION(PulseSizeKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inFrequency))
    ((float)(inShrink))
    ((float)(inGrow))
    ((int)(inWaveType))
    ((float)(inPhase))
    ((float)(inCurrentTime))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float centerX = (float)(inWidth) / 2.0f;
        float centerY = (float)(inHeight) / 2.0f;

        float X;
        float m;

        if (inWaveType == 0) {
            X = (inFrequency * inCurrentTime) + inPhase;
            m = sin(X * 3.14159f);
        }
        else {
            X = ((inFrequency * inCurrentTime) + inPhase) / 2.0f + inPhase;
            m = TriangleWaveDevice(X);
        }

        float range = inGrow - inShrink;
        float ds = (range * ((m + 1.0f) / 2.0f)) + inShrink;

        float scaleFactor = 1.0f / ds;

        float srcX = (inXY.x - centerX) * scaleFactor + centerX;
        float srcY = (inXY.y - centerY) * scaleFactor + centerY;

        bool outsideBounds = false;

        if (inXTiles) {
            if (inMirror) {
                float intPart = floor(fabs(srcX / inWidth));
                float fracPart = fabs(srcX / inWidth) - intPart;
                int isOdd = (int)intPart & 1;
                srcX = isOdd ? (1.0f - fracPart) * inWidth : fracPart * inWidth;
            }
            else {
                srcX = fmodf_custom(fmodf_custom(srcX, inWidth) + inWidth, inWidth);
            }
        }
        else {
            if (srcX < 0 || srcX >= inWidth) {
                outsideBounds = true;
            }
        }

        if (inYTiles) {
            if (inMirror) {
                float intPart = floor(fabs(srcY / inHeight));
                float fracPart = fabs(srcY / inHeight) - intPart;
                int isOdd = (int)intPart & 1;
                srcY = isOdd ? (1.0f - fracPart) * inHeight : fracPart * inHeight;
            }
            else {
                srcY = fmodf_custom(fmodf_custom(srcY, inHeight) + inHeight, inHeight);
            }
        }
        else {
            if (srcY < 0 || srcY >= inHeight) {
                outsideBounds = true;
            }
        }

        if (outsideBounds) {
            float4 transparent = { 0.0f, 0.0f, 0.0f, 0.0f };
            WriteFloat4(transparent, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        srcX = fmaxf_custom(0.0f, fminf_custom(inWidth - 1.001f, srcX));
        srcY = fmaxf_custom(0.0f, fminf_custom(inHeight - 1.001f, srcY));

        int x0 = (int)srcX;
        int y0 = (int)srcY;
        int x1 = min_int(x0 + 1, inWidth - 1);
        int y1 = min_int(y0 + 1, inHeight - 1);

        float fx = srcX - x0;
        float fy = srcY - y0;

        float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
        float4 p01 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
        float4 p10 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
        float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

        float4 pixel;
        float oneMinusFx = 1.0f - fx;
        float oneMinusFy = 1.0f - fy;

        pixel.x = oneMinusFx * oneMinusFy * p00.x +
            fx * oneMinusFy * p01.x +
            oneMinusFx * fy * p10.x +
            fx * fy * p11.x;

        pixel.y = oneMinusFx * oneMinusFy * p00.y +
            fx * oneMinusFy * p01.y +
            oneMinusFx * fy * p10.y +
            fx * fy * p11.y;

        pixel.z = oneMinusFx * oneMinusFy * p00.z +
            fx * oneMinusFy * p01.z +
            oneMinusFx * fy * p10.z +
            fx * fy * p11.z;

        pixel.w = oneMinusFx * oneMinusFy * p00.w +
            fx * oneMinusFy * p01.w +
            oneMinusFx * fy * p10.w +
            fx * fy * p11.w;

        WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
void PulseSize_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float frequency,
    float shrink,
    float grow,
    int waveType,
    float phase,
    float currentTime,
    int xTiles,
    int yTiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    PulseSizeKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height,
        frequency, shrink, grow, waveType, phase, currentTime, xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif  
#endif  
