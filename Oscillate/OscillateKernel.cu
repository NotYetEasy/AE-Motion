#ifndef OSCILLATE
#define OSCILLATE

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

GF_KERNEL_FUNCTION(OscillateKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inAngle))
    ((float)(inFrequency))
    ((float)(inMagnitude))
    ((int)(inDirection))
    ((int)(inWaveType))
    ((float)(inPhase))
    ((float)(inCurrentTime))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror))
    ((float)(inDownsampleX))
    ((float)(inDownsampleY))
    ((int)(inCompatibilityEnabled))     
    ((float)(inCompatAngle))            
    ((float)(inCompatFrequency))        
    ((float)(inCompatMagnitude))        
    ((int)(inCompatWaveType))            
    ((int)(inNormalEnabled)),            
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        if ((!inNormalEnabled && !inCompatibilityEnabled) || (inNormalEnabled && inCompatibilityEnabled)) {
            float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
            WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        float centerX = (float)(inWidth) / 2.0f;
        float centerY = (float)(inHeight) / 2.0f;

        float offsetX = 0.0f, offsetY = 0.0f;
        float scale = 100.0f;

        if (inCompatibilityEnabled) {
            float angleRad = inCompatAngle * 3.14159f / 180.0f;

            float dx = sin(angleRad);
            float dy = cos(angleRad);

            float duration = 1.0f;
            float t = inCurrentTime;
            float m;

            if (inCompatWaveType == 0) {
                m = sin(t * duration * inCompatFrequency * 3.14159f);
            }
            else {
                float wavePhase = t * duration * inCompatFrequency / 2.0f;
                m = TriangleWaveDevice(wavePhase);
            }

            offsetX = dx * inCompatMagnitude * m;
            offsetY = dy * inCompatMagnitude * m;
        }
        else if (inNormalEnabled) {
            float X;
            float m;

            if (inWaveType == 0) {
                X = (inFrequency * 2.0f * inCurrentTime) + (inPhase * 2.0f);
                m = sin(X * 3.14159f);
            }
            else {
                X = ((inFrequency * 2.0f * inCurrentTime) + (inPhase * 2.0f)) / 2.0f + inPhase;
                m = TriangleWaveDevice(X);
            }

            float angleRad = inAngle * 3.14159f / 180.0f;
            float dx = cos(angleRad);
            float dy = sin(angleRad);

            switch (inDirection) {
            case 0:       
                offsetX = dx * (inMagnitude * inDownsampleX) * m;
                offsetY = dy * (inMagnitude * inDownsampleY) * m;
                break;

            case 1:      
                scale = 100.0f - (inMagnitude * m * 0.1f);
                break;

            case 2: {        
                offsetX = dx * (inMagnitude * inDownsampleX) * m;
                offsetY = dy * (inMagnitude * inDownsampleY) * m;

                float phaseShift = inWaveType == 0 ? 0.25f : 0.125f;
                float X2;

                if (inWaveType == 0) {
                    X2 = (inFrequency * 2.0f * inCurrentTime) + ((inPhase + phaseShift) * 2.0f);
                    m = sin(X2 * 3.14159f);
                }
                else {
                    X2 = ((inFrequency * 2.0f * inCurrentTime) + ((inPhase + phaseShift) * 2.0f)) / 2.0f + (inPhase + phaseShift);
                    m = TriangleWaveDevice(X2);
                }
                scale = 100.0f - (inMagnitude * m * 0.1f);
                break;
            }
            }
        }

        float scaleFactorX = 100.0f / scale;
        float scaleFactorY = 100.0f / scale;

        float srcX = (inXY.x - centerX) * scaleFactorX + centerX - offsetX;
        float srcY = (inXY.y - centerY) * scaleFactorY + centerY - offsetY;

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
void Oscillate_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float angle,
    float frequency,
    float magnitude,
    int direction,
    int waveType,
    float phase,
    float currentTime,
    int xTiles,
    int yTiles,
    int mirror,
    float downsample_x,
    float downsample_y,
    int compatibilityEnabled,     
    float compatAngle,            
    float compatFrequency,        
    float compatMagnitude,        
    int compatWaveType,            
    int normalEnabled)             
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    OscillateKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height,
        angle, frequency, magnitude, direction, waveType, phase, currentTime, xTiles, yTiles, mirror,
        downsample_x, downsample_y, compatibilityEnabled, compatAngle, compatFrequency, compatMagnitude, compatWaveType, normalEnabled);

    cudaDeviceSynchronize();
}
#endif  
#endif