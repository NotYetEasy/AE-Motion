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

// Device-friendly min and max functions
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

// Triangle wave calculation for device
__device__ float TriangleWaveDevice(float t) {
    // Shift phase by 0.75 and normalize to [0,1]
    t = fmodf_custom(t + 0.75f, 1.0f);

    // Handle negative values
    if (t < 0)
        t += 1.0f;

    // Transform to triangle wave [-1,1]
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
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        // Calculate center point
        float centerX = (float)(inWidth) / 2.0f;
        float centerY = (float)(inHeight) / 2.0f;

        // Calculate wave value based on time, frequency and phase
        float X;
        float m;

        // Calculate wave value (sine or triangle)
        if (inWaveType == 0) {
            // Sine wave
            X = (inFrequency * 2.0f * inCurrentTime) + (inPhase * 2.0f);
            m = sin(X * 3.14159f);
        }
        else {
            // Triangle wave
            X = ((inFrequency * 2.0f * inCurrentTime) + (inPhase * 2.0f)) / 2.0f + inPhase;
            m = TriangleWaveDevice(X);
        }

        // Calculate angle in radians for direction vector
        float angleRad = inAngle * 3.14159f / 180.0f;
        float dx = cos(angleRad);
        float dy = sin(angleRad);

        // Initialize transformation values
        float offsetX = 0.0f, offsetY = 0.0f;
        float scale = 100.0f;

        // Apply effect based on direction mode
        switch (inDirection) {
        case 0: // Angle mode - position offset only
            offsetX = dx * inMagnitude * m;
            offsetY = dy * inMagnitude * m;
            break;

        case 1: // Depth mode - scale only
            scale = 100.0f - (inMagnitude * m * 0.1f);
            break;

        case 2: { // Orbit mode - position offset and scale
            offsetX = dx * inMagnitude * m;
            offsetY = dy * inMagnitude * m;

            // Calculate second wave with phase shift for scale
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

        // Calculate scale factor
        float scaleFactorX = 100.0f / scale;
        float scaleFactorY = 100.0f / scale;

        // Calculate source coordinates with transformation
        float srcX = (inXY.x - centerX) * scaleFactorX + centerX - offsetX;
        float srcY = (inXY.y - centerY) * scaleFactorY + centerY - offsetY;

        // Initialize variables to track if we're sampling outside the image
        bool outsideBounds = false;

        // Handle tiling based on X and Y Tiles parameters
        if (inXTiles) {
            // X tiling is enabled
            if (inMirror) {
                // Mirror tiling: create ping-pong pattern
                float intPart = floor(fabs(srcX / inWidth));
                float fracPart = fabs(srcX / inWidth) - intPart;
                int isOdd = (int)intPart & 1;
                srcX = isOdd ? (1.0f - fracPart) * inWidth : fracPart * inWidth;
            }
            else {
                // Regular repeat tiling
                srcX = fmodf_custom(fmodf_custom(srcX, inWidth) + inWidth, inWidth);
            }
        }
        else {
            // X tiling is disabled - check if outside bounds
            if (srcX < 0 || srcX >= inWidth) {
                outsideBounds = true;
            }
        }

        // Apply Y tiling
        if (inYTiles) {
            // Y tiling is enabled
            if (inMirror) {
                // Mirror tiling: create ping-pong pattern
                float intPart = floor(fabs(srcY / inHeight));
                float fracPart = fabs(srcY / inHeight) - intPart;
                int isOdd = (int)intPart & 1;
                srcY = isOdd ? (1.0f - fracPart) * inHeight : fracPart * inHeight;
            }
            else {
                // Regular repeat tiling
                srcY = fmodf_custom(fmodf_custom(srcY, inHeight) + inHeight, inHeight);
            }
        }
        else {
            // Y tiling is disabled - check if outside bounds
            if (srcY < 0 || srcY >= inHeight) {
                outsideBounds = true;
            }
        }

        // If we're outside bounds and tiling is disabled, return transparent pixel
        if (outsideBounds) {
            float4 transparent = { 0.0f, 0.0f, 0.0f, 0.0f };
            WriteFloat4(transparent, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        // At this point, we're guaranteed to be within bounds or using tiling
        // Clamp coordinates to valid range to avoid any out-of-bounds access
        srcX = fmaxf_custom(0.0f, fminf_custom(inWidth - 1.001f, srcX));
        srcY = fmaxf_custom(0.0f, fminf_custom(inHeight - 1.001f, srcY));

        // Get integer and fractional parts
        int x0 = (int)srcX;
        int y0 = (int)srcY;
        int x1 = min_int(x0 + 1, inWidth - 1);
        int y1 = min_int(y0 + 1, inHeight - 1);

        float fx = srcX - x0;
        float fy = srcY - y0;

        // Get the four surrounding pixels
        float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
        float4 p01 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
        float4 p10 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
        float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

        // Bilinear interpolation for each channel
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
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    OscillateKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height,
        angle, frequency, magnitude, direction, waveType, phase, currentTime, xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif // __NVCC__
#endif
