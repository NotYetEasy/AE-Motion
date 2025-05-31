#ifndef Swing
#define Swing

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

__device__ unsigned int min_uint(unsigned int a, unsigned int b) {
    return a < b ? a : b;
}

__device__ float fminf_custom(float a, float b) {
    return a < b ? a : b;
}

__device__ float fmaxf_custom(float a, float b) {
    return a > b ? a : b;
}

__device__ float triangleWave(float t) {
    t = fmodf(t + 0.75f, 1.0f);

    if (t < 0)
        t += 1.0f;

    return (fabs(t - 0.5f) - 0.25f) * 4.0f;
}

GF_KERNEL_FUNCTION(SwingKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inFrequency))
    ((float)(inAngle1))
    ((float)(inAngle2))
    ((float)(inPhase))
    ((float)(inTime))
    ((int)(inWaveType))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror))
    ((float)(inAccumulatedPhase))
    ((int)(inHasFrequencyKeyframes))
    ((int)(inNormalEnabled))
    ((int)(inCompatibilityEnabled))
    ((float)(inCompatFrequency))
    ((float)(inCompatAngle1))
    ((float)(inCompatAngle2))
    ((float)(inCompatPhase))
    ((int)(inCompatWaveType)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        if ((inNormalEnabled && inCompatibilityEnabled) || (!inNormalEnabled && !inCompatibilityEnabled)) {
            float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
            WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        float effectivePhase;
        float m;
        float angleRad;

        if (inCompatibilityEnabled) {
            if (inCompatWaveType == 0) {  
                m = sin(((inTime * inCompatFrequency) + inCompatPhase) * 3.14159265f);
            }
            else {  
                m = triangleWave(((inTime * inCompatFrequency) + inCompatPhase) / 2.0f);
            }

            float finalAngle = ((inCompatAngle2 - inCompatAngle1) * ((m + 1.0f) / 2.0f)) + inCompatAngle1;
            angleRad = -finalAngle * 0.01745329f;    
        }
        else {
            if (inHasFrequencyKeyframes && inAccumulatedPhase > 0.0f) {
                effectivePhase = inPhase + inAccumulatedPhase;

                if (inWaveType == 0) {
                    m = sin(effectivePhase * 3.14159265f);
                }
                else {
                    m = triangleWave(effectivePhase / 2.0f);
                }
            }
            else {
                effectivePhase = inPhase + (inTime * inFrequency);

                if (inWaveType == 0) {
                    m = sin(effectivePhase * 3.14159265f);
                }
                else {
                    m = triangleWave(effectivePhase / 2.0f);
                }
            }

            float t = (m + 1.0f) / 2.0f;

            float finalAngle = -(inAngle1 + t * (inAngle2 - inAngle1));
            angleRad = finalAngle * 0.01745329f;    
        }

        float centerX = inWidth / 2.0f;
        float centerY = inHeight / 2.0f;

        float dx = inXY.x - centerX;
        float dy = inXY.y - centerY;

        float cos_rot = cos(angleRad);
        float sin_rot = sin(angleRad);
        float rotated_x = (dx * cos_rot - dy * sin_rot) + centerX;
        float rotated_y = (dx * sin_rot + dy * cos_rot) + centerY;

        float u = rotated_x / (float)inWidth;
        float v = rotated_y / (float)inHeight;

        bool outsideBounds = false;

        if (inXTiles) {
            if (inMirror) {
                float fracPart = fmodf(fabs(u), 1.0f);
                int isOdd = (int)floor(fabs(u)) & 1;
                u = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                u = u - floor(u);
            }
        }
        else if (u < 0.0f || u >= 1.0f) {
            outsideBounds = true;
        }

        if (inYTiles) {
            if (inMirror) {
                float fracPart = fmodf(fabs(v), 1.0f);
                int isOdd = (int)floor(fabs(v)) & 1;
                v = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                v = v - floor(v);
            }
        }
        else if (v < 0.0f || v >= 1.0f) {
            outsideBounds = true;
        }

        float4 pixel;

        if (outsideBounds) {
            pixel.x = 0.0f;
            pixel.y = 0.0f;
            pixel.z = 0.0f;
            pixel.w = 0.0f;
        }
        else {
            u = fmaxf_custom(0.0f, fminf_custom(0.9999f, u));
            v = fmaxf_custom(0.0f, fminf_custom(0.9999f, v));

            float x = u * inWidth;
            float y = v * inHeight;

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

            pixel.x = p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11;
            pixel.y = p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11;
            pixel.z = p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11;
            pixel.w = p00.w * w00 + p10.w * w10 + p01.w * w01 + p11.w * w11;
        }

        WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
void Swing_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float frequency,
    float angle1,
    float angle2,
    float phase,
    float time,
    int waveType,
    int xTiles,
    int yTiles,
    int mirror,
    float accumulatedPhase,
    int hasFrequencyKeyframes,
    int normalEnabled,
    int compatibilityEnabled,
    float compatFrequency,
    float compatAngle1,
    float compatAngle2,
    float compatPhase,
    int compatWaveType)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    SwingKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height,
        frequency, angle1, angle2, phase, time, waveType, xTiles, yTiles, mirror,
        accumulatedPhase, hasFrequencyKeyframes, normalEnabled, compatibilityEnabled,
        compatFrequency, compatAngle1, compatAngle2, compatPhase, compatWaveType);

    cudaDeviceSynchronize();
}
#endif
#endif 
