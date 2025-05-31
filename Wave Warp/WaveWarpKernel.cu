#ifndef WaveWarp
#define WaveWarp

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

__device__ float mixf(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

__device__ float smoothstepf(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

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

GF_KERNEL_FUNCTION(WaveWarpKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inPhase))
    ((float)(inDirection))
    ((float)(inSpacing))
    ((float)(inMagnitude))
    ((float)(inWarpAngle))
    ((float)(inDamping))
    ((float)(inDampingSpace))
    ((float)(inDampingOrigin))
    ((int)(inScreenSpace))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float uvX = (float)inXY.x / (float)inWidth;
        float uvY = (float)inXY.y / (float)inHeight;

        float direction_rad = inDirection * 0.0174533f;
        float warpangle_rad = inWarpAngle * 0.0174533f;

        float a1 = direction_rad;
        float a2 = direction_rad - warpangle_rad;

        float st_x = uvX;
        float st_y = 1.0f - uvY;

        float raw_v_x = cos(a1);
        float raw_v_y = sin(a1);

        float raw_p = (st_x * raw_v_x) + (st_y * raw_v_y);

        float space_damp = 1.0f;
        if (inDampingSpace < 0.0f) {
            space_damp = 1.0f - (fminf_custom(fabs(raw_p - inDampingOrigin), 1.0f) * (0.0f - inDampingSpace));
        }
        else if (inDampingSpace > 0.0f) {
            space_damp = 1.0f - ((1.0f - fminf_custom(fabs(raw_p - inDampingOrigin), 1.0f)) * inDampingSpace);
        }

        float space = inSpacing * space_damp;

        float v_x = cos(a1) * space;
        float v_y = sin(a1) * space;

        float p = (st_x * v_x) + (st_y * v_y);

        float ddist = fabs(p / space);

        float damp = 1.0f;
        if (inDamping < 0.0f) {
            damp = 1.0f - (fminf_custom(fabs(ddist - inDampingOrigin), 1.0f) * (0.0f - inDamping));
        }
        else if (inDamping > 0.0f) {
            damp = 1.0f - ((1.0f - fminf_custom(fabs(ddist - inDampingOrigin), 1.0f)) * inDamping);
        }

        float offs_x = cos(a2) * (inMagnitude * damp) / 100.0f;
        float offs_y = sin(a2) * (inMagnitude * damp) / 100.0f;

        float wave = sin(p + inPhase * 6.28318f);
        offs_x *= wave;
        offs_y *= wave;

        float finalSrcX = uvX + offs_x;
        float finalSrcY = uvY - offs_y;      

        bool outsideBounds = false;

        if (inXTiles) {
            if (inMirror) {
                float fracPart = fmodf(fabs(finalSrcX), 1.0f);
                int isOdd = (int)floor(fabs(finalSrcX)) & 1;
                finalSrcX = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                finalSrcX = finalSrcX - floor(finalSrcX);
                if (finalSrcX < 0.0f) finalSrcX += 1.0f;
            }
        }
        else {
            if (finalSrcX < 0.0f || finalSrcX >= 1.0f) {
                outsideBounds = true;
            }
        }

        if (inYTiles) {
            if (inMirror) {
                float fracPart = fmodf(fabs(finalSrcY), 1.0f);
                int isOdd = (int)floor(fabs(finalSrcY)) & 1;
                finalSrcY = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                finalSrcY = finalSrcY - floor(finalSrcY);
                if (finalSrcY < 0.0f) finalSrcY += 1.0f;
            }
        }
        else {
            if (finalSrcY < 0.0f || finalSrcY >= 1.0f) {
                outsideBounds = true;
            }
        }

        float4 pixel;

        if (outsideBounds) {
            pixel.x = 0.0f;
            pixel.y = 0.0f;
            pixel.z = 0.0f;
            pixel.w = 0.0f;
        }
        else {
            finalSrcX = fmaxf_custom(0.0f, fminf_custom(0.9999f, finalSrcX));
            finalSrcY = fmaxf_custom(0.0f, fminf_custom(0.9999f, finalSrcY));

            float x = finalSrcX * inWidth;
            float y = finalSrcY * inHeight;

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
void WaveWarp_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float phase,
    float direction,
    float spacing,
    float magnitude,
    float warpAngle,
    float damping,
    float dampingSpace,
    float dampingOrigin,
    int screenSpace,
    int xTiles,
    int yTiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    WaveWarpKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height,
        phase, direction, spacing, magnitude, warpAngle, damping,
        dampingSpace, dampingOrigin, screenSpace, xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif  
#endif
