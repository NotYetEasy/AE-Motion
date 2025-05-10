#ifndef SDK_INVERT_PROC_AMP
#define SDK_INVERT_PROC_AMP

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
#define exp exp2
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



GF_KERNEL_FUNCTION(InnerPinchBulgeKernel,
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
    ((float)(inRadius))
    ((float)(inFeather))
    ((int)(inUseGaussian))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float uvX = (float)inXY.x / (float)inWidth;
        float uvY = (float)inXY.y / (float)inHeight;

        float centerX = (float)inWidth / 2.0f + inCenterX / 65536.0f;
        float centerY = (float)inHeight / 2.0f - inCenterY / 65536.0f;

        float centerNormX = centerX / (float)inWidth;
        float centerNormY = centerY / (float)inHeight;

        float offsetX = uvX - centerNormX;
        float offsetY = uvY - centerNormY;

        float aspectRatio = (float)inWidth / (float)inHeight;
        offsetX *= aspectRatio;

        float dist = sqrt(offsetX * offsetX + offsetY * offsetY);

        float prevOffsetX = offsetX;
        float prevOffsetY = offsetY;

        if (dist < inRadius) {
            float p = dist / inRadius;

            if (inStrength > 0.0f) {
                if (inUseGaussian) {
                    float factor = 1.0f;
                    if (p > 0.0f) {
                        float gp = fminf_custom(fmaxf_custom((p - 0.0f) / (inRadius / dist - 0.0f), 0.0f), 1.0f);
                        gp -= 1.0f;
                        float gsmooth = exp(-((gp * gp) / 0.125f));
                        factor = mixf(1.0f, gsmooth, inStrength * 0.75f);
                    }
                    offsetX *= factor;
                    offsetY *= factor;
                }
                else {
                    float t = fmaxf_custom(0.0f, fminf_custom((p - 0.0f) / (inRadius / dist - 0.0f), 1.0f));
                    float smoothstepVal = t * t * (3.0f - 2.0f * t);
                    float factor = mixf(1.0f, smoothstepVal, inStrength * 0.75f);
                    offsetX *= factor;
                    offsetY *= factor;
                }
            }
            else {
                float factor = mixf(1.0f, pow(p, 1.0f + inStrength * 0.75f) * inRadius / dist, 1.0f - p);
                offsetX *= factor;
                offsetY *= factor;
            }

            if (inFeather > 0.0f) {
                float featherAmount = inFeather;
                float edgeX1 = smoothstepf(0.0f, featherAmount, uvX);
                float edgeX2 = smoothstepf(1.0f, 1.0f - featherAmount, uvX);
                float edgeY1 = smoothstepf(0.0f, featherAmount, uvY);
                float edgeY2 = smoothstepf(1.0f, 1.0f - featherAmount, uvY);
                float damping = edgeX1 * edgeX2 * edgeY1 * edgeY2;

                offsetX = prevOffsetX * (1.0f - damping) + offsetX * damping;
                offsetY = prevOffsetY * (1.0f - damping) + offsetY * damping;
            }
        }

        offsetX /= aspectRatio;

        float finalSrcX = offsetX + centerNormX;
        float finalSrcY = offsetY + centerNormY;

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
void InnerPinchBulge_CUDA(
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
    float radius,
    float feather,
    int useGaussian,
    int xTiles,
    int yTiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    InnerPinchBulgeKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height,
        centerX, centerY, strength, radius, feather, useGaussian, xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif  

#endif
