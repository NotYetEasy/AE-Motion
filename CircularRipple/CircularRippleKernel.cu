#ifndef CircularRipple
#define CircularRipple

#include "PrGPU/KernelSupport/KernelCore.h" 
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#define fmod(x,y) (x - y * floor(x/y))
#define fabsf abs
#define sinf sin
#define floorf floor
#endif

GF_DEVICE_FUNCTION float min_custom(float a, float b) {
    return a < b ? a : b;
}

GF_DEVICE_FUNCTION float max_custom(float a, float b) {
    return a > b ? a : b;
}

GF_DEVICE_FUNCTION int min_int(int a, int b) {
    return a < b ? a : b;
}

GF_DEVICE_FUNCTION int max_int(int a, int b) {
    return a > b ? a : b;
}

GF_DEVICE_FUNCTION float fmod_custom(float x, float y) {
    return x - y * floorf(x / y);
}

GF_KERNEL_FUNCTION(CircularRippleKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inCenterX))
    ((float)(inCenterY))
    ((float)(inFrequency))
    ((float)(inStrength))
    ((float)(inPhase))
    ((float)(inRadius))
    ((float)(inFeather))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

        float uvX = (float)inXY.x / inWidth;
        float uvY = (float)inXY.y / inHeight;

        float centerNormX = inCenterX / inWidth;
        float centerNormY = inCenterY / inHeight;

        float offsetX = uvX - centerNormX;
        float offsetY = uvY - centerNormY;

        offsetY *= (float)inHeight / inWidth;

        float dist = sqrt(offsetX * offsetX + offsetY * offsetY);

        float featherSize = inRadius * 0.5f * inFeather;
        float innerRadius = max_custom(0.0f, inRadius - featherSize);
        float outerRadius = max_custom(innerRadius + 0.00001f, inRadius + featherSize);

        float damping;
        if (dist >= outerRadius) {
            damping = 0.0f;
        }
        else if (dist <= innerRadius) {
            damping = 1.0f;
        }
        else {
            float t = (dist - innerRadius) / (outerRadius - innerRadius);
            t = 1.0f - t;   
            damping = t * t * (3.0f - 2.0f * t);
        }

        const float PI = 3.14159265358979323846f;
        float angle = (dist * inFrequency * PI * 2.0f) + (inPhase * PI * 2.0f);
        float sinVal = sinf(angle);

        float normX = 0.0f, normY = 0.0f;
        if (dist > 0.0001f) {
            normX = offsetX / dist;
            normY = offsetY / dist;
        }

        float strength_factor = inStrength / 2.0f;
        float offsetFactorX = sinVal * strength_factor * normX * damping;
        float offsetFactorY = sinVal * strength_factor * normY * damping;

        offsetX += offsetFactorX;
        offsetY += offsetFactorY;

        offsetY /= (float)inHeight / inWidth;

        float finalSrcX = (offsetX + centerNormX);
        float finalSrcY = (offsetY + centerNormY);

        bool outsideBounds = false;

        if (inXTiles) {
            if (inMirror) {
                float fracPart = fmod_custom(fabsf(finalSrcX), 1.0f);
                int isOdd = (int)floorf(fabsf(finalSrcX)) & 1;
                finalSrcX = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                finalSrcX = fmod_custom(fmod_custom(finalSrcX, 1.0f) + 1.0f, 1.0f);
            }
        }
        else {
            if (finalSrcX < 0.0f || finalSrcX >= 1.0f) {
                outsideBounds = true;
            }
        }

        if (inYTiles) {
            if (inMirror) {
                float fracPart = fmod_custom(fabsf(finalSrcY), 1.0f);
                int isOdd = (int)floorf(fabsf(finalSrcY)) & 1;
                finalSrcY = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                finalSrcY = fmod_custom(fmod_custom(finalSrcY, 1.0f) + 1.0f, 1.0f);
            }
        }
        else {
            if (finalSrcY < 0.0f || finalSrcY >= 1.0f) {
                outsideBounds = true;
            }
        }

        float4 result;

        if (outsideBounds) {
            result.x = 0.0f;
            result.y = 0.0f;
            result.z = 0.0f;
            result.w = 0.0f;
        }
        else {
            float x = finalSrcX * inWidth;
            float y = finalSrcY * inHeight;

            x = fmax(0.0f, fmin(inWidth - 1.001f, x));
            y = fmax(0.0f, fmin(inHeight - 1.001f, y));

            int x0 = (int)floorf(x);
            int y0 = (int)floorf(y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            float fx = x - x0;
            float fy = y - y0;

            x0 = max_int(0, min_int(x0, (int)inWidth - 1));
            y0 = max_int(0, min_int(y0, (int)inHeight - 1));
            x1 = max_int(0, min_int(x1, (int)inWidth - 1));
            y1 = max_int(0, min_int(y1, (int)inHeight - 1));

            float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
            float4 p01 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
            float4 p10 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
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
        }

        WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}


#endif
#if __NVCC__
void CircularRipple_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float centerX,
    float centerY,
    float frequency,
    float strength,
    float phase,
    float radius,
    float feather,
    int xTiles,
    int yTiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    CircularRippleKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, centerX, centerY, frequency, strength, phase, radius, feather, xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif  
#endif
