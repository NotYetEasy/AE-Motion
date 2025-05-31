#ifndef RasterTransform
#define RasterTransform

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#define fmod(x,y) (x - y * floor(x/y))
#define fabsf abs
#define sinf sin
#define cosf cos
#define floorf floor
#define fmodf(x,y) fmod(x,y)
#endif

GF_DEVICE_FUNCTION float min_custom(float a, float b) {
    return a < b ? a : b;
}

GF_DEVICE_FUNCTION float max_custom(float a, float b) {
    return a > b ? a : b;
}

GF_DEVICE_FUNCTION float fmod_custom(float x, float y) {
    return x - y * floorf(x / y);
}

GF_KERNEL_FUNCTION(RasterTransformKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inScale))
    ((float)(inAngle))
    ((float)(inOffsetX))
    ((float)(inOffsetY))
    ((int)(inMaskToLayer))
    ((float)(inAlpha))
    ((float)(inFill))
    ((int)(inSample))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float width = (float)inWidth;
        float height = (float)inHeight;

        float st_x = (float)inXY.x / width;
        float st_y = (float)inXY.y / height;

        st_x -= inOffsetX / 500.0f;
        st_y -= inOffsetY / 500.0f;

        st_x -= 0.5f;
        st_y -= 0.5f;

        st_x *= width / height;

        float angle_rad = inAngle * 0.0174533f;
        float cos_angle = cosf(angle_rad);
        float sin_angle = sinf(angle_rad);

        float rotated_x = st_x * cos_angle - st_y * sin_angle;
        float rotated_y = st_x * sin_angle + st_y * cos_angle;
        st_x = rotated_x;
        st_y = rotated_y;

        st_x /= inScale;
        st_y /= inScale;

        st_x /= width / height;

        st_x += 0.5f;
        st_y += 0.5f;

        float sample_x = st_x * width;
        float sample_y = st_y * height;

        float4 transformedPixel = { 0.0f, 0.0f, 0.0f, 0.0f };
        float4 basePixel = { 0.0f, 0.0f, 0.0f, 0.0f };

        if (inMaskToLayer || inFill > 0.0001f) {
            basePixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
        }

        bool outsideBounds = false;

        if (inXTiles) {
            if (inMirror) {
                float fracPart = fmod_custom(fabsf(sample_x / width), 1.0f);
                int isOdd = (int)floorf(fabsf(sample_x / width)) & 1;
                sample_x = isOdd ? (1.0f - fracPart) * width : fracPart * width;
            }
            else {
                sample_x = fmod_custom(fmod_custom(sample_x, width) + width, width);
            }
        }
        else {
            if (sample_x < 0 || sample_x >= width) {
                outsideBounds = true;
            }
        }

        if (inYTiles) {
            if (inMirror) {
                float fracPart = fmod_custom(fabsf(sample_y / height), 1.0f);
                int isOdd = (int)floorf(fabsf(sample_y / height)) & 1;
                sample_y = isOdd ? (1.0f - fracPart) * height : fracPart * height;
            }
            else {
                sample_y = fmod_custom(fmod_custom(sample_y, height) + height, height);
            }
        }
        else {
            if (sample_y < 0 || sample_y >= height) {
                outsideBounds = true;
            }
        }

        if (!outsideBounds) {
            if (inSample == 0) {     
                unsigned int rounded_x = (unsigned int)(sample_x + 0.5f);
                unsigned int rounded_y = (unsigned int)(sample_y + 0.5f);

                rounded_x = min(rounded_x, inWidth - 1);
                rounded_y = min(rounded_y, inHeight - 1);

                transformedPixel = ReadFloat4(inSrc, rounded_y * inSrcPitch + rounded_x, !!in16f);
            }
            else {    
                unsigned int x1 = (unsigned int)sample_x;
                unsigned int y1 = (unsigned int)sample_y;
                unsigned int x2 = min(x1 + 1, inWidth - 1);
                unsigned int y2 = min(y1 + 1, inHeight - 1);

                float fx = sample_x - x1;
                float fy = sample_y - y1;

                float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);
                float4 p12 = ReadFloat4(inSrc, y1 * inSrcPitch + x2, !!in16f);
                float4 p21 = ReadFloat4(inSrc, y2 * inSrcPitch + x1, !!in16f);
                float4 p22 = ReadFloat4(inSrc, y2 * inSrcPitch + x2, !!in16f);

                transformedPixel.x = (1 - fx) * (1 - fy) * p11.x + fx * (1 - fy) * p12.x + (1 - fx) * fy * p21.x + fx * fy * p22.x;
                transformedPixel.y = (1 - fx) * (1 - fy) * p11.y + fx * (1 - fy) * p12.y + (1 - fx) * fy * p21.y + fx * fy * p22.y;
                transformedPixel.z = (1 - fx) * (1 - fy) * p11.z + fx * (1 - fy) * p12.z + (1 - fx) * fy * p21.z + fx * fy * p22.z;
                transformedPixel.w = (1 - fx) * (1 - fy) * p11.w + fx * (1 - fy) * p12.w + (1 - fx) * fy * p21.w + fx * fy * p22.w;
            }
        }

        float4 outPixel;

        if (inMaskToLayer) {
            float baseA = basePixel.w;
            float fillF = inFill;
            float alphaF = inAlpha;

            outPixel.x = basePixel.x * fillF * (1.0f - transformedPixel.w * baseA * alphaF) + transformedPixel.x * baseA * alphaF;
            outPixel.y = basePixel.y * fillF * (1.0f - transformedPixel.w * baseA * alphaF) + transformedPixel.y * baseA * alphaF;
            outPixel.z = basePixel.z * fillF * (1.0f - transformedPixel.w * baseA * alphaF) + transformedPixel.z * baseA * alphaF;
            outPixel.w = basePixel.w;
        }
        else if (inFill > 0.0001f) {
            float fillF = inFill;
            float alphaF = inAlpha;

            outPixel.x = basePixel.x * fillF * (1.0f - transformedPixel.w * alphaF) + transformedPixel.x * alphaF;
            outPixel.y = basePixel.y * fillF * (1.0f - transformedPixel.w * alphaF) + transformedPixel.y * alphaF;
            outPixel.z = basePixel.z * fillF * (1.0f - transformedPixel.w * alphaF) + transformedPixel.z * alphaF;
            outPixel.w = basePixel.w * fillF * (1.0f - transformedPixel.w * alphaF) + transformedPixel.w * alphaF;
        }
        else {
            float alphaF = inAlpha;
            outPixel.x = transformedPixel.x * alphaF;
            outPixel.y = transformedPixel.y * alphaF;
            outPixel.z = transformedPixel.z * alphaF;
            outPixel.w = transformedPixel.w * alphaF;
        }

        WriteFloat4(outPixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif
#if __NVCC__
void RasterTransform_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float scale,
    float angle,
    float offsetX,
    float offsetY,
    int maskToLayer,
    float alpha,
    float fill,
    int sample,
    int xTiles,
    int yTiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    RasterTransformKernel << < gridDim, blockDim, 0 >> > (
        (float4 const*)src,
        (float4*)dst,
        srcPitch,
        dstPitch,
        is16f,
        width,
        height,
        scale,
        angle,
        offsetX,
        offsetY,
        maskToLayer,
        alpha,
        fill,
        sample,
        xTiles,
        yTiles,
        mirror
        );

    cudaDeviceSynchronize();
}
#endif  
#endif