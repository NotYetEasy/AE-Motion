#ifndef Transforms
#define Transforms

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif

GF_KERNEL_FUNCTION(TransformKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inXPos))
    ((float)(inYPos))
    ((float)(inRotation))
    ((float)(inScale))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float curr_x = (float)inXY.x;
        float curr_y = (float)inXY.y;

        float center_x = inWidth / 2.0f;
        float center_y = inHeight / 2.0f;

        float dx = curr_x - inXPos;
        float dy = curr_y - inYPos;

        dx /= inScale / 100.0f;
        dy /= inScale / 100.0f;

        float cos_theta = cos(-inRotation * 3.14159265358979323846f / 180.0f);
        float sin_theta = sin(-inRotation * 3.14159265358979323846f / 180.0f);
        float rotated_x = dx * cos_theta - dy * sin_theta;
        float rotated_y = dx * sin_theta + dy * cos_theta;

        float src_x = rotated_x + center_x;
        float src_y = rotated_y + center_y;

        if (inXTiles)
        {
            if (inMirror)
            {
                float fracPartX = fmod(abs(src_x / inWidth), 1.0f);
                int isOddX = (int)(src_x / inWidth) & 1;
                src_x = isOddX ? (1.0f - fracPartX) * inWidth : fracPartX * inWidth;
            }
            else
            {
                src_x = fmod(src_x, (float)inWidth);
                if (src_x < 0) src_x += inWidth;
            }
        }

        if (inYTiles)
        {
            if (inMirror)
            {
                float fracPartY = fmod(abs(src_y / inHeight), 1.0f);
                int isOddY = (int)(src_y / inHeight) & 1;
                src_y = isOddY ? (1.0f - fracPartY) * inHeight : fracPartY * inHeight;
            }
            else
            {
                src_y = fmod(src_y, (float)inHeight);
                if (src_y < 0) src_y += inHeight;
            }
        }

        float4 pixel;

        if (src_x >= 0 && src_x < inWidth && src_y >= 0 && src_y < inHeight)
        {
            int x_int = (int)src_x;
            int y_int = (int)src_y;
            float x_frac = src_x - x_int;
            float y_frac = src_y - y_int;

            int x0 = max(0, min(inWidth - 1, x_int));
            int x1 = max(0, min(inWidth - 1, x_int + 1));
            int y0 = max(0, min(inHeight - 1, y_int));
            int y1 = max(0, min(inHeight - 1, y_int + 1));

            float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
            float4 p01 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
            float4 p10 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
            float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

            pixel.x = (1.0f - x_frac) * (1.0f - y_frac) * p00.x +
                x_frac * (1.0f - y_frac) * p01.x +
                (1.0f - x_frac) * y_frac * p10.x +
                x_frac * y_frac * p11.x;

            pixel.y = (1.0f - x_frac) * (1.0f - y_frac) * p00.y +
                x_frac * (1.0f - y_frac) * p01.y +
                (1.0f - x_frac) * y_frac * p10.y +
                x_frac * y_frac * p11.y;

            pixel.z = (1.0f - x_frac) * (1.0f - y_frac) * p00.z +
                x_frac * (1.0f - y_frac) * p01.z +
                (1.0f - x_frac) * y_frac * p10.z +
                x_frac * y_frac * p11.z;

            pixel.w = (1.0f - x_frac) * (1.0f - y_frac) * p00.w +
                x_frac * (1.0f - y_frac) * p01.w +
                (1.0f - x_frac) * y_frac * p10.w +
                x_frac * y_frac * p11.w;
        }
        else
        {
            pixel.x = 0.0f;
            pixel.y = 0.0f;
            pixel.z = 0.0f;
            pixel.w = 0.0f;
        }

        WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
void Transform_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float xPos,
    float yPos,
    float rotation,
    float scale,
    int xTiles,
    int yTiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    TransformKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst,
        srcPitch, dstPitch, is16f, width, height,
        xPos, yPos, rotation, scale,
        xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif  
#endif