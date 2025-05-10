#ifndef Swirl
#define Swirl

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif
GF_KERNEL_FUNCTION(SwirlKernel,
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
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float width = (float)inWidth;
        float height = (float)inHeight;
        float x = (float)inXY.x / width;
        float y = (float)inXY.y / height;

        float centerX = inCenterX / width;
        float centerY = 1.0f - inCenterY / height;

        float dx = x - centerX;
        float dy = y - centerY;

        float convRateX = 1.0f;
        float convRateY = 1.0f;

        if (height > width) {
            convRateY = height / width;
        }
        else {
            convRateX = width / height;
        }

        dx *= convRateX;
        dy *= convRateY;

        float dist = sqrt(dx * dx + dy * dy);

        float srcX = x;
        float srcY = y;

        if (dist < inRadius) {
            float percent = (inRadius - dist) / inRadius;

            float T = inStrength;
            float A = (T <= 0.5f) ?
                ((T / 0.5f)) :
                (1.0f - ((T - 0.5f) / 0.5f));

            float theta = percent * percent * A * 8.0f * 3.14159f;
            float sinTheta = -sin(theta);
            float cosTheta = cos(theta);

            float newDx = dx * cosTheta - dy * sinTheta;
            float newDy = dx * sinTheta + dy * cosTheta;

            newDx /= convRateX;
            newDy /= convRateY;

            srcX = centerX + newDx;
            srcY = centerY + newDy;
        }

        if (inXTiles) {
            if (inMirror) {
                float fracPart = fmod(abs(srcX), 1.0f);
                int isOdd = (int)abs(srcX) & 1;
                srcX = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                srcX = srcX - floor(srcX);
            }
        }
        else if (srcX < 0.0f || srcX > 1.0f) {
            float4 transparent = { 0.0f, 0.0f, 0.0f, 0.0f };
            WriteFloat4(transparent, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }
        else {
            srcX = fmax(fmin(srcX, 0.9999f), 0.0f);
        }

        if (inYTiles) {
            if (inMirror) {
                float fracPart = fmod(abs(srcY), 1.0f);
                int isOdd = (int)abs(srcY) & 1;
                srcY = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                srcY = srcY - floor(srcY);
            }
        }
        else if (srcY < 0.0f || srcY > 1.0f) {
            float4 transparent = { 0.0f, 0.0f, 0.0f, 0.0f };
            WriteFloat4(transparent, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }
        else {
            srcY = fmax(fmin(srcY, 0.9999f), 0.0f);
        }

        srcX *= width;
        srcY *= height;

        int x1 = (int)srcX;
        int y1 = (int)srcY;
        int x2 = min(x1 + 1, inWidth - 1);
        int y2 = min(y1 + 1, inHeight - 1);

        float fx = srcX - x1;
        float fy = srcY - y1;

        float4 p1 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);
        float4 p2 = ReadFloat4(inSrc, y1 * inSrcPitch + x2, !!in16f);
        float4 p3 = ReadFloat4(inSrc, y2 * inSrcPitch + x1, !!in16f);
        float4 p4 = ReadFloat4(inSrc, y2 * inSrcPitch + x2, !!in16f);

        float4 result;
        result.x = (1 - fx) * (1 - fy) * p1.x + fx * (1 - fy) * p2.x + (1 - fx) * fy * p3.x + fx * fy * p4.x;
        result.y = (1 - fx) * (1 - fy) * p1.y + fx * (1 - fy) * p2.y + (1 - fx) * fy * p3.y + fx * fy * p4.y;
        result.z = (1 - fx) * (1 - fy) * p1.z + fx * (1 - fy) * p2.z + (1 - fx) * fy * p3.z + fx * fy * p4.z;
        result.w = (1 - fx) * (1 - fy) * p1.w + fx * (1 - fy) * p2.w + (1 - fx) * fy * p3.w + fx * fy * p4.w;

        WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
void Swirl_CUDA(
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
    int xTiles,
    int yTiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    SwirlKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f,
        width, height, centerX, centerY, strength, radius, xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif  
#endif