#ifndef Tiles
#define Tiles

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif


GF_KERNEL_FUNCTION(TilesKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inCropF)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float centerX = inWidth / 2.0f;
        float centerY = inHeight / 2.0f;

        float pixelSizeX = 1.0f / inWidth;
        float pixelSizeY = 1.0f / inHeight;
        float pixelSize = (pixelSizeX + pixelSizeY) / 2.0f;

        float adjustedCropF = inCropF + pixelSize * 4.0f;

        float srcX = centerX + (inXY.x - centerX) / adjustedCropF;
        float srcY = centerY + (inXY.y - centerY) / adjustedCropF;

        if (srcX < 0.0f || srcX >= inWidth || srcY < 0.0f || srcY >= inHeight) {
            float4 transparent = { 0.0f, 0.0f, 0.0f, 0.0f };
            WriteFloat4(transparent, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        int x0 = (int)srcX;
        int y0 = (int)srcY;
        int x1 = min(x0 + 1, inWidth - 1);
        int y1 = min(y0 + 1, inHeight - 1);

        float fx = srcX - x0;
        float fy = srcY - y0;

        float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
        float4 p01 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
        float4 p10 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
        float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

        float4 pixel;
        pixel.x = (1.0f - fx) * (1.0f - fy) * p00.x + fx * (1.0f - fy) * p01.x + (1.0f - fx) * fy * p10.x + fx * fy * p11.x;
        pixel.y = (1.0f - fx) * (1.0f - fy) * p00.y + fx * (1.0f - fy) * p01.y + (1.0f - fx) * fy * p10.y + fx * fy * p11.y;
        pixel.z = (1.0f - fx) * (1.0f - fy) * p00.z + fx * (1.0f - fy) * p01.z + (1.0f - fx) * fy * p10.z + fx * fy * p11.z;
        pixel.w = (1.0f - fx) * (1.0f - fy) * p00.w + fx * (1.0f - fy) * p01.w + (1.0f - fx) * fy * p10.w + fx * fy * p11.w;

        WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}

#endif

#if __NVCC__
void Tiles_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float cropF)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	TilesKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, cropF);

	cudaDeviceSynchronize();
}
#endif  
#endif