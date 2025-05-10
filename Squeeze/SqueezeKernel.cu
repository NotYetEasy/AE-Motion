#ifndef Squeeze
#define Squeeze

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif
GF_KERNEL_FUNCTION(SqueezeKernel,
	((GF_PTR_READ_ONLY(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((int)(inDstPitch))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight))
	((float)(inStrength))
	((int)(inXTiles))
	((int)(inYTiles))
	((int)(inMirror)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float width = (float)inWidth;
		float height = (float)inHeight;

		float normX = (float)inXY.x / width;
		float normY = (float)inXY.y / height;

		float stX = 2.0f * normX - 1.0f;
		float stY = 2.0f * normY - 1.0f;

		float str = inStrength / 2.0f;

		float absY = abs(stY);
		float absX = abs(stX);

		const float epsilon = 0.0001f;

		float xdiv = 1.0f + (1.0f - (absY * absY)) * -str;
		float ydiv = 1.0f + (1.0f - (absX * absX)) * str;

		xdiv = fmax(fmin(xdiv, 2.0f), epsilon);
		ydiv = fmax(fmin(ydiv, 2.0f), epsilon);

		stX /= xdiv;
		stY /= ydiv;

		float newNormX = stX / 2.0f + 0.5f;
		float newNormY = stY / 2.0f + 0.5f;

		float sourceX = newNormX * width;
		float sourceY = newNormY * height;

		bool outsideBounds = false;

		if (inXTiles) {
			if (inMirror) {
				float fracPart = fmod(abs(sourceX / width), 1.0f);
				int isOdd = (int)(sourceX / width) & 1;
				sourceX = isOdd ? width * (1.0f - fracPart) : width * fracPart;
			}
			else {
				sourceX = fmod(sourceX, width);
				if (sourceX < 0) sourceX += width;
			}
		}
		else {
			if (sourceX < 0 || sourceX >= width) {
				outsideBounds = true;
			}
		}

		if (inYTiles) {
			if (inMirror) {
				float fracPart = fmod(abs(sourceY / height), 1.0f);
				int isOdd = (int)(sourceY / height) & 1;
				sourceY = isOdd ? height * (1.0f - fracPart) : height * fracPart;
			}
			else {
				sourceY = fmod(sourceY, height);
				if (sourceY < 0) sourceY += height;
			}
		}
		else {
			if (sourceY < 0 || sourceY >= height) {
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
			int x1 = (int)sourceX;
			int y1 = (int)sourceY;
			int x2 = x1 + 1;
			int y2 = y1 + 1;

			float fx = sourceX - x1;
			float fy = sourceY - y1;

			x1 = max(min(x1, (int)inWidth - 1), 0);
			y1 = max(min(y1, (int)inHeight - 1), 0);
			x2 = max(min(x2, (int)inWidth - 1), 0);
			y2 = max(min(y2, (int)inHeight - 1), 0);

			float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);
			float4 p12 = ReadFloat4(inSrc, y2 * inSrcPitch + x1, !!in16f);
			float4 p21 = ReadFloat4(inSrc, y1 * inSrcPitch + x2, !!in16f);
			float4 p22 = ReadFloat4(inSrc, y2 * inSrcPitch + x2, !!in16f);

			float oneMinusFx = 1.0f - fx;
			float oneMinusFy = 1.0f - fy;

			float w00 = oneMinusFx * oneMinusFy;
			float w10 = fx * oneMinusFy;
			float w01 = oneMinusFx * fy;
			float w11 = fx * fy;

			pixel.w = p11.w * w00 + p21.w * w10 + p12.w * w01 + p22.w * w11;

			if (pixel.w > 0) {
				pixel.x = p11.x * w00 + p21.x * w10 + p12.x * w01 + p22.x * w11;
				pixel.y = p11.y * w00 + p21.y * w10 + p12.y * w01 + p22.y * w11;
				pixel.z = p11.z * w00 + p21.z * w10 + p12.z * w01 + p22.z * w11;
			}
			else {
				pixel.x = 0.0f;
				pixel.y = 0.0f;
				pixel.z = 0.0f;
			}
		}

		WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}
#endif
#if __NVCC__
void Squeeze_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float inStrength,
	int inXTiles,
	int inYTiles,
	int inMirror)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	SqueezeKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, inStrength, inXTiles, inYTiles, inMirror);

	cudaDeviceSynchronize();
}
#endif  
#endif