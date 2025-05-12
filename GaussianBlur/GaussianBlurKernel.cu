#ifndef GaussianBlur
#define GaussianBlur

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif
GF_KERNEL_FUNCTION(GaussianBlurHorizontalKernel,
	((GF_PTR_READ_ONLY(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((int)(inDstPitch))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight))
	((float)(inStrength)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

		float previewSize = sqrt((float)(inWidth * inWidth + inHeight * inHeight));
		float kernelSize = (inStrength * 1.14f / 4.0f / 3.0f * previewSize);

		float numBlurPixelsPerSide = fmax(1.0f, kernelSize);
		float adjSigma = numBlurPixelsPerSide / 2.14596602f;

		float incrementalGaussianX = 1.0f / (sqrt(2.0f * 3.14159265f) * adjSigma);
		float incrementalGaussianY = exp(-0.5f / (adjSigma * adjSigma));
		float incrementalGaussianZ = incrementalGaussianY * incrementalGaussianY;

		float4 result = { 0.0f, 0.0f, 0.0f, 0.0f };
		float coefficientSum = 0.0f;

		result.x += pixel.x * incrementalGaussianX;
		result.y += pixel.y * incrementalGaussianX;
		result.z += pixel.z * incrementalGaussianX;
		result.w += pixel.w * incrementalGaussianX;
		coefficientSum += incrementalGaussianX;

		for (float i = 1.0f; i <= numBlurPixelsPerSide; i += 2.0f)
		{
			float offset0 = i;
			float offset1 = i + 1.0f;

			incrementalGaussianX *= incrementalGaussianY;
			incrementalGaussianY *= incrementalGaussianZ;
			float weight0 = incrementalGaussianX;
			coefficientSum += (2.0f * weight0);

			incrementalGaussianX *= incrementalGaussianY;
			incrementalGaussianY *= incrementalGaussianZ;
			float weight1 = incrementalGaussianX;
			coefficientSum += (2.0f * weight1);

			float weightL = weight0 + weight1;
			float offsetL = (offset0 * weight0 + offset1 * weight1) / weightL;

			int leftX = (int)(inXY.x - offsetL);
			int rightX = (int)(inXY.x + offsetL);

			leftX = max(0, leftX);
			rightX = min(rightX, (int)inWidth - 1);

			float4 leftPixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + leftX, !!in16f);
			float4 rightPixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + rightX, !!in16f);

			result.x += (leftPixel.x * weightL + rightPixel.x * weightL);
			result.y += (leftPixel.y * weightL + rightPixel.y * weightL);
			result.z += (leftPixel.z * weightL + rightPixel.z * weightL);
			result.w += (leftPixel.w * weightL + rightPixel.w * weightL);
		}

		if (coefficientSum > 0.0f)
		{
			result.x /= coefficientSum;
			result.y /= coefficientSum;
			result.z /= coefficientSum;
			result.w /= coefficientSum;
		}

		WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}

GF_KERNEL_FUNCTION(GaussianBlurVerticalKernel,
	((GF_PTR_READ_ONLY(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((int)(inDstPitch))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight))
	((float)(inStrength)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

		float previewSize = sqrt((float)(inWidth * inWidth + inHeight * inHeight));
		float kernelSize = (inStrength * 1.14f / 4.0f / 3.0f * previewSize);

		float numBlurPixelsPerSide = fmax(1.0f, kernelSize);
		float adjSigma = numBlurPixelsPerSide / 2.14596602f;

		float incrementalGaussianX = 1.0f / (sqrt(2.0f * 3.14159265f) * adjSigma);
		float incrementalGaussianY = exp(-0.5f / (adjSigma * adjSigma));
		float incrementalGaussianZ = incrementalGaussianY * incrementalGaussianY;

		float4 result = { 0.0f, 0.0f, 0.0f, 0.0f };
		float coefficientSum = 0.0f;

		result.x += pixel.x * incrementalGaussianX;
		result.y += pixel.y * incrementalGaussianX;
		result.z += pixel.z * incrementalGaussianX;
		result.w += pixel.w * incrementalGaussianX;
		coefficientSum += incrementalGaussianX;

		for (float i = 1.0f; i <= numBlurPixelsPerSide; i += 2.0f)
		{
			float offset0 = i;
			float offset1 = i + 1.0f;

			incrementalGaussianX *= incrementalGaussianY;
			incrementalGaussianY *= incrementalGaussianZ;
			float weight0 = incrementalGaussianX;
			coefficientSum += (2.0f * weight0);

			incrementalGaussianX *= incrementalGaussianY;
			incrementalGaussianY *= incrementalGaussianZ;
			float weight1 = incrementalGaussianX;
			coefficientSum += (2.0f * weight1);

			float weightL = weight0 + weight1;
			float offsetL = (offset0 * weight0 + offset1 * weight1) / weightL;

			int topY = (int)(inXY.y - offsetL);
			int bottomY = (int)(inXY.y + offsetL);

			topY = max(0, topY);
			bottomY = min(bottomY, (int)inHeight - 1);

			float4 topPixel = ReadFloat4(inSrc, topY * inSrcPitch + inXY.x, !!in16f);
			float4 bottomPixel = ReadFloat4(inSrc, bottomY * inSrcPitch + inXY.x, !!in16f);

			result.x += (topPixel.x * weightL + bottomPixel.x * weightL);
			result.y += (topPixel.y * weightL + bottomPixel.y * weightL);
			result.z += (topPixel.z * weightL + bottomPixel.z * weightL);
			result.w += (topPixel.w * weightL + bottomPixel.w * weightL);
		}

		if (coefficientSum > 0.0f)
		{
			result.x /= coefficientSum;
			result.y /= coefficientSum;
			result.z /= coefficientSum;
			result.w /= coefficientSum;
		}

		WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}

#endif
#if __NVCC__
void GaussianBlur_Horizontal_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float strength)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	GaussianBlurHorizontalKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, strength);

	cudaDeviceSynchronize();
}

void GaussianBlur_Vertical_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float strength)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	GaussianBlurVerticalKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, strength);

	cudaDeviceSynchronize();
}
#endif  
#endif