#ifndef LinearStreaks
#define LinearStreaks

#include "PrGPU/KernelSupport/KernelCore.h" 
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#define __device__
#endif

__device__ int min_int(int a, int b) {
	return a < b ? a : b;
}

__device__ int max_int(int a, int b) {
	return a > b ? a : b;
}

__device__ float fminf_custom(float a, float b) {
	return a < b ? a : b;
}

__device__ float fmaxf_custom(float a, float b) {
	return a > b ? a : b;
}

GF_KERNEL_FUNCTION(LinearStreaksKernel,
	((GF_PTR_READ_ONLY(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((int)(inDstPitch))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight))
	((float)(inStrength))
	((float)(inAngle))
	((float)(inAlpha))
	((float)(inBias))
	((int)(inRMode))
	((int)(inGMode))
	((int)(inBMode))
	((int)(inAMode)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float4 originalColor = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
		float4 outColor = originalColor;

		float rad = inAngle * 0.0174533f;

		float velocityX = cos(rad) * inStrength;
		float velocityY = -sin(rad) * inStrength;

		velocityX *= inHeight / (float)inWidth;

		float texelSizeX = 1.0f / inWidth;
		float texelSizeY = 1.0f / inHeight;

		float speed = sqrt(velocityX * velocityX / (texelSizeX * texelSizeX) +
			velocityY * velocityY / (texelSizeY * texelSizeY));

		float fSpeed = fmaxf_custom(1.0f, fminf_custom(speed, 100.0f));
		int nSamples = (int)fSpeed;

		if (outColor.w > 0.0f) {
			outColor.x /= outColor.w;
			outColor.y /= outColor.w;
			outColor.z /= outColor.w;
		}

		for (int i = 1; i < nSamples; i++) {
			float t = (float)i / (float)(nSamples - 1) - (0.5f + inBias / 2.0f);
			float offsetX = velocityX * t;
			float offsetY = velocityY * t;

			float sampleX = (inXY.x + 0.5f) / inWidth - offsetX;
			float sampleY = (inXY.y + 0.5f) / inHeight - offsetY;

			bool insideTexture = (sampleX >= 0.0f && sampleX <= 1.0f &&
				sampleY >= 0.0f && sampleY <= 1.0f);

			float4 c = { 0, 0, 0, 0 };
			if (insideTexture) {
				int pixX = (int)(sampleX * inWidth);
				int pixY = (int)(sampleY * inHeight);

				pixX = max_int(0, min_int(pixX, (int)inWidth - 1));
				pixY = max_int(0, min_int(pixY, (int)inHeight - 1));

				c = ReadFloat4(inSrc, pixY * inSrcPitch + pixX, !!in16f);

				if (c.w > 0.0f) {
					c.x /= c.w;
					c.y /= c.w;
					c.z /= c.w;
				}
			}

			if (inRMode == 0)   
				outColor.x = fminf_custom(outColor.x, c.x);
			else if (inRMode == 1)   
				outColor.x = fmaxf_custom(outColor.x, c.x);
			else if (inRMode == 3)   
				outColor.x += c.x;

			if (inGMode == 0)   
				outColor.y = fminf_custom(outColor.y, c.y);
			else if (inGMode == 1)   
				outColor.y = fmaxf_custom(outColor.y, c.y);
			else if (inGMode == 3)   
				outColor.y += c.y;

			if (inBMode == 0)   
				outColor.z = fminf_custom(outColor.z, c.z);
			else if (inBMode == 1)   
				outColor.z = fmaxf_custom(outColor.z, c.z);
			else if (inBMode == 3)   
				outColor.z += c.z;

			if (inAMode == 0)   
				outColor.w = fminf_custom(outColor.w, c.w);
			else if (inAMode == 1)   
				outColor.w = fmaxf_custom(outColor.w, c.w);
			else if (inAMode == 3)   
				outColor.w += c.w;
		}

		float nSamplesFloat = (float)nSamples;
		outColor.x /= (inRMode == 3) ? nSamplesFloat : 1.0f;
		outColor.y /= (inGMode == 3) ? nSamplesFloat : 1.0f;
		outColor.z /= (inBMode == 3) ? nSamplesFloat : 1.0f;
		outColor.w /= (inAMode == 3) ? nSamplesFloat : 1.0f;

		outColor.x *= outColor.w;
		outColor.y *= outColor.w;
		outColor.z *= outColor.w;

		float4 finalColor;
		finalColor.x = originalColor.x * (1.0f - inAlpha) + outColor.x * inAlpha;
		finalColor.y = originalColor.y * (1.0f - inAlpha) + outColor.y * inAlpha;
		finalColor.z = originalColor.z * (1.0f - inAlpha) + outColor.z * inAlpha;
		finalColor.w = originalColor.w * (1.0f - inAlpha) + outColor.w * inAlpha;

		WriteFloat4(finalColor, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}
#endif

#if __NVCC__
void Linear_Streaks_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float strength,
	float angle,
	float alpha,
	float bias,
	int rMode,
	int gMode,
	int bMode,
	int aMode)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	LinearStreaksKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, strength, angle, alpha, bias, rMode, gMode, bMode, aMode);

	cudaDeviceSynchronize();
}
#endif  
#endif
