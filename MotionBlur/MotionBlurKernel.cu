#ifndef MotionBlur
#define MotionBlur

#include "PrGPU/KernelSupport/KernelCore.h" 
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#define fabs abs
#endif

GF_KERNEL_FUNCTION(MotionBlurKernel,
	((GF_PTR_READ_ONLY(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((int)(inDstPitch))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight))
	((float)(inMotionX))
	((float)(inMotionY))
	((float)(inTuneValue))
	((float)(inDownsampleX))
	((float)(inDownsampleY)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float velocity_x = inMotionX * inTuneValue * 0.7f * inDownsampleX;
		float velocity_y = inMotionY * inTuneValue * 0.7f * inDownsampleY;

		float max_dimension = fmax((float)inWidth, (float)inHeight);
		float texel_size = 1.0f / max_dimension;
		float normalized_velocity_x = velocity_x / (float)inWidth;
		float normalized_velocity_y = velocity_y / (float)inHeight;
		float speed = sqrt((normalized_velocity_x * normalized_velocity_x) +
			(normalized_velocity_y * normalized_velocity_y)) / texel_size;

		int nSamples = (int)fmin(100.0f, fmax(2.0f, speed));

		float4 accum = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
		float r = accum.x;
		float g = accum.y;
		float b = accum.z;
		float a = accum.w;

		for (int i = 1; i < nSamples; i++)
		{
			float offset_factor = ((float)i / (float)(nSamples - 1)) - 0.5f;
			float offset_x = velocity_x * offset_factor;
			float offset_y = velocity_y * offset_factor;

			float aspect_ratio = (float)inWidth / (float)inHeight;
			if (aspect_ratio != 1.0f) {
				offset_y *= aspect_ratio;
			}

			float sample_x = (float)inXY.x - offset_x;
			float sample_y = (float)inXY.y - offset_y;

			int ix = (int)(sample_x + 0.5f);
			int iy = (int)(sample_y + 0.5f);

			ix = (int)fmax(0.0f, fmin((float)ix, (float)inWidth - 1));
			iy = (int)fmax(0.0f, fmin((float)iy, (float)inHeight - 1));

			float4 sample = ReadFloat4(inSrc, iy * inSrcPitch + ix, !!in16f);
			r += sample.x;
			g += sample.y;
			b += sample.z;
			a += sample.w;
		}

		float inv_samples = 1.0f / (float)nSamples;
		float4 result;
		result.x = r * inv_samples;
		result.y = g * inv_samples;
		result.z = b * inv_samples;
		result.w = a * inv_samples;

		WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}

GF_KERNEL_FUNCTION(ScaleBlurKernel,
	((GF_PTR_READ_ONLY(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((int)(inDstPitch))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight))
	((float)(inScaleVelocity))
	((float)(inAnchorX))
	((float)(inAnchorY))
	((float)(inTuneValue))
	((float)(inDownsampleX))
	((float)(inDownsampleY)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float cx = inAnchorX * inDownsampleX;
		float cy = inAnchorY * inDownsampleY;

		float max_dimension = fmax((float)inWidth, (float)inHeight);
		float texel_size_x = 1.0f / max_dimension;
		float texel_size_y = 1.0f / max_dimension;

		float norm_cx = cx / (float)inWidth;
		float norm_cy = cy / (float)inHeight;
		float norm_x = (float)inXY.x / (float)inWidth;
		float norm_y = (float)inXY.y / (float)inHeight;

		float v_x = norm_x - norm_cx;
		float v_y = norm_y - norm_cy;

		float aspect_ratio = (float)inWidth / (float)inHeight;
		v_y *= aspect_ratio;

		float speed = fabs(inScaleVelocity * inTuneValue) / 2.0f;
		float length_v = sqrt(v_x * v_x + v_y * v_y);
		speed *= length_v;

		int nSamples = (int)fmin(100.01f, fmax(1.01f, speed));

		float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
		float r = pixel.x;
		float g = pixel.y;
		float b = pixel.z;
		float a = pixel.w;

		if (nSamples > 1)
		{
			float vnorm_x = 0.0f, vnorm_y = 0.0f;
			if (length_v > 0.0001f) {
				vnorm_x = v_x / length_v;
				vnorm_y = v_y / length_v;
			}

			vnorm_x *= texel_size_x * speed;
			vnorm_y *= texel_size_y * speed;

			for (int i = 1; i < nSamples; i++)
			{
				float offset_factor = ((float)i / (float)(nSamples - 1)) - 0.5f;
				float offset_x = vnorm_x * offset_factor;
				float offset_y = vnorm_y * offset_factor;

				offset_y = offset_y * (float)inWidth / (float)inHeight;

				float sample_x = (float)inXY.x - (offset_x * (float)inWidth);
				float sample_y = (float)inXY.y - (offset_y * (float)inHeight);

				int ix = (int)(sample_x + 0.5f);
				int iy = (int)(sample_y + 0.5f);

				ix = (int)fmax(0.0f, fmin((float)ix, (float)inWidth - 1));
				iy = (int)fmax(0.0f, fmin((float)iy, (float)inHeight - 1));

				float4 sample = ReadFloat4(inSrc, iy * inSrcPitch + ix, !!in16f);
				r += sample.x;
				g += sample.y;
				b += sample.z;
				a += sample.w;
			}

			float inv_samples = 1.0f / (float)nSamples;
			pixel.x = r * inv_samples;
			pixel.y = g * inv_samples;
			pixel.z = b * inv_samples;
			pixel.w = a * inv_samples;
		}

		WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}

GF_KERNEL_FUNCTION(AngleBlurKernel,
	((GF_PTR_READ_ONLY(float4))(inSrc))
	((GF_PTR(float4))(outDst)),
	((int)(inSrcPitch))
	((int)(inDstPitch))
	((int)(in16f))
	((unsigned int)(inWidth))
	((unsigned int)(inHeight))
	((float)(inRotationAngle))
	((float)(inAnchorX))
	((float)(inAnchorY))
	((float)(inTuneValue))
	((float)(inDownsampleX))
	((float)(inDownsampleY)),
	((uint2)(inXY)(KERNEL_XY)))
{
	if (inXY.x < inWidth && inXY.y < inHeight)
	{
		float cx = inAnchorX * inDownsampleX;
		float cy = inAnchorY * inDownsampleY;

		float angle_rad = fabs(inRotationAngle) * inTuneValue;

		if (angle_rad < 0.001f)
		{
			float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
			WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
			return;
		}

		float dx = (float)inXY.x - cx;
		float dy = (float)inXY.y - cy;

		float distance = sqrt(dx * dx + dy * dy);

		if (distance < 0.01f)
		{
			float4 pixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
			WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
			return;
		}

		float dx_norm = dx / distance;
		float dy_norm = dy / distance;

		float arc_length = distance * angle_rad;

		int nSamples = (int)fmax(2.0f, fmin((arc_length * 2.0f), 100.0f));

		float r = 0.0f;
		float g = 0.0f;
		float b = 0.0f;
		float a = 0.0f;

		for (int i = 0; i < nSamples; i++)
		{
			float sample_angle = -angle_rad / 2.0f + angle_rad * (float)i / ((float)nSamples - 1.0f);

			float sin_angle = sin(sample_angle);
			float cos_angle = cos(sample_angle);

			float rotated_dx = dx_norm * cos_angle - dy_norm * sin_angle;
			float rotated_dy = dx_norm * sin_angle + dy_norm * cos_angle;

			float sample_x = cx + (rotated_dx * distance);
			float sample_y = cy + (rotated_dy * distance);

			int ix = (int)(sample_x + 0.5f);
			int iy = (int)(sample_y + 0.5f);

			ix = (int)fmax(0.0f, fmin((float)ix, (float)inWidth - 1));
			iy = (int)fmax(0.0f, fmin((float)iy, (float)inHeight - 1));

			float4 sample = ReadFloat4(inSrc, iy * inSrcPitch + ix, !!in16f);
			r += sample.x;
			g += sample.y;
			b += sample.z;
			a += sample.w;
		}

		float inv_samples = 1.0f / (float)nSamples;
		float4 result;
		result.x = r * inv_samples;
		result.y = g * inv_samples;
		result.z = b * inv_samples;
		result.w = a * inv_samples;

		WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
	}
}

#endif
#if __NVCC__
void Motion_Blur_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float motionX,
	float motionY,
	float tuneValue,
	float downsampleX,
	float downsampleY)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	MotionBlurKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, motionX, motionY, tuneValue, downsampleX, downsampleY);

	cudaDeviceSynchronize();
}

void Scale_Blur_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float scaleVelocity,
	float anchorX,
	float anchorY,
	float tuneValue,
	float downsampleX,
	float downsampleY)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	ScaleBlurKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, scaleVelocity, anchorX, anchorY, tuneValue, downsampleX, downsampleY);

	cudaDeviceSynchronize();
}

void Angle_Blur_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float rotationAngle,
	float anchorX,
	float anchorY,
	float tuneValue,
	float downsampleX,
	float downsampleY)
{
	dim3 blockDim(16, 16, 1);
	dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

	AngleBlurKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, rotationAngle, anchorX, anchorY, tuneValue, downsampleX, downsampleY);

	cudaDeviceSynchronize();
}
#endif  
#endif
