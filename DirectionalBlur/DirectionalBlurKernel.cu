#ifndef SDK_DIRECTIONAL_BLUR
#define SDK_DIRECTIONAL_BLUR

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif

GF_KERNEL_FUNCTION(DirectionalBlurKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inStrength))
    ((float)(inAngle)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        if (inStrength <= 0.001f) {
            float4 srcPixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
            WriteFloat4(srcPixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        float angle_rad = -inAngle * 0.017453292519943295769236907684886;  
        float velocity_x = cos(angle_rad) * inStrength;
        float velocity_y = -sin(angle_rad) * inStrength;

        float aspect = (float)inWidth / (float)inHeight;
        float adjusted_velocity_x = velocity_x * aspect;

        float texelSize_x = 1.0f / (float)inWidth;
        float texelSize_y = 1.0f / (float)inHeight;

        float speed_x = adjusted_velocity_x / texelSize_x;
        float speed_y = velocity_y / texelSize_y;
        float speed = sqrt(speed_x * speed_x + speed_y * speed_y);

        int nSamples = (int)(speed < 100.0f ? (speed > 1.0f ? speed : 1.0f) : 100.0f);
        if (nSamples <= 1) {
            float4 srcPixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);
            WriteFloat4(srcPixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        float normX = (float)inXY.x / (float)inWidth;
        float normY = (float)inXY.y / (float)inHeight;

        float4 origPixel = ReadFloat4(inSrc, inXY.y * inSrcPitch + inXY.x, !!in16f);

        float accumR = origPixel.x;
        float accumG = origPixel.y;
        float accumB = origPixel.z;
        float accumA = origPixel.w;

        float totalWeight = 1.0f;        

        float inv_nSamples_minus_1 = 1.0f / (float)(nSamples - 1);

        for (int i = 1; i < nSamples; i++)
        {
            float t = (float)i * inv_nSamples_minus_1 - 0.5f;
            float sample_norm_x = normX - adjusted_velocity_x * t;
            float sample_norm_y = normY - velocity_y * t;

            if (sample_norm_x >= 0.0f && sample_norm_x <= 1.0f &&
                sample_norm_y >= 0.0f && sample_norm_y <= 1.0f) {

                float sample_x = sample_norm_x * (float)inWidth;
                float sample_y = sample_norm_y * (float)inHeight;

                int x0 = (int)floor(sample_x);
                int y0 = (int)floor(sample_y);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                float fx = sample_x - (float)x0;
                float fy = sample_y - (float)y0;

                float4 p00 = { 0.0f, 0.0f, 0.0f, 0.0f };
                float4 p01 = { 0.0f, 0.0f, 0.0f, 0.0f };
                float4 p10 = { 0.0f, 0.0f, 0.0f, 0.0f };
                float4 p11 = { 0.0f, 0.0f, 0.0f, 0.0f };

                if (x0 >= 0 && x0 < inWidth && y0 >= 0 && y0 < inHeight) {
                    p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
                }
                if (x0 >= 0 && x0 < inWidth && y1 >= 0 && y1 < inHeight) {
                    p01 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
                }
                if (x1 >= 0 && x1 < inWidth && y0 >= 0 && y0 < inHeight) {
                    p10 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
                }
                if (x1 >= 0 && x1 < inWidth && y1 >= 0 && y1 < inHeight) {
                    p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);
                }

                float w00 = (1.0f - fx) * (1.0f - fy);
                float w01 = (1.0f - fx) * fy;
                float w10 = fx * (1.0f - fy);
                float w11 = fx * fy;

                float sampleR = p00.x * w00 + p01.x * w01 + p10.x * w10 + p11.x * w11;
                float sampleG = p00.y * w00 + p01.y * w01 + p10.y * w10 + p11.y * w11;
                float sampleB = p00.z * w00 + p01.z * w01 + p10.z * w10 + p11.z * w11;
                float sampleA = p00.w * w00 + p01.w * w01 + p10.w * w10 + p11.w * w11;

                accumR += sampleR;
                accumG += sampleG;
                accumB += sampleB;
                accumA += sampleA;

                totalWeight += 1.0f;
            }
        }

        float finalWeight = (totalWeight > 0.0f) ? (1.0f / totalWeight) : 1.0f;

        float4 result;
        result.x = accumR * finalWeight;
        result.y = accumG * finalWeight;
        result.z = accumB * finalWeight;
        result.w = accumA * finalWeight;

        WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}



#endif

#if __NVCC__
void DirectionalBlur_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float strength,
    float angle)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    DirectionalBlurKernel << <gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height, strength, angle);

    cudaDeviceSynchronize();
}
#endif  
#endif
