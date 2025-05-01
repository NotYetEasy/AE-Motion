#ifndef SDK_INVERT_PROC_AMP
#define SDK_INVERT_PROC_AMP

#include "PrGPU/KernelSupport/KernelCore.h" //includes KernelWrapper.h
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#define __device__
#define fabs abs
#define fmodf fmod
#define fmaxf max
#define fminf min
#endif

// Helper function to implement mix operation (GLSL-style lerp)
__device__ float mixf(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

// Helper function to implement smoothstep operation
__device__ float smoothstepf(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

// Device-friendly min and max functions
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

// PinchBulge kernel function - matches the original CPU implementation
GF_KERNEL_FUNCTION(PinchBulgeKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(centerX))
    ((float)(centerY))
    ((float)(strength))
    ((float)(radius))
    ((int)(xTiles))
    ((int)(yTiles))
    ((int)(mirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        // Get input layer dimensions
        float width = (float)inWidth;
        float height = (float)inHeight;

        // Calculate normalized UV coordinates (0-1 range)
        float uvX = (float)inXY.x / width;
        float uvY = (float)inXY.y / height;

        // Calculate center in normalized coordinates
        float centerNormX = centerX / width;
        float centerNormY = centerY / height;

        // Offset UV by center
        float offsetX = uvX - centerNormX;
        float offsetY = uvY - centerNormY;

        // Adjust for aspect ratio (multiply X by aspect ratio)
        float aspectRatio = width / height;
        offsetX *= aspectRatio;

        // Calculate distance from center
        float dist = sqrt(offsetX * offsetX + offsetY * offsetY);

        // Apply distortion if within radius
        if (dist < radius) {
            float p = dist / radius;

            if (strength > 0.0f) {
                // Bulge effect - EXACTLY matching the original CPU implementation
                float factor = mixf(1.0f, smoothstepf(0.0f, radius / fmaxf_custom(dist, 0.001f), p), strength * 0.75f);
                offsetX *= factor;
                offsetY *= factor;
            }
            else {
                // Pinch effect - EXACTLY matching the original CPU implementation
                float factor = mixf(1.0f, pow(p, 1.0f + strength * 0.75f) * radius / fmaxf_custom(dist, 0.001f), 1.0f - p);
                offsetX *= factor;
                offsetY *= factor;
            }
        }

        // Undo aspect ratio adjustment
        offsetX /= aspectRatio;

        // Calculate final sampling coordinates in normalized space (0-1)
        float finalSrcX = offsetX + centerNormX;
        float finalSrcY = offsetY + centerNormY;

        // Handle tiling and mirroring
        bool outsideBounds = false;

        // Handle X tiling
        if (xTiles != 0) {
            if (mirror != 0) {
                // Mirror tiling: create ping-pong pattern
                float fracPart = fmodf(fabs(finalSrcX), 1.0f);
                int isOdd = (int)floor(fabs(finalSrcX)) & 1;
                finalSrcX = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                // Regular repeat tiling
                finalSrcX = finalSrcX - floor(finalSrcX);
            }
        }
        else if (finalSrcX < 0.0f || finalSrcX >= 1.0f) {
            outsideBounds = true;
        }

        // Handle Y tiling
        if (yTiles != 0) {
            if (mirror != 0) {
                // Mirror tiling: create ping-pong pattern
                float fracPart = fmodf(fabs(finalSrcY), 1.0f);
                int isOdd = (int)floor(fabs(finalSrcY)) & 1;
                finalSrcY = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                // Regular repeat tiling
                finalSrcY = finalSrcY - floor(finalSrcY);
            }
        }
        else if (finalSrcY < 0.0f || finalSrcY >= 1.0f) {
            outsideBounds = true;
        }

        // Sample the source image with bilinear interpolation
        float4 pixel;

        if (outsideBounds) {
            // Return transparent pixel if outside bounds and tiling is disabled
            pixel.x = 0.0f;
            pixel.y = 0.0f;
            pixel.z = 0.0f;
            pixel.w = 0.0f;
        }
        else {
            // Clamp to valid range
            finalSrcX = fmaxf_custom(0.0f, fminf_custom(0.9999f, finalSrcX));
            finalSrcY = fmaxf_custom(0.0f, fminf_custom(0.9999f, finalSrcY));

            // Convert to pixel coordinates
            float x = finalSrcX * width;
            float y = finalSrcY * height;

            // Get the four surrounding pixel coordinates
            int x0 = (int)floor(x);
            int y0 = (int)floor(y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            // Ensure coordinates are within bounds
            x0 = max_int(0, min_int(width - 1, x0));
            y0 = max_int(0, min_int(height - 1, y0));
            x1 = max_int(0, min_int(width - 1, x1));
            y1 = max_int(0, min_int(height - 1, y1));

            // Calculate interpolation factors
            float fx = x - x0;
            float fy = y - y0;

            // Sample the four surrounding pixels
            float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
            float4 p10 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
            float4 p01 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
            float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

            // Bilinear interpolation
            float oneMinusFx = 1.0f - fx;
            float oneMinusFy = 1.0f - fy;

            float w00 = oneMinusFx * oneMinusFy;
            float w10 = fx * oneMinusFy;
            float w01 = oneMinusFx * fy;
            float w11 = fx * fy;

            pixel.x = p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11;
            pixel.y = p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11;
            pixel.z = p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11;
            pixel.w = p00.w * w00 + p10.w * w10 + p01.w * w01 + p11.w * w11;
        }

        WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
// Add the PinchBulge CUDA implementation
void PinchBulge_CUDA(
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

    PinchBulgeKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst, srcPitch, dstPitch, is16f, width, height,
        centerX, centerY, strength, radius, xTiles, yTiles, mirror);

    cudaDeviceSynchronize();
}
#endif // __NVCC__
#endif
