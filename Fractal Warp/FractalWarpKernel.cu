#ifndef FractalWarp
#define FractalWarp

#include "PrGPU/KernelSupport/KernelCore.h"  
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
#define floorf floor
#define sinf sin
#define sqrtf sqrt
#endif

__device__ float mixf(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

__device__ float fractf(float x) {
    return x - floorf(x);
}

__device__ float random_hash(float x, float y) {
    return fractf(sinf(x * 12.9898f + y * 78.233f) * 43758.5453123f);
}

__device__ float noise(float x, float y) {
    float i = floorf(x);
    float j = floorf(y);
    float f = x - i;
    float g = y - j;

    float a = random_hash(i, j);
    float b = random_hash(i + 1.0f, j);
    float c = random_hash(i, j + 1.0f);
    float d = random_hash(i + 1.0f, j + 1.0f);

    float u = f * f * (3.0f - 2.0f * f);
    float v = g * g * (3.0f - 2.0f * g);

    return mixf(mixf(a, b, u), mixf(c, d, u), v);
}

__device__ float fbm(float x, float y, float px, float py, int octaveCount, float intensity) {
    float value = 0.0f;
    float amplitude = 0.5f;

    for (int i = 0; i < octaveCount; i++) {
        value += amplitude * noise(x, y);

        x = x * 2.0f + px;
        y = y * 2.0f + py;

        amplitude *= intensity;
    }

    return value;
}

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

GF_KERNEL_FUNCTION(FractalWarpKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inPositionX))
    ((float)(inPositionY))
    ((float)(inParallaxX))
    ((float)(inParallaxY))
    ((float)(inMagnitude))
    ((float)(inDetail))
    ((float)(inLacunarity))
    ((int)(inScreenSpace))
    ((int)(inOctaves))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float adjusted_x = (float)inXY.x;
        float adjusted_y = (float)inXY.y;

        const float width = (float)inWidth;
        const float height = (float)inHeight;
        const float aspectRatio = width / height;

        float st_x, st_y;
        if (inScreenSpace) {
            st_x = adjusted_x / width;
            st_y = 1.0f - (adjusted_y / height);
            st_x *= aspectRatio;
        }
        else {
            st_x = adjusted_x / width;
            st_y = 1.0f - (adjusted_y / height);
            st_x *= aspectRatio;
        }

        float position_offset_x = inPositionX / 1000.0f * -1.0f;
        float position_offset_y = inPositionY / 1000.0f * 1.0f;

        st_x += position_offset_x;
        st_y += position_offset_y;

        float parallax_x = inParallaxX / 200.0f * -1.0f;
        float parallax_y = inParallaxY / 200.0f * 1.0f;

        const float dx = fbm(
            ((st_x - 0.5f) * 3.0f * inDetail) + 0.5f,
            ((st_y - 0.5f) * 3.0f * inDetail) + 0.5f,
            parallax_x, parallax_y,
            inOctaves, inLacunarity
        );

        const float dy = fbm(
            ((st_x + 25.3f - 0.5f) * 3.0f * inDetail) + 0.5f,
            ((st_y + 12.9f - 0.5f) * 3.0f * inDetail) + 0.5f,
            parallax_x, parallax_y,
            inOctaves, inLacunarity
        );

        float sample_x, sample_y;
        if (inScreenSpace) {
            sample_x = adjusted_x / width;
            sample_y = 1.0f - (adjusted_y / height);
        }
        else {
            sample_x = adjusted_x / width;
            sample_y = 1.0f - (adjusted_y / height);
        }

        sample_x += (dx - 0.5f) * inMagnitude;
        sample_y += (dy - 0.5f) * inMagnitude;

        sample_x *= width;
        sample_y = (1.0f - sample_y) * height;

        bool outsideBounds = false;

        float norm_x = sample_x / width;
        float norm_y = sample_y / height;

        if (inXTiles) {
            if (inMirror) {
                float fracPart = fmodf(fabs(norm_x), 1.0f);
                int isOdd = (int)floorf(fabs(norm_x)) & 1;
                norm_x = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                norm_x = norm_x - floorf(norm_x);
                if (norm_x < 0) norm_x += 1.0f;
            }
            sample_x = norm_x * width;
        }
        else {
            if (sample_x < 0.0f || sample_x >= width) {
                outsideBounds = true;
            }
        }

        if (inYTiles) {
            if (inMirror) {
                float fracPart = fmodf(fabs(norm_y), 1.0f);
                int isOdd = (int)floorf(fabs(norm_y)) & 1;
                norm_y = isOdd ? 1.0f - fracPart : fracPart;
            }
            else {
                norm_y = norm_y - floorf(norm_y);
                if (norm_y < 0) norm_y += 1.0f;
            }
            sample_y = norm_y * height;
        }
        else {
            if (sample_y < 0.0f || sample_y >= height) {
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
            sample_x = fmaxf_custom(0.0f, fminf_custom(width - 1.001f, sample_x));
            sample_y = fmaxf_custom(0.0f, fminf_custom(height - 1.001f, sample_y));

            int x0 = (int)floorf(sample_x);
            int y0 = (int)floorf(sample_y);
            int x1 = x0 + 1;
            int y1 = y0 + 1;

            x0 = max_int(0, min_int(inWidth - 1, x0));
            y0 = max_int(0, min_int(inHeight - 1, y0));
            x1 = max_int(0, min_int(inWidth - 1, x1));
            y1 = max_int(0, min_int(inHeight - 1, y1));

            float fx = sample_x - x0;
            float fy = sample_y - y0;

            float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
            float4 p10 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
            float4 p01 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
            float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

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
void FractalWarp_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float positionX,
    float positionY,
    float parallaxX,
    float parallaxY,
    float magnitude,
    float detail,
    float lacunarity,
    int screenSpace,
    int octaves,
    int x_tiles,
    int y_tiles,
    int mirror)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    FractalWarpKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst,
        srcPitch, dstPitch, is16f, width, height,
        positionX, positionY, parallaxX, parallaxY,
        magnitude, detail, lacunarity, screenSpace, octaves,
        x_tiles, y_tiles, mirror);

    cudaDeviceSynchronize();
}
#endif  
#endif
