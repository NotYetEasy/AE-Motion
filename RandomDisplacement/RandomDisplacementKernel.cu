#ifndef RandomDisplacementKernel_cl_h
#define RandomDisplacementKernel_cl_h

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

static const int p_array[256] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
};

static const float grad3_array[12][3] = {
    {1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f},
    {1.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, -1.0f},
    {0.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 1.0f}, {0.0f, 1.0f, -1.0f}, {0.0f, -1.0f, -1.0f}
};

GF_DEVICE_FUNCTION int get_p(int idx) {
    return p_array[idx & 0xFF];
}

GF_DEVICE_FUNCTION int get_perm(int idx) {
    return p_array[idx & 0xFF];
}

GF_DEVICE_FUNCTION int get_permMod12(int idx) {
    return get_perm(idx) % 12;
}

GF_DEVICE_FUNCTION void get_grad3(int idx, out float3 grad) {
    idx = idx % 12;
    grad.x = grad3_array[idx][0];
    grad.y = grad3_array[idx][1];
    grad.z = grad3_array[idx][2];
}
#else
__constant__ int p[256] = {
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
};

__constant__ float grad3[12][3] = {
    {1.0f, 1.0f, 0.0f}, {-1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f},
    {1.0f, 0.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, -1.0f}, {-1.0f, 0.0f, -1.0f},
    {0.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 1.0f}, {0.0f, 1.0f, -1.0f}, {0.0f, -1.0f, -1.0f}
};

GF_DEVICE_FUNCTION int get_p(int idx) {
    return p[idx & 0xFF];
}

GF_DEVICE_FUNCTION int get_perm(int idx) {
    return p[idx & 0xFF];
}

GF_DEVICE_FUNCTION int get_permMod12(int idx) {
    return get_perm(idx) % 12;
}

GF_DEVICE_FUNCTION void get_grad3(int idx, float* grad) {
    idx = idx % 12;
    grad[0] = grad3[idx][0];
    grad[1] = grad3[idx][1];
    grad[2] = grad3[idx][2];
}
#endif

__device__ int min_int(int a, int b) {
    return a < b ? a : b;
}

__device__ unsigned int min_uint(unsigned int a, unsigned int b) {
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

__device__ float fmodf_custom(float x, float y) {
    return x - y * floor(x / y);
}

#define F2_CONST 0.366025404f       
#define G2_CONST 0.211324865f       
#define F3_CONST 0.333333333f     
#define G3_CONST 0.166666667f     

GF_DEVICE_FUNCTION int fastfloor(float x) {
    int xi = (int)x;
    return x < xi ? xi - 1 : xi;
}

#if GF_DEVICE_TARGET_HLSL
GF_DEVICE_FUNCTION float dot_product(float3 g, float x, float y, float z) {
    return g.x * x + g.y * y + g.z * z;
}
#else
GF_DEVICE_FUNCTION float dot_product(float* g, float x, float y, float z) {
    return g[0] * x + g[1] * y + g[2] * z;
}
#endif

GF_DEVICE_FUNCTION float simplex_noise(float x, float y, float z = 0.0f, int dimensions = 3) {
    if (dimensions == 2) {
        float n0, n1, n2;

        float s = (x + y) * F2_CONST;
        int i = fastfloor(x + s);
        int j = fastfloor(y + s);

        float t = (i + j) * G2_CONST;
        float X0 = i - t;
        float Y0 = j - t;
        float x0 = x - X0;
        float y0 = y - Y0;

        int i1, j1;
        if (x0 > y0) {
            i1 = 1;
            j1 = 0;
        }
        else {
            i1 = 0;
            j1 = 1;
        }

        float x1 = x0 - i1 + G2_CONST;
        float y1 = y0 - j1 + G2_CONST;
        float x2 = x0 - 1.0f + 2.0f * G2_CONST;
        float y2 = y0 - 1.0f + 2.0f * G2_CONST;

        int ii = i & 255;
        int jj = j & 255;
        int gi0 = get_permMod12(ii + get_perm(jj));
        int gi1 = get_permMod12(ii + i1 + get_perm(jj + j1));
        int gi2 = get_permMod12(ii + 1 + get_perm(jj + 1));

        float t0 = 0.5f - x0 * x0 - y0 * y0;
        if (t0 < 0) {
            n0 = 0.0f;
        }
        else {
            t0 *= t0;
#if GF_DEVICE_TARGET_HLSL
            float3 g0;
            get_grad3(gi0, g0);
            n0 = t0 * t0 * dot_product(g0, x0, y0, 0);
#else
            float g0[3];
            get_grad3(gi0, g0);
            n0 = t0 * t0 * dot_product(g0, x0, y0, 0);
#endif
        }

        float t1 = 0.5f - x1 * x1 - y1 * y1;
        if (t1 < 0) {
            n1 = 0.0f;
        }
        else {
            t1 *= t1;
#if GF_DEVICE_TARGET_HLSL
            float3 g1;
            get_grad3(gi1, g1);
            n1 = t1 * t1 * dot_product(g1, x1, y1, 0);
#else
            float g1[3];
            get_grad3(gi1, g1);
            n1 = t1 * t1 * dot_product(g1, x1, y1, 0);
#endif
        }

        float t2 = 0.5f - x2 * x2 - y2 * y2;
        if (t2 < 0) {
            n2 = 0.0f;
        }
        else {
            t2 *= t2;
#if GF_DEVICE_TARGET_HLSL
            float3 g2;
            get_grad3(gi2, g2);
            n2 = t2 * t2 * dot_product(g2, x2, y2, 0);
#else
            float g2[3];
            get_grad3(gi2, g2);
            n2 = t2 * t2 * dot_product(g2, x2, y2, 0);
#endif
        }

        return 70.0f * (n0 + n1 + n2);
    }
    else {
        float n0, n1, n2, n3;

        float s = (x + y + z) * F3_CONST;
        int i = fastfloor(x + s);
        int j = fastfloor(y + s);
        int k = fastfloor(z + s);

        float t = (i + j + k) * G3_CONST;
        float X0 = i - t;
        float Y0 = j - t;
        float Z0 = k - t;
        float x0 = x - X0;
        float y0 = y - Y0;
        float z0 = z - Z0;

        int i1, j1, k1;
        int i2, j2, k2;
        if (x0 >= y0) {
            if (y0 >= z0) {
                i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
            }
            else if (x0 >= z0) {
                i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1;
            }
            else {
                i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1;
            }
        }
        else {
            if (y0 < z0) {
                i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1;
            }
            else if (x0 < z0) {
                i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1;
            }
            else {
                i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0;
            }
        }

        float x1 = x0 - i1 + G3_CONST;
        float y1 = y0 - j1 + G3_CONST;
        float z1 = z0 - k1 + G3_CONST;
        float x2 = x0 - i2 + 2.0f * G3_CONST;
        float y2 = y0 - j2 + 2.0f * G3_CONST;
        float z2 = z0 - k2 + 2.0f * G3_CONST;
        float x3 = x0 - 1.0f + 3.0f * G3_CONST;
        float y3 = y0 - 1.0f + 3.0f * G3_CONST;
        float z3 = z0 - 1.0f + 3.0f * G3_CONST;

        int ii = i & 255;
        int jj = j & 255;
        int kk = k & 255;
        int gi0 = get_permMod12(ii + get_perm(jj + get_perm(kk)));
        int gi1 = get_permMod12(ii + i1 + get_perm(jj + j1 + get_perm(kk + k1)));
        int gi2 = get_permMod12(ii + i2 + get_perm(jj + j2 + get_perm(kk + k2)));
        int gi3 = get_permMod12(ii + 1 + get_perm(jj + 1 + get_perm(kk + 1)));

        float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
        if (t0 < 0) n0 = 0.0f;
        else {
            t0 *= t0;
#if GF_DEVICE_TARGET_HLSL
            float3 g0;
            get_grad3(gi0, g0);
            n0 = t0 * t0 * dot_product(g0, x0, y0, z0);
#else
            float g0[3];
            get_grad3(gi0, g0);
            n0 = t0 * t0 * dot_product(g0, x0, y0, z0);
#endif
        }

        float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
        if (t1 < 0) n1 = 0.0f;
        else {
            t1 *= t1;
#if GF_DEVICE_TARGET_HLSL
            float3 g1;
            get_grad3(gi1, g1);
            n1 = t1 * t1 * dot_product(g1, x1, y1, z1);
#else
            float g1[3];
            get_grad3(gi1, g1);
            n1 = t1 * t1 * dot_product(g1, x1, y1, z1);
#endif
        }

        float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
        if (t2 < 0) n2 = 0.0f;
        else {
            t2 *= t2;
#if GF_DEVICE_TARGET_HLSL
            float3 g2;
            get_grad3(gi2, g2);
            n2 = t2 * t2 * dot_product(g2, x2, y2, z2);
#else
            float g2[3];
            get_grad3(gi2, g2);
            n2 = t2 * t2 * dot_product(g2, x2, y2, z2);
#endif
        }

        float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
        if (t3 < 0) n3 = 0.0f;
        else {
            t3 *= t3;
#if GF_DEVICE_TARGET_HLSL
            float3 g3;
            get_grad3(gi3, g3);
            n3 = t3 * t3 * dot_product(g3, x3, y3, z3);
#else
            float g3[3];
            get_grad3(gi3, g3);
            n3 = t3 * t3 * dot_product(g3, x3, y3, z3);
#endif
        }

        return 32.0f * (n0 + n1 + n2 + n3);
    }
}

GF_KERNEL_FUNCTION(RandomDisplacementKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(inMagnitude))
    ((float)(inEvolution))
    ((float)(inSeed))
    ((float)(inScatter))
    ((int)(inXTiles))
    ((int)(inYTiles))
    ((int)(inMirror))
    ((float)(inDownsampleX))
    ((float)(inDownsampleY)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float layerPosX = inWidth / 2.0f;
        float layerPosY = inHeight / 2.0f;


        float adjustedScatter = inScatter / inDownsampleX;

        float dx = simplex_noise(layerPosX * adjustedScatter / 50.0f + inSeed * 54623.245f, layerPosY * adjustedScatter / 500.0f, inEvolution + inSeed * 49235.319798f, 3);

        float dy = simplex_noise(layerPosX * adjustedScatter / 50.0f, layerPosY * adjustedScatter / 500.0f + inSeed * 8723.5647f, inEvolution + 7468.329f + inSeed * 19337.940385f, 3);

        dx *= -inMagnitude * inDownsampleX;
        dy *= inMagnitude * inDownsampleY;

        float srcX = (float)inXY.x - dx;
        float srcY = (float)inXY.y - dy;

        bool outsideBounds = false;

        if (inXTiles) {
            if (inMirror) {
                float intPart = floor(fabs(srcX / inWidth));
                float fracPart = fabs(srcX / inWidth) - intPart;
                int isOdd = (int)intPart & 1;
                srcX = isOdd ? (1.0f - fracPart) * inWidth : fracPart * inWidth;
            }
            else {
                srcX = fmodf_custom(fmodf_custom(srcX, (float)inWidth) + inWidth, (float)inWidth);
            }
        }
        else {
            if (srcX < 0 || srcX >= inWidth) {
                outsideBounds = true;
            }
        }

        if (inYTiles) {
            if (inMirror) {
                float intPart = floor(fabs(srcY / inHeight));
                float fracPart = fabs(srcY / inHeight) - intPart;
                int isOdd = (int)intPart & 1;
                srcY = isOdd ? (1.0f - fracPart) * inHeight : fracPart * inHeight;
            }
            else {
                srcY = fmodf_custom(fmodf_custom(srcY, (float)inHeight) + inHeight, (float)inHeight);
            }
        }
        else {
            if (srcY < 0 || srcY >= inHeight) {
                outsideBounds = true;
            }
        }

        if (outsideBounds) {
            float4 transparent = { 0.0f, 0.0f, 0.0f, 0.0f };
            WriteFloat4(transparent, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
            return;
        }

        srcX = fmaxf_custom(0.0f, fminf_custom((float)inWidth - 1.001f, srcX));
        srcY = fmaxf_custom(0.0f, fminf_custom((float)inHeight - 1.001f, srcY));

        int x0 = (int)srcX;
        int y0 = (int)srcY;
        int x1 = min_int(x0 + 1, inWidth - 1);
        int y1 = min_int(y0 + 1, inHeight - 1);

        float fx = srcX - x0;
        float fy = srcY - y0;

        float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
        float4 p01 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
        float4 p10 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
        float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

        float4 pixel;
        float oneMinusFx = 1.0f - fx;
        float oneMinusFy = 1.0f - fy;

        pixel.x = oneMinusFx * oneMinusFy * p00.x +
            fx * oneMinusFy * p01.x +
            oneMinusFx * fy * p10.x +
            fx * fy * p11.x;

        pixel.y = oneMinusFx * oneMinusFy * p00.y +
            fx * oneMinusFy * p01.y +
            oneMinusFx * fy * p10.y +
            fx * fy * p11.y;

        pixel.z = oneMinusFx * oneMinusFy * p00.z +
            fx * oneMinusFy * p01.z +
            oneMinusFx * fy * p10.z +
            fx * fy * p11.z;

        pixel.w = oneMinusFx * oneMinusFy * p00.w +
            fx * oneMinusFy * p01.w +
            oneMinusFx * fy * p10.w +
            fx * fy * p11.w;

        WriteFloat4(pixel, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
void RandomDisplacement_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float magnitude,
    float evolution,
    float seed,
    float scatter,
    int x_tiles,
    int y_tiles,
    int mirror,
    float downsample_x,
    float downsample_y)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    RandomDisplacementKernel << < gridDim, blockDim, 0 >> > ((float4 const*)src, (float4*)dst,
        srcPitch, dstPitch, is16f, width, height,
        magnitude, evolution, seed, scatter, x_tiles, y_tiles, mirror,
        downsample_x, downsample_y);

    cudaDeviceSynchronize();
}
#endif  
#endif
