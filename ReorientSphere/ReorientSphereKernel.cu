#ifndef ReorientSphere
#define ReorientSphere

#include "PrGPU/KernelSupport/KernelCore.h"  
#include "PrGPU/KernelSupport/KernelMemory.h"

#if GF_DEVICE_TARGET_DEVICE
#if GF_DEVICE_TARGET_HLSL
#define fmax max
#define fmin min
#endif

GF_KERNEL_FUNCTION(ReorientSphereKernel,
    ((GF_PTR_READ_ONLY(float4))(inSrc))
    ((GF_PTR(float4))(outDst)),
    ((int)(inSrcPitch))
    ((int)(inDstPitch))
    ((int)(in16f))
    ((unsigned int)(inWidth))
    ((unsigned int)(inHeight))
    ((float)(orientation0))
    ((float)(orientation1))
    ((float)(orientation2))
    ((float)(orientation3))
    ((float)(orientation4))
    ((float)(orientation5))
    ((float)(orientation6))
    ((float)(orientation7))
    ((float)(orientation8))
    ((float)(orientation9))
    ((float)(orientation10))
    ((float)(orientation11))
    ((float)(orientation12))
    ((float)(orientation13))
    ((float)(orientation14))
    ((float)(orientation15))
    ((float)(rotationX))
    ((float)(rotationY))
    ((float)(rotationZ))
    ((float)(downsample_factor_x))
    ((float)(downsample_factor_y)),
    ((uint2)(inXY)(KERNEL_XY)))
{
    if (inXY.x < inWidth && inXY.y < inHeight)
    {
        float u = (float)inXY.x / inWidth;
        float v = (float)inXY.y / inHeight;

        float lon = u * 2.0f * 3.14159265358979323846f;
        float lat = asin((v - 0.5f) * 2.0f);

        float xyz[3];
        xyz[0] = cos(lon) * cos(lat);
        xyz[1] = sin(lon) * cos(lat);
        xyz[2] = sin(lat);

        // Apply orientation with downsampling factors
        float orientation[16] = {
            orientation0, orientation1, orientation2, orientation3,
            orientation4, orientation5, orientation6, orientation7,
            orientation8, orientation9, orientation10, orientation11,
            orientation12, orientation13, orientation14, orientation15
        };

        float origX = xyz[0], origY = xyz[1], origZ = xyz[2];
        xyz[0] = orientation[0] * origX + orientation[4] * origY + orientation[8] * origZ + orientation[12];
        xyz[1] = orientation[1] * origX + orientation[5] * origY + orientation[9] * origZ + orientation[13];
        xyz[2] = orientation[2] * origX + orientation[6] * origY + orientation[10] * origZ + orientation[14];

        // Apply rotation with downsampling factors
        float rx = rotationX * 0.0174533f * downsample_factor_x;
        float ry = rotationY * 0.0174533f * downsample_factor_y;
        float rz = rotationZ * 0.0174533f * downsample_factor_x;

        {
            float s = sin(rz);
            float c = cos(rz);
            float x = xyz[0], y = xyz[1];
            xyz[0] = x * c - y * s;
            xyz[1] = x * s + y * c;
        }

        {
            float s = sin(rx);
            float c = cos(rx);
            float y = xyz[1], z = xyz[2];
            xyz[1] = y * c - z * s;
            xyz[2] = y * s + z * c;
        }

        {
            float s = sin(ry);
            float c = cos(ry);
            float x = xyz[0], z = xyz[2];
            xyz[0] = x * c + z * s;
            xyz[2] = -x * s + z * c;
        }

        lat = asin(xyz[2]);
        lon = atan2(xyz[1], xyz[0]);
        if (lon < 0.0f) {
            lon += 2.0f * 3.14159265358979323846f;
        }

        float new_u = lon / (2.0f * 3.14159265358979323846f);
        float new_v = (sin(lat) / 2.0f) + 0.5f;

        new_u = fmax(fmin(1.0f, new_u), 0.0f);
        new_v = fmax(fmin(1.0f, new_v), 0.0f);

        int x0 = (int)(new_u * (inWidth - 1));
        int y0 = (int)(new_v * (inHeight - 1));
        int x1 = min(x0 + 1, inWidth - 1);
        int y1 = min(y0 + 1, inHeight - 1);
        float fx = new_u * (inWidth - 1) - x0;
        float fy = new_v * (inHeight - 1) - y0;

        float4 p00 = ReadFloat4(inSrc, y0 * inSrcPitch + x0, !!in16f);
        float4 p10 = ReadFloat4(inSrc, y0 * inSrcPitch + x1, !!in16f);
        float4 p01 = ReadFloat4(inSrc, y1 * inSrcPitch + x0, !!in16f);
        float4 p11 = ReadFloat4(inSrc, y1 * inSrcPitch + x1, !!in16f);

        float w00 = (1.0f - fx) * (1.0f - fy);
        float w10 = fx * (1.0f - fy);
        float w01 = (1.0f - fx) * fy;
        float w11 = fx * fy;

        float4 result;
        result.x = w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x;
        result.y = w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y;
        result.z = w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z;
        result.w = w00 * p00.w + w10 * p10.w + w01 * p01.w + w11 * p11.w;

        WriteFloat4(result, outDst, inXY.y * inDstPitch + inXY.x, !!in16f);
    }
}
#endif

#if __NVCC__
void ReorientSphere_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float orientation[16],
    float rotationX,
    float rotationY,
    float rotationZ,
    float downsample_factor_x,
    float downsample_factor_y)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y, 1);

    ReorientSphereKernel << < gridDim, blockDim, 0 >> > (
        (float4 const*)src,
        (float4*)dst,
        srcPitch,
        dstPitch,
        is16f,
        width,
        height,
        orientation[0],
        orientation[1],
        orientation[2],
        orientation[3],
        orientation[4],
        orientation[5],
        orientation[6],
        orientation[7],
        orientation[8],
        orientation[9],
        orientation[10],
        orientation[11],
        orientation[12],
        orientation[13],
        orientation[14],
        orientation[15],
        rotationX,
        rotationY,
        rotationZ,
        downsample_factor_x,
        downsample_factor_y);

    cudaDeviceSynchronize();
}
#endif
#endif