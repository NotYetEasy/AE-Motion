#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#define NOMINMAX
#include "ReorientSphere.h"
#include <iostream>
#include <algorithm>

#define _USE_MATH_DEFINES
#include <math.h>

inline PF_Err CL2Err(cl_int cl_result) {
    if (cl_result == CL_SUCCESS) {
        return PF_Err_NONE;
    }
    else {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))

extern void ReorientSphere_CUDA(
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
    float downsample_factor_y);


static void multiplyMatrices(const Matrix4x4& a, const Matrix4x4& b, Matrix4x4& result) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            result.m[i * 4 + j] = 0;
            for (int k = 0; k < 4; k++) {
                result.m[i * 4 + j] += a.m[i * 4 + k] * b.m[k * 4 + j];
            }
        }
    }
}

static Matrix4x4 rotationAxisAngle(float x, float y, float z, float angle) {
    Matrix4x4 m;

    float length = sqrt(x * x + y * y + z * z);
    if (length > 0.0f) {
        x /= length;
        y /= length;
        z /= length;
    }

    float s = sin(angle);
    float c = cos(angle);
    float ic = 1.0f - c;

    m.m[0] = x * x * ic + c;      m.m[4] = y * x * ic - s * z;    m.m[8] = z * x * ic + s * y;    m.m[12] = 0.0f;
    m.m[1] = x * y * ic + s * z;  m.m[5] = y * y * ic + c;        m.m[9] = z * y * ic - s * x;    m.m[13] = 0.0f;
    m.m[2] = x * z * ic - s * y;  m.m[6] = y * z * ic + s * x;    m.m[10] = z * z * ic + c;       m.m[14] = 0.0f;
    m.m[3] = 0.0f;                m.m[7] = 0.0f;                  m.m[11] = 0.0f;                 m.m[15] = 1.0f;

    return m;
}

static Matrix4x4 orientationToMatrix(const PF_ParamDef* param, float downsample_factor_x = 1.0f, float downsample_factor_y = 1.0f) {
    Matrix4x4 m;
    for (int i = 0; i < 16; i++) {
        m.m[i] = (i % 5 == 0) ? 1.0f : 0.0f;
    }

    float rx = param->u.point3d_d.x_value * 0.0174533f * downsample_factor_x;
    float ry = param->u.point3d_d.y_value * 0.0174533f * downsample_factor_y;
    float rz = -param->u.point3d_d.z_value * 0.0174533f * downsample_factor_x;

    Matrix4x4 matX = rotationAxisAngle(1.0f, 0.0f, 0.0f, rx);
    Matrix4x4 matY = rotationAxisAngle(0.0f, 1.0f, 0.0f, ry);
    Matrix4x4 matZ = rotationAxisAngle(0.0f, 0.0f, 1.0f, rz);

    Matrix4x4 temp;
    multiplyMatrices(matZ, matY, temp);
    multiplyMatrices(temp, matX, m);

    return m;
}

static PF_Err
About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
        "%s v%d.%d\r%s",
        STR_NAME,
        MAJOR_VERSION,
        MINOR_VERSION,
        STR_DESCRIPTION);

    return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
    PF_InData* in_dataP,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    out_data->my_version = PF_VERSION(MAJOR_VERSION,
        MINOR_VERSION,
        BUG_VERSION,
        STAGE_VERSION,
        BUILD_VERSION);

    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;

    out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE |
        PF_OutFlag2_SUPPORTS_SMART_RENDER |
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

    if (in_dataP->appl_id == 'PrMr') {

        AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
            AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_dataP,
                kPFPixelFormatSuite,
                kPFPixelFormatSuiteVersion1,
                out_data);

        (*pixelFormatSuite->ClearSupportedPixelFormats)(in_dataP->effect_ref);
        (*pixelFormatSuite->AddSupportedPixelFormat)(
            in_dataP->effect_ref,
            PrPixelFormat_VUYA_4444_32f);
    }
    else {
        out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 | PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;
    }

    return err;
}

static PF_Err
ParamsSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);

    def.flags = PF_ParamFlag_SUPERVISE;
    PF_ADD_POINT_3D(STR_ORIENTATION_PARAM,
        0, 0, 0,
        ORIENTATION_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_POINT_3D(STR_ROTATION_PARAM,
        0, 0, 0,
        ROTATION_DISK_ID);

    out_data->num_params = REORIENT_NUM_PARAMS;

    return err;
}

#if HAS_METAL
PF_Err NSError2PFErr(NSError* inError)
{
    if (inError)
    {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return PF_Err_NONE;
}
#endif 

struct OpenCLGPUData
{
    cl_kernel reorient_kernel;
};

#if HAS_HLSL
inline PF_Err DXErr(bool inSuccess) {
    if (inSuccess) { return PF_Err_NONE; }
    else { return PF_Err_INTERNAL_STRUCT_DAMAGED; }
}
#define DX_ERR(FUNC) ERR(DXErr(FUNC))
#include "DirectXUtils.h"
struct DirectXGPUData
{
    DXContextPtr mContext;
    ShaderObjectPtr mReorientShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState> reorient_pipeline;
};
#endif

static PF_Err
GPUDeviceSetup(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_GPUDeviceSetupExtra* extraP)
{
    PF_Err err = PF_Err_NONE;

    PF_GPUDeviceInfo device_info;
    AEFX_CLR_STRUCT(device_info);

    AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
        kPFHandleSuite,
        kPFHandleSuiteVersion1,
        out_dataP);

    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite =
        AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP,
            kPFGPUDeviceSuite,
            kPFGPUDeviceSuiteVersion1,
            out_dataP);

    gpuDeviceSuite->GetDeviceInfo(in_dataP->effect_ref,
        extraP->input->device_index,
        &device_info);

    if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
    else if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {

        PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(OpenCLGPUData));
        OpenCLGPUData* cl_gpu_data = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_int result = CL_SUCCESS;

        char const* k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";

        size_t sizes[] = { strlen(k16fString), strlen(kReorientSphereKernel_OpenCLString) };
        char const* strings[] = { k16fString, kReorientSphereKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->reorient_kernel = clCreateKernel(program, "ReorientSphereKernel", &result);
            CL_ERR(result);
        }

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#if HAS_HLSL
    else if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
    {
        PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(DirectXGPUData));
        DirectXGPUData* dx_gpu_data = reinterpret_cast<DirectXGPUData*>(*gpu_dataH);
        memset(dx_gpu_data, 0, sizeof(DirectXGPUData));

        dx_gpu_data->mContext = std::make_shared<DXContext>();
        dx_gpu_data->mReorientShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"ReorientSphereKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mReorientShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kReorientSphere_Kernel_MetalString encoding : NSUTF8StringEncoding];
        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

        NSError* error = nil;
        id<MTLLibrary> library = [[device newLibraryWithSource : source options : nil error : &error]autorelease];

        if (!err && !library) {
            err = NSError2PFErr(error);
        }

        NSString* getError = error.localizedDescription;

        PF_Handle metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
        MetalGPUData* metal_data = reinterpret_cast<MetalGPUData*>(*metal_handle);

        if (err == PF_Err_NONE)
        {
            id<MTLFunction> reorient_function = nil;
            NSString* reorient_name = [NSString stringWithCString : "ReorientSphereKernel" encoding : NSUTF8StringEncoding];

            reorient_function = [[library newFunctionWithName : reorient_name]autorelease];

            if (!reorient_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->reorient_pipeline = [device newComputePipelineStateWithFunction : reorient_function error : &error];
                err = NSError2PFErr(error);
            }

            if (!err) {
                extraP->output->gpu_data = metal_handle;
                out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
            }
        }
    }
#endif
    return err;
}

static PF_Err
GPUDeviceSetdown(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_GPUDeviceSetdownExtra* extraP)
{
    PF_Err err = PF_Err_NONE;

    if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        (void)clReleaseKernel(cl_gpu_dataP->reorient_kernel);

        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
            kPFHandleSuite,
            kPFHandleSuiteVersion1,
            out_dataP);

        handle_suite->host_dispose_handle(gpu_dataH);
    }
#if HAS_HLSL
    else if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        DirectXGPUData* dx_gpu_dataP = reinterpret_cast<DirectXGPUData*>(*gpu_dataH);

        dx_gpu_dataP->mContext.reset();
        dx_gpu_dataP->mReorientShader.reset();

        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
            kPFHandleSuite,
            kPFHandleSuiteVersion1,
            out_dataP);

        handle_suite->host_dispose_handle(gpu_dataH);
    }
#endif

    return err;
}

template<typename PixelT>
static PF_Err
ReorientSphereFunc(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelT* inP,
    PixelT* outP)
{
    PF_Err err = PF_Err_NONE;
    ReorientInfo* riP = reinterpret_cast<ReorientInfo*>(refcon);

    if (!riP) return PF_Err_BAD_CALLBACK_PARAM;

    float u = (float)xL / riP->width;
    float v = (float)yL / riP->height;

    if (u < 0.0f || u > 1.0f || v < 0.0f || v > 1.0f) {
        outP->alpha = 0;
        outP->red = outP->green = outP->blue = 0;
        return err;
    }

    float lon = u * 2.0f * M_PI;
    float lat = asin((v - 0.5f) * 2.0f);

    float xyz[3];
    xyz[0] = cos(lon) * cos(lat);
    xyz[1] = sin(lon) * cos(lat);
    xyz[2] = sin(lat);

    float origX = xyz[0], origY = xyz[1], origZ = xyz[2];
    xyz[0] = riP->orientation[0] * origX + riP->orientation[4] * origY + riP->orientation[8] * origZ + riP->orientation[12];
    xyz[1] = riP->orientation[1] * origX + riP->orientation[5] * origY + riP->orientation[9] * origZ + riP->orientation[13];
    xyz[2] = riP->orientation[2] * origX + riP->orientation[6] * origY + riP->orientation[10] * origZ + riP->orientation[14];

    float rx = riP->rotation[0] * 0.0174533f * riP->downsample_factor_x;
    float ry = riP->rotation[1] * 0.0174533f * riP->downsample_factor_y;
    float rz = riP->rotation[2] * 0.0174533f * riP->downsample_factor_x;        

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
        lon += 2.0f * M_PI;
    }

    float new_u = lon / (2.0f * M_PI);
    float new_v = (sin(lat) / 2.0f) + 0.5f;

    int x0 = (int)(new_u * (riP->width - 1));
    int y0 = (int)(new_v * (riP->height - 1));
    int x1 = std::min(x0 + 1, riP->width - 1);
    int y1 = std::min(y0 + 1, riP->height - 1);
    float fx = new_u * (riP->width - 1) - x0;
    float fy = new_v * (riP->height - 1) - y0;

    PixelT* p00 = NULL;
    PixelT* p10 = NULL;
    PixelT* p01 = NULL;
    PixelT* p11 = NULL;

    if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
        PF_Pixel8* row0 = (PF_Pixel8*)((char*)riP->input_worldP->data + y0 * riP->input_worldP->rowbytes);
        PF_Pixel8* row1 = (PF_Pixel8*)((char*)riP->input_worldP->data + y1 * riP->input_worldP->rowbytes);
        p00 = &row0[x0];
        p10 = &row0[x1];
        p01 = &row1[x0];
        p11 = &row1[x1];
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        PF_Pixel16* row0 = (PF_Pixel16*)((char*)riP->input_worldP->data + y0 * riP->input_worldP->rowbytes);
        PF_Pixel16* row1 = (PF_Pixel16*)((char*)riP->input_worldP->data + y1 * riP->input_worldP->rowbytes);
        p00 = &row0[x0];
        p10 = &row0[x1];
        p01 = &row1[x0];
        p11 = &row1[x1];
    }
    else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        PF_PixelFloat* row0 = (PF_PixelFloat*)((char*)riP->input_worldP->data + y0 * riP->input_worldP->rowbytes);
        PF_PixelFloat* row1 = (PF_PixelFloat*)((char*)riP->input_worldP->data + y1 * riP->input_worldP->rowbytes);
        p00 = &row0[x0];
        p10 = &row0[x1];
        p01 = &row1[x0];
        p11 = &row1[x1];
    }

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
        outP->alpha = (A_u_char)(w00 * p00->alpha + w10 * p10->alpha + w01 * p01->alpha + w11 * p11->alpha + 0.5f);
        outP->red = (A_u_char)(w00 * p00->red + w10 * p10->red + w01 * p01->red + w11 * p11->red + 0.5f);
        outP->green = (A_u_char)(w00 * p00->green + w10 * p10->green + w01 * p01->green + w11 * p11->green + 0.5f);
        outP->blue = (A_u_char)(w00 * p00->blue + w10 * p10->blue + w01 * p01->blue + w11 * p11->blue + 0.5f);
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        outP->alpha = (A_u_short)(w00 * p00->alpha + w10 * p10->alpha + w01 * p01->alpha + w11 * p11->alpha + 0.5f);
        outP->red = (A_u_short)(w00 * p00->red + w10 * p10->red + w01 * p01->red + w11 * p11->red + 0.5f);
        outP->green = (A_u_short)(w00 * p00->green + w10 * p10->green + w01 * p01->green + w11 * p11->green + 0.5f);
        outP->blue = (A_u_short)(w00 * p00->blue + w10 * p10->blue + w01 * p01->blue + w11 * p11->blue + 0.5f);
    }
    else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        outP->alpha = w00 * p00->alpha + w10 * p10->alpha + w01 * p01->alpha + w11 * p11->alpha;
        outP->red = w00 * p00->red + w10 * p10->red + w01 * p01->red + w11 * p11->red;
        outP->green = w00 * p00->green + w10 * p10->green + w01 * p01->green + w11 * p11->green;
        outP->blue = w00 * p00->blue + w10 * p10->blue + w01 * p01->blue + w11 * p11->blue;
    }

    return err;
}

static PF_Err
Render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    ReorientInfo riP;
    AEFX_CLR_STRUCT(riP);

    riP.width = output->width;
    riP.height = output->height;

    Matrix4x4 orient_matrix = orientationToMatrix(params[REORIENT_ORIENTATION]);

    for (int i = 0; i < 16; i++) {
        riP.orientation[i] = orient_matrix.m[i];
    }

    riP.rotation[0] = -params[REORIENT_ROTATION]->u.point3d_d.x_value;
    riP.rotation[1] = -params[REORIENT_ROTATION]->u.point3d_d.y_value;
    riP.rotation[2] = params[REORIENT_ROTATION]->u.point3d_d.z_value;

    riP.input_worldP = &params[REORIENT_INPUT]->u.ld;

    PF_PixelFormat pixelFormat;
    PF_WorldSuite2* wsP = NULL;
    ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
    ERR(wsP->PF_GetPixelFormat(output, &pixelFormat));
    ERR(suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2));

    switch (pixelFormat) {
    case PF_PixelFormat_ARGB128:
        ERR(suites.IterateFloatSuite1()->iterate(
            in_data,
            0,
            output->height,
            &params[REORIENT_INPUT]->u.ld,
            NULL,
            (void*)&riP,
            (PF_IteratePixelFloatFunc)ReorientSphereFunc<PF_PixelFloat>,
            output));
        break;

    case PF_PixelFormat_ARGB64:
        ERR(suites.Iterate16Suite1()->iterate(
            in_data,
            0,
            output->height,
            &params[REORIENT_INPUT]->u.ld,
            NULL,
            (void*)&riP,
            ReorientSphereFunc<PF_Pixel16>,
            output));
        break;

    case PF_PixelFormat_ARGB32:
    default:
        ERR(suites.Iterate8Suite1()->iterate(
            in_data,
            0,
            output->height,
            &params[REORIENT_INPUT]->u.ld,
            NULL,
            (void*)&riP,
            ReorientSphereFunc<PF_Pixel8>,
            output));
        break;
    }

    return err;
}

static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        ReorientInfo* infoP = reinterpret_cast<ReorientInfo*>(pre_render_dataPV);
        free(infoP);
    }
}

static PF_Err
PreRender(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PreRenderExtra* extraP)
{
    PF_Err err = PF_Err_NONE;
    PF_CheckoutResult in_result;
    PF_RenderRequest req = extraP->input->output_request;

    extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

    ReorientInfo* infoP = reinterpret_cast<ReorientInfo*>(malloc(sizeof(ReorientInfo)));

    if (infoP) {
        AEFX_CLR_STRUCT(*infoP);

        infoP->downsample_factor_x = (float)in_dataP->downsample_x.den / (float)in_dataP->downsample_x.num;
        infoP->downsample_factor_y = (float)in_dataP->downsample_y.den / (float)in_dataP->downsample_y.num;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            REORIENT_INPUT,
            REORIENT_INPUT,
            &req,
            in_dataP->current_time,
            in_dataP->time_step,
            in_dataP->time_scale,
            &in_result));

        infoP->width = in_result.result_rect.right - in_result.result_rect.left;
        infoP->height = in_result.result_rect.bottom - in_result.result_rect.top;

        PF_ParamDef orientation_param;
        AEFX_CLR_STRUCT(orientation_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, REORIENT_ORIENTATION, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &orientation_param));

        Matrix4x4 orient_matrix = orientationToMatrix(&orientation_param, infoP->downsample_factor_x, infoP->downsample_factor_y);
        for (int i = 0; i < 16; i++) {
            infoP->orientation[i] = orient_matrix.m[i];
        }

        PF_ParamDef rotation_param;
        AEFX_CLR_STRUCT(rotation_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, REORIENT_ROTATION, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &rotation_param));

        infoP->rotation[0] = -rotation_param.u.point3d_d.x_value;
        infoP->rotation[1] = -rotation_param.u.point3d_d.y_value;
        infoP->rotation[2] = rotation_param.u.point3d_d.z_value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        UnionLRect(&in_result.result_rect, &extraP->output->result_rect);
        UnionLRect(&in_result.max_result_rect, &extraP->output->max_result_rect);

        PF_CHECKIN_PARAM(in_dataP, &orientation_param);
        PF_CHECKIN_PARAM(in_dataP, &rotation_param);
    }
    else {
        err = PF_Err_OUT_OF_MEMORY;
    }
    return err;
}


static PF_Err
SmartRenderCPU(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    ReorientInfo* infoP)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    infoP->input_worldP = input_worldP;

    if (!err) {
        switch (pixel_format) {
        case PF_PixelFormat_ARGB128: {
            AEFX_SuiteScoper<PF_iterateFloatSuite1> iterateFloatSuite =
                AEFX_SuiteScoper<PF_iterateFloatSuite1>(in_data,
                    kPFIterateFloatSuite,
                    kPFIterateFloatSuiteVersion1,
                    out_data);
            iterateFloatSuite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)infoP,
                ReorientSphereFunc<PF_PixelFloat>,
                output_worldP);
            break;
        }

        case PF_PixelFormat_ARGB64: {
            AEFX_SuiteScoper<PF_iterate16Suite1> iterate16Suite =
                AEFX_SuiteScoper<PF_iterate16Suite1>(in_data,
                    kPFIterate16Suite,
                    kPFIterate16SuiteVersion1,
                    out_data);
            iterate16Suite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)infoP,
                ReorientSphereFunc<PF_Pixel16>,
                output_worldP);
            break;
        }

        case PF_PixelFormat_ARGB32: {
            AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
                AEFX_SuiteScoper<PF_Iterate8Suite1>(in_data,
                    kPFIterate8Suite,
                    kPFIterate8SuiteVersion1,
                    out_data);

            iterate8Suite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)infoP,
                ReorientSphereFunc<PF_Pixel8>,
                output_worldP);
            break;
        }

        default:
            err = PF_Err_BAD_CALLBACK_PARAM;
            break;
        }
    }

    return err;
}

static size_t
RoundUp(
    size_t inValue,
    size_t inMultiple)
{
    return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}

typedef struct
{
    int mSrcPitch;
    int mDstPitch;
    int m16f;
    int mWidth;
    int mHeight;
    float orientation[16];
    float rotationX;
    float rotationY;
    float rotationZ;
    float downsample_factor_x;
    float downsample_factor_y;
} ReorientSphereParams;

size_t DivideRoundUp(
    size_t inValue,
    size_t inMultiple)
{
    return inValue ? (inValue + inMultiple - 1) / inMultiple : 0;
}

static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    ReorientInfo* infoP)
{
    PF_Err err = PF_Err_NONE;

    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite = AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP,
        kPFGPUDeviceSuite,
        kPFGPUDeviceSuiteVersion1,
        out_dataP);

    if (pixel_format != PF_PixelFormat_GPU_BGRA128) {
        err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
    }
    A_long bytes_per_pixel = 16;

    PF_GPUDeviceInfo device_info;
    ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref, extraP->input->device_index, &device_info));

    void* src_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

    void* dst_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

    ReorientSphereParams reorient_params;

    reorient_params.mWidth = input_worldP->width;
    reorient_params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    reorient_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    reorient_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    reorient_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    for (int i = 0; i < 16; i++) {
        reorient_params.orientation[i] = infoP->orientation[i];
    }

    reorient_params.rotationX = infoP->rotation[0];
    reorient_params.rotationY = infoP->rotation[1];
    reorient_params.rotationZ = infoP->rotation[2];
    reorient_params.downsample_factor_x = infoP->downsample_factor_x;
    reorient_params.downsample_factor_y = infoP->downsample_factor_y;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(int), &reorient_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(int), &reorient_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(int), &reorient_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(int), &reorient_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(int), &reorient_params.mHeight));

        for (int i = 0; i < 16; i++) {
            CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(float), &reorient_params.orientation[i]));
        }

        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(float), &reorient_params.rotationX));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(float), &reorient_params.rotationY));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(float), &reorient_params.rotationZ));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(float), &reorient_params.downsample_factor_x));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->reorient_kernel, param_index++, sizeof(float), &reorient_params.downsample_factor_y));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(reorient_params.mWidth, threadBlock[0]), RoundUp(reorient_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->reorient_kernel,
            2,
            0,
            grid,
            threadBlock,
            0,
            0,
            0));
    }
#if HAS_CUDA
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_CUDA) {

        ReorientSphere_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            reorient_params.mSrcPitch,
            reorient_params.mDstPitch,
            reorient_params.m16f,
            reorient_params.mWidth,
            reorient_params.mHeight,
            reorient_params.orientation,
            reorient_params.rotationX,
            reorient_params.rotationY,
            reorient_params.rotationZ,
            reorient_params.downsample_factor_x,
            reorient_params.downsample_factor_y);

        if (cudaPeekAtLastError() != cudaSuccess) {
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }
    }
#endif
#if HAS_HLSL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        DirectXGPUData* dx_gpu_data = reinterpret_cast<DirectXGPUData*>(*gpu_dataH);

        {
            DXShaderExecution shaderExecution(
                dx_gpu_data->mContext,
                dx_gpu_data->mReorientShader,
                3);

            DX_ERR(shaderExecution.SetParamBuffer(&reorient_params, sizeof(ReorientSphereParams)));
            DX_ERR(shaderExecution.SetUnorderedAccessView(
                (ID3D12Resource*)dst_mem,
                reorient_params.mHeight * dst_row_bytes));
            DX_ERR(shaderExecution.SetShaderResourceView(
                (ID3D12Resource*)src_mem,
                reorient_params.mHeight * src_row_bytes));
            DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(reorient_params.mWidth, 16), (UINT)DivideRoundUp(reorient_params.mHeight, 16)));
        }
    }
#endif
#if HAS_METAL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        Handle metal_handle = (Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> reorient_param_buffer = [[device newBufferWithBytes : &reorient_params
            length : sizeof(ReorientSphereParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->reorient_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(reorient_params.mWidth, threadsPerGroup.width), DivideRoundUp(reorient_params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->reorient_pipeline] ;
        [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
        [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
        [computeEncoder setBuffer : reorient_param_buffer offset : 0 atIndex : 2] ;
        [computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;
        [computeEncoder endEncoding] ;
        [commandBuffer commit] ;

        err = NSError2PFErr([commandBuffer error]);
    }
#endif 

    return err;
}

static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extraP,
    bool isGPU)
{
    PF_Err err = PF_Err_NONE,
        err2 = PF_Err_NONE;

    PF_EffectWorld* input_worldP = NULL,
        * output_worldP = NULL;

    ReorientInfo* infoP = reinterpret_cast<ReorientInfo*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, REORIENT_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
            kPFWorldSuite,
            kPFWorldSuiteVersion2,
            out_data);
        PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
        ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

        if (isGPU) {
            ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
        }
        else {
            ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
        }
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, REORIENT_INPUT));
    }
    else {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}

extern "C" DllExport
PF_Err PluginDataEntryFunction2(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB2 inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;

    result = PF_REGISTER_EFFECT_EXT2(
        inPtr,
        inPluginDataCallBackPtr,
        "DKT Reorient Sphere",
        "DKT Reorient Sphere",
        "DKT Effects",
        AE_RESERVED_INFO,
        "EffectMain",
        "https://www.adobe.com");

    return result;
}

PF_Err
EffectMain(
    PF_Cmd cmd,
    PF_InData* in_dataP,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err err = PF_Err_NONE;

    try {
        switch (cmd)
        {
        case PF_Cmd_ABOUT:
            err = About(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_GLOBAL_SETUP:
            err = GlobalSetup(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_GPU_DEVICE_SETUP:
            err = GPUDeviceSetup(in_dataP, out_data, (PF_GPUDeviceSetupExtra*)extra);
            break;
        case PF_Cmd_GPU_DEVICE_SETDOWN:
            err = GPUDeviceSetdown(in_dataP, out_data, (PF_GPUDeviceSetdownExtra*)extra);
            break;
        case PF_Cmd_RENDER:
            err = Render(in_dataP, out_data, params, output);
            break;
        case PF_Cmd_SMART_PRE_RENDER:
            err = PreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra);
            break;
        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, false);
            break;
        case PF_Cmd_SMART_RENDER_GPU:
            err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, true);
            break;
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    return err;
}