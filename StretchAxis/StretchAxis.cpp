#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "StretchAxis.h"
#include <iostream>

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

extern void StretchAxis_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float scale,
    float angle,
    int contentOnly,
    int xTiles,
    int yTiles,
    int mirror);

static PF_Err
About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_SPRINTF(out_data->return_msg,
        "%s, v%d.%d\r%s",
        NAME,
        MAJOR_VERSION,
        MINOR_VERSION,
        DESCRIPTION);

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

    PF_ADD_FLOAT_SLIDERX("Scale",
        0.01,
        50.00,
        0.01,
        5.00,
        1.00,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        SCALE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_FLOAT_SLIDERX("Angle",
        0,
        3600,
        0,
        180,
        0,
        PF_Precision_TENTHS,
        0,
        0,
        ANGLE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_CHECKBOX("Mask to Layer",
        "",
        FALSE,
        0,
        CONTENT_ONLY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_START;
    PF_STRCPY(def.name, "Tiles");
    def.flags = PF_ParamFlag_START_COLLAPSED;      
    PF_ADD_PARAM(in_data, -1, &def);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("X Tiles",
        "",
        FALSE,
        0,
        X_TILES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Y Tiles",
        "",
        FALSE,
        0,
        Y_TILES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Mirror",
        "",
        FALSE,
        0,
        MIRROR_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    PF_ADD_PARAM(in_data, -1, &def);

    out_data->num_params = STRETCH_NUM_PARAMS;

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
    cl_kernel stretch_kernel;
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
    ShaderObjectPtr mStretchShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>stretch_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kStretchAxisKernel_OpenCLString) };
        char const* strings[] = { k16fString, kStretchAxisKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->stretch_kernel = clCreateKernel(program, "StretchAxisKernel", &result);
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
        dx_gpu_data->mStretchShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"StretchAxisKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mStretchShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kStretchAxis_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> stretch_function = nil;
            NSString* stretch_name = [NSString stringWithCString : "StretchAxisKernel" encoding : NSUTF8StringEncoding];

            stretch_function = [[library newFunctionWithName : stretch_name]autorelease];

            if (!stretch_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->stretch_pipeline = [device newComputePipelineStateWithFunction : stretch_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->stretch_kernel);

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
        dx_gpu_dataP->mStretchShader.reset();

        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
            kPFHandleSuite,
            kPFHandleSuiteVersion1,
            out_dataP);

        handle_suite->host_dispose_handle(gpu_dataH);
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*gpu_dataH);

        [metal_dataP->stretch_pipeline release] ;

        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
            kPFHandleSuite,
            kPFHandleSuiteVersion1,
            out_dataP);

        handle_suite->host_dispose_handle(gpu_dataH);
    }
#endif
    return err;
}

template <typename PixelType>
static void SampleBilinear(
    PF_EffectWorld* input,
    float u,
    float v,
    PixelType* outPixel,
    bool x_tiles,
    bool y_tiles,
    bool mirror)
{
    int width = input->width;
    int height = input->height;

    bool outsideBounds = false;
    float originalU = u;
    float originalV = v;

    if (x_tiles) {
        if (mirror) {
            float intPart;
            float fracPart = modff(fabsf(u), &intPart);
            int isOdd = (int)intPart & 1;
            u = isOdd ? 1.0f - fracPart : fracPart;
        }
        else {
            u = u - floorf(u);
            if (u < 0.0f) u += 1.0f;
        }
    }
    else {
        if (u < 0.0f || u >= 1.0f) {
            outsideBounds = true;
        }
    }

    if (y_tiles) {
        if (mirror) {
            float intPart;
            float fracPart = modff(fabsf(v), &intPart);
            int isOdd = (int)intPart & 1;
            v = isOdd ? 1.0f - fracPart : fracPart;
        }
        else {
            v = v - floorf(v);
            if (v < 0.0f) v += 1.0f;
        }
    }
    else {
        if (v < 0.0f || v >= 1.0f) {
            outsideBounds = true;
        }
    }

    if (outsideBounds) {
        if (sizeof(PixelType) == sizeof(PF_Pixel)) {
            PF_Pixel8* out_8 = reinterpret_cast<PF_Pixel8*>(outPixel);
            out_8->alpha = 0;
            out_8->red = 0;
            out_8->green = 0;
            out_8->blue = 0;
        }
        else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
            PF_Pixel16* out_16 = reinterpret_cast<PF_Pixel16*>(outPixel);
            out_16->alpha = 0;
            out_16->red = 0;
            out_16->green = 0;
            out_16->blue = 0;
        }
        else {  
            PF_PixelFloat* out_f = reinterpret_cast<PF_PixelFloat*>(outPixel);
            out_f->alpha = 0.0f;
            out_f->red = 0.0f;
            out_f->green = 0.0f;
            out_f->blue = 0.0f;
        }
        return;
    }

    u = MAX(0.0f, MIN(0.9999f, u));
    v = MAX(0.0f, MIN(0.9999f, v));

    float x = u * width;
    float y = v * height;

    int x0 = (int)floorf(x);
    int y0 = (int)floorf(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = MAX(0, MIN(width - 1, x0));
    y0 = MAX(0, MIN(height - 1, y0));
    x1 = MAX(0, MIN(width - 1, x1));
    y1 = MAX(0, MIN(height - 1, y1));

    float fx = x - x0;
    float fy = y - y0;

    PixelType* p00;
    PixelType* p10;
    PixelType* p01;
    PixelType* p11;

    if (sizeof(PixelType) == sizeof(PF_Pixel)) {
        PF_Pixel8* base = reinterpret_cast<PF_Pixel8*>(input->data);
        p00 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_Pixel8)) + x0);
        p10 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_Pixel8)) + x1);
        p01 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_Pixel8)) + x0);
        p11 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_Pixel8)) + x1);
    }
    else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
        PF_Pixel16* base = reinterpret_cast<PF_Pixel16*>(input->data);
        p00 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_Pixel16)) + x0);
        p10 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_Pixel16)) + x1);
        p01 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_Pixel16)) + x0);
        p11 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_Pixel16)) + x1);
    }
    else {  
        PF_PixelFloat* base = reinterpret_cast<PF_PixelFloat*>(input->data);
        p00 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_PixelFloat)) + x0);
        p10 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_PixelFloat)) + x1);
        p01 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_PixelFloat)) + x0);
        p11 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_PixelFloat)) + x1);
    }

    if (sizeof(PixelType) == sizeof(PF_Pixel)) {
        PF_Pixel8* p00_8 = reinterpret_cast<PF_Pixel8*>(p00);
        PF_Pixel8* p10_8 = reinterpret_cast<PF_Pixel8*>(p10);
        PF_Pixel8* p01_8 = reinterpret_cast<PF_Pixel8*>(p01);
        PF_Pixel8* p11_8 = reinterpret_cast<PF_Pixel8*>(p11);
        PF_Pixel8* out_8 = reinterpret_cast<PF_Pixel8*>(outPixel);

        float oneMinusFx = 1.0f - fx;
        float oneMinusFy = 1.0f - fy;

        float w00 = oneMinusFx * oneMinusFy;
        float w10 = fx * oneMinusFy;
        float w01 = oneMinusFx * fy;
        float w11 = fx * fy;

        out_8->alpha = (A_u_char)(p00_8->alpha * w00 + p10_8->alpha * w10 + p01_8->alpha * w01 + p11_8->alpha * w11);
        out_8->red = (A_u_char)(p00_8->red * w00 + p10_8->red * w10 + p01_8->red * w01 + p11_8->red * w11);
        out_8->green = (A_u_char)(p00_8->green * w00 + p10_8->green * w10 + p01_8->green * w01 + p11_8->green * w11);
        out_8->blue = (A_u_char)(p00_8->blue * w00 + p10_8->blue * w10 + p01_8->blue * w01 + p11_8->blue * w11);
    }
    else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
        PF_Pixel16* p00_16 = reinterpret_cast<PF_Pixel16*>(p00);
        PF_Pixel16* p10_16 = reinterpret_cast<PF_Pixel16*>(p10);
        PF_Pixel16* p01_16 = reinterpret_cast<PF_Pixel16*>(p01);
        PF_Pixel16* p11_16 = reinterpret_cast<PF_Pixel16*>(p11);
        PF_Pixel16* out_16 = reinterpret_cast<PF_Pixel16*>(outPixel);

        float oneMinusFx = 1.0f - fx;
        float oneMinusFy = 1.0f - fy;

        float w00 = oneMinusFx * oneMinusFy;
        float w10 = fx * oneMinusFy;
        float w01 = oneMinusFx * fy;
        float w11 = fx * fy;

        out_16->alpha = (A_u_short)(p00_16->alpha * w00 + p10_16->alpha * w10 + p01_16->alpha * w01 + p11_16->alpha * w11);
        out_16->red = (A_u_short)(p00_16->red * w00 + p10_16->red * w10 + p01_16->red * w01 + p11_16->red * w11);
        out_16->green = (A_u_short)(p00_16->green * w00 + p10_16->green * w10 + p01_16->green * w01 + p11_16->green * w11);
        out_16->blue = (A_u_short)(p00_16->blue * w00 + p10_16->blue * w10 + p01_16->blue * w01 + p11_16->blue * w11);
    }
    else {  
        PF_PixelFloat* p00_f = reinterpret_cast<PF_PixelFloat*>(p00);
        PF_PixelFloat* p10_f = reinterpret_cast<PF_PixelFloat*>(p10);
        PF_PixelFloat* p01_f = reinterpret_cast<PF_PixelFloat*>(p01);
        PF_PixelFloat* p11_f = reinterpret_cast<PF_PixelFloat*>(p11);
        PF_PixelFloat* out_f = reinterpret_cast<PF_PixelFloat*>(outPixel);

        float oneMinusFx = 1.0f - fx;
        float oneMinusFy = 1.0f - fy;

        float w00 = oneMinusFx * oneMinusFy;
        float w10 = fx * oneMinusFy;
        float w01 = oneMinusFx * fy;
        float w11 = fx * fy;

        out_f->alpha = p00_f->alpha * w00 + p10_f->alpha * w10 + p01_f->alpha * w01 + p11_f->alpha * w11;
        out_f->red = p00_f->red * w00 + p10_f->red * w10 + p01_f->red * w01 + p11_f->red * w11;
        out_f->green = p00_f->green * w00 + p10_f->green * w10 + p01_f->green * w01 + p11_f->green * w11;
        out_f->blue = p00_f->blue * w00 + p10_f->blue * w10 + p01_f->blue * w01 + p11_f->blue * w11;
    }
}

template <typename PixelType>
static PF_Err
StretchFunc(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PixelType* inP,
    PixelType* outP)
{
    PF_Err err = PF_Err_NONE;
    StretchInfo* siP = reinterpret_cast<StretchInfo*>(refcon);

    if (!siP || !siP->in_data) {
        *outP = *inP;      
        return PF_Err_BAD_CALLBACK_PARAM;
    }

    if (siP->content_only && inP->alpha == 0) {
        *outP = *inP;
        return err;
    }

    float width = (float)siP->width;
    float height = (float)siP->height;
    float scale = (float)siP->scale;
    float angle = (float)siP->angle;

    float x = (float)xL - siP->offset_x;
    float y = (float)yL - siP->offset_y;

    x -= siP->input_center_x;
    y -= siP->input_center_y;

    const float rad = -angle * (3.14159265358979323846f / 180.0f);

    const float cos_rad = cosf(rad);
    const float sin_rad = sinf(rad);
    const float cos_neg_rad = cosf(-rad);
    const float sin_neg_rad = sinf(-rad);

    float x_rot = x * cos_rad - y * sin_rad;
    float y_rot = x * sin_rad + y * cos_rad;

    if (scale != 0.0f) {
        x_rot /= scale;
    }

    float x_final = x_rot * cos_neg_rad - y_rot * sin_neg_rad;
    float y_final = x_rot * sin_neg_rad + y_rot * cos_neg_rad;

    float sampleX = x_final + siP->input_center_x;
    float sampleY = y_final + siP->input_center_y;

    float uvX = sampleX / width;
    float uvY = sampleY / height;

    SampleBilinear(siP->input, uvX, uvY, outP, siP->x_tiles, siP->y_tiles, siP->mirror);

    if (siP->content_only) {
        outP->alpha = inP->alpha;
    }

    return err;
}

static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        StretchParams* infoP = reinterpret_cast<StretchParams*>(pre_render_dataPV);
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

    StretchParams* infoP = reinterpret_cast<StretchParams*>(malloc(sizeof(StretchParams)));

    if (infoP) {
        AEFX_CLR_STRUCT(*infoP);

        PF_ParamDef cur_param;

        ERR(PF_CHECKOUT_PARAM(in_dataP, STRETCH_SCALE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->scale = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, STRETCH_ANGLE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->angle = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, STRETCH_CONTENT_ONLY, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->content_only = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, STRETCH_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->x_tiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, STRETCH_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->y_tiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, STRETCH_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mirror = cur_param.u.bd.value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            STRETCH_INPUT,
            STRETCH_INPUT,
            &req,
            in_dataP->current_time,
            in_dataP->time_step,
            in_dataP->time_scale,
            &in_result));

        UnionLRect(&in_result.result_rect, &extraP->output->result_rect);
        UnionLRect(&in_result.max_result_rect, &extraP->output->max_result_rect);
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
    StretchParams* infoP)
{
    PF_Err err = PF_Err_NONE;

    StretchInfo stretchInfo;
    AEFX_CLR_STRUCT(stretchInfo);

    stretchInfo.in_data = in_data;
    stretchInfo.input = input_worldP;
    stretchInfo.src = input_worldP->data;
    stretchInfo.rowbytes = input_worldP->rowbytes;
    stretchInfo.width = input_worldP->width;
    stretchInfo.height = input_worldP->height;
    stretchInfo.scale = infoP->scale;
    stretchInfo.angle = infoP->angle;
    stretchInfo.content_only = infoP->content_only;
    stretchInfo.x_tiles = infoP->x_tiles;
    stretchInfo.y_tiles = infoP->y_tiles;
    stretchInfo.mirror = infoP->mirror;

    stretchInfo.input_center_x = input_worldP->width / 2.0f;
    stretchInfo.input_center_y = input_worldP->height / 2.0f;
    stretchInfo.output_center_x = output_worldP->width / 2.0f;
    stretchInfo.output_center_y = output_worldP->height / 2.0f;
    stretchInfo.offset_x = stretchInfo.output_center_x - stretchInfo.input_center_x;
    stretchInfo.offset_y = stretchInfo.output_center_y - stretchInfo.input_center_y;

    PF_Point origin = { 0, 0 };

    if (!err) {
        switch (pixel_format) {
        case PF_PixelFormat_ARGB128: {
            AEFX_SuiteScoper<PF_iterateFloatSuite2> iterateFloatSuite =
                AEFX_SuiteScoper<PF_iterateFloatSuite2>(in_data,
                    kPFIterateFloatSuite,
                    kPFIterateFloatSuiteVersion2,
                    out_data);
            iterateFloatSuite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)&stretchInfo,
                StretchFunc<PF_PixelFloat>,
                output_worldP);
            break;
        }

        case PF_PixelFormat_ARGB64: {
            AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
                AEFX_SuiteScoper<PF_iterate16Suite2>(in_data,
                    kPFIterate16Suite,
                    kPFIterate16SuiteVersion2,
                    out_data);
            iterate16Suite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)&stretchInfo,
                StretchFunc<PF_Pixel16>,
                output_worldP);
            break;
        }

        case PF_PixelFormat_ARGB32: {
            AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
                AEFX_SuiteScoper<PF_Iterate8Suite2>(in_data,
                    kPFIterate8Suite,
                    kPFIterate8SuiteVersion2,
                    out_data);
            iterate8Suite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)&stretchInfo,
                StretchFunc<PF_Pixel8>,
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

size_t DivideRoundUp(
    size_t inValue,
    size_t inMultiple)
{
    return inValue ? (inValue + inMultiple - 1) / inMultiple : 0;
}

typedef struct
{
    int mSrcPitch;
    int mDstPitch;
    int m16f;
    int mWidth;
    int mHeight;
    float mScale;
    float mAngle;
    int mContentOnly;
    int mXTiles;
    int mYTiles;
    int mMirror;
} StretchAxisParams;

static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    StretchParams* infoP)
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

    StretchAxisParams params;
    params.mWidth = input_worldP->width;
    params.mHeight = input_worldP->height;
    params.mSrcPitch = input_worldP->rowbytes / bytes_per_pixel;
    params.mDstPitch = output_worldP->rowbytes / bytes_per_pixel;
    params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
    params.mScale = infoP->scale;
    params.mAngle = infoP->angle;
    params.mContentOnly = infoP->content_only;
    params.mXTiles = infoP->x_tiles;
    params.mYTiles = infoP->y_tiles;
    params.mMirror = infoP->mirror;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(float), &params.mScale));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(float), &params.mAngle));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.mContentOnly));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.mXTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.mYTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->stretch_kernel, param_index++, sizeof(int), &params.mMirror));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->stretch_kernel,
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
        StretchAxis_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            params.mSrcPitch,
            params.mDstPitch,
            params.m16f,
            params.mWidth,
            params.mHeight,
            params.mScale,
            params.mAngle,
            params.mContentOnly,
            params.mXTiles,
            params.mYTiles,
            params.mMirror);

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

        DXShaderExecution shaderExecution(
            dx_gpu_data->mContext,
            dx_gpu_data->mStretchShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(StretchAxisParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView(
            (ID3D12Resource*)dst_mem,
            params.mHeight * output_worldP->rowbytes));
        DX_ERR(shaderExecution.SetShaderResourceView(
            (ID3D12Resource*)src_mem,
            params.mHeight * input_worldP->rowbytes));
        DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(params.mWidth, 16), (UINT)DivideRoundUp(params.mHeight, 16)));
    }
#endif
#if HAS_METAL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        Handle metal_handle = (Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &params
            length : sizeof(StretchAxisParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->stretch_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(params.mWidth, threadsPerGroup.width), DivideRoundUp(params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->stretch_pipeline] ;
        [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
        [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
        [computeEncoder setBuffer : param_buffer offset : 0 atIndex : 2] ;
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
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    PF_EffectWorld* input_worldP = NULL, * output_worldP = NULL;

    StretchParams* infoP = reinterpret_cast<StretchParams*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, STRETCH_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, STRETCH_INPUT));
    }
    else {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}

static PF_Err
Render(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    if (in_dataP->appl_id == 'PrMr')
    {
        StretchParams stretchParams;
        stretchParams.scale = params[STRETCH_SCALE]->u.fs_d.value;
        stretchParams.angle = params[STRETCH_ANGLE]->u.fs_d.value;
        stretchParams.content_only = params[STRETCH_CONTENT_ONLY]->u.bd.value;
        stretchParams.x_tiles = params[STRETCH_X_TILES]->u.bd.value;
        stretchParams.y_tiles = params[STRETCH_Y_TILES]->u.bd.value;
        stretchParams.mirror = params[STRETCH_MIRROR]->u.bd.value;

        StretchInfo stretchInfo;
        AEFX_CLR_STRUCT(stretchInfo);

        stretchInfo.in_data = in_dataP;
        stretchInfo.input = &params[STRETCH_INPUT]->u.ld;
        stretchInfo.src = params[STRETCH_INPUT]->u.ld.data;
        stretchInfo.rowbytes = params[STRETCH_INPUT]->u.ld.rowbytes;
        stretchInfo.width = params[STRETCH_INPUT]->u.ld.width;
        stretchInfo.height = params[STRETCH_INPUT]->u.ld.height;
        stretchInfo.scale = stretchParams.scale;
        stretchInfo.angle = stretchParams.angle;
        stretchInfo.content_only = stretchParams.content_only;
        stretchInfo.x_tiles = stretchParams.x_tiles;
        stretchInfo.y_tiles = stretchParams.y_tiles;
        stretchInfo.mirror = stretchParams.mirror;

        stretchInfo.input_center_x = stretchInfo.width / 2.0f;
        stretchInfo.input_center_y = stretchInfo.height / 2.0f;
        stretchInfo.output_center_x = output->width / 2.0f;
        stretchInfo.output_center_y = output->height / 2.0f;
        stretchInfo.offset_x = stretchInfo.output_center_x - stretchInfo.input_center_x;
        stretchInfo.offset_y = stretchInfo.output_center_y - stretchInfo.input_center_y;

        PF_Point origin = { 0, 0 };

        for (int y = 0; y < output->height; y++) {
            for (int x = 0; x < output->width; x++) {
                PF_PixelFloat inPixel, outPixel;

                const float* srcData = (const float*)((const char*)params[STRETCH_INPUT]->u.ld.data +
                    y * params[STRETCH_INPUT]->u.ld.rowbytes + x * sizeof(PF_PixelFloat));
                inPixel = *(const PF_PixelFloat*)srcData;

                StretchFunc<PF_PixelFloat>(&stretchInfo, x, y, &inPixel, &outPixel);

                float* dstData = (float*)((char*)output->data + y * output->rowbytes + x * sizeof(PF_PixelFloat));
                *(PF_PixelFloat*)dstData = outPixel;
            }
        }
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
        "Stretch Axis",  
        "DKT Stretch Axis",   
        "DKT Effects",  
        AE_RESERVED_INFO,   
        "EffectMain",      
        "");      

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
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}


