#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "Swirl.h"
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

extern void Swirl_CUDA(
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
    int mirror);

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
    PF_Err	err = PF_Err_NONE;

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
    PF_Err      err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);

    PF_ADD_POINT(STR_CENTER_PARAM,
        50,
        50,
        false,
        CENTER_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_FLOAT_SLIDERX(STR_STRENGTH_PARAM,
        SWIRL_STRENGTH_MIN,
        SWIRL_STRENGTH_MAX,
        -0.1,
        0.1,
        0.1,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        STRENGTH_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_FLOAT_SLIDERX(STR_RADIUS_PARAM,
        SWIRL_RADIUS_MIN,
        SWIRL_RADIUS_MAX,
        SWIRL_RADIUS_MIN,
        SWIRL_RADIUS_MAX,
        SWIRL_RADIUS_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        RADIUS_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_START;
    PF_STRCPY(def.name, "Tiles");
    def.flags = PF_ParamFlag_START_COLLAPSED;      
    PF_ADD_PARAM(in_data, -1, &def);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(STR_X_TILES_PARAM,
        "",
        FALSE,
        0,
        X_TILES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(STR_Y_TILES_PARAM,
        "",
        FALSE,
        0,
        Y_TILES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(STR_MIRROR_PARAM,
        "",
        FALSE,
        0,
        MIRROR_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    PF_ADD_PARAM(in_data, -1, &def);

    out_data->num_params = SWIRL_NUM_PARAMS;

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
    cl_kernel swirl_kernel;
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
    ShaderObjectPtr mSwirlShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>swirl_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kSwirlKernel_OpenCLString) };
        char const* strings[] = { k16fString, kSwirlKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->swirl_kernel = clCreateKernel(program, "SwirlKernel", &result);
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
        dx_gpu_data->mSwirlShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"SwirlKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mSwirlShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kSwirlKernelMetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> swirl_function = nil;
            NSString* swirl_name = [NSString stringWithCString : "SwirlKernel" encoding : NSUTF8StringEncoding];

            swirl_function = [[library newFunctionWithName : swirl_name]autorelease];

            if (!swirl_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->swirl_pipeline = [device newComputePipelineStateWithFunction : swirl_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->swirl_kernel);

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
        dx_gpu_dataP->mSwirlShader.reset();

        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
            kPFHandleSuite,
            kPFHandleSuiteVersion1,
            out_dataP);

        handle_suite->host_dispose_handle(gpu_dataH);
    }
#endif

    return err;
}

template<typename T>
inline T CLAMP(T value, T min_val, T max_val) {
    return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

template<typename PixelT>
PixelT SampleBilinear(PF_EffectWorld* input, float x, float y, PF_PixelFormat format, bool x_tiles, bool y_tiles, bool mirror) {
    float width = static_cast<float>(input->width);
    float height = static_cast<float>(input->height);

    PixelT result;
    if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        result.alpha = 0.0f;
        result.red = 0.0f;
        result.green = 0.0f;
        result.blue = 0.0f;
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        result.alpha = 0;
        result.red = 0;
        result.green = 0;
        result.blue = 0;
    }
    else {
        result.alpha = 0;
        result.red = 0;
        result.green = 0;
        result.blue = 0;
    }

    float u = x / width;
    float v = y / height;

    if (x_tiles) {
        if (mirror) {
            float intPart;
            float fracPart = modff(fabsf(u), &intPart);
            int isOdd = (int)intPart & 1;
            u = isOdd ? 1.0f - fracPart : fracPart;
        }
        else {
            u = u - floorf(u);
        }
    }
    else if (u < 0.0f || u > 1.0f) {
        return result;
    }
    else {
        u = CLAMP(u, 0.0f, 0.9999f);
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
        }
    }
    else if (v < 0.0f || v > 1.0f) {
        return result;
    }
    else {
        v = CLAMP(v, 0.0f, 0.9999f);
    }

    x = u * width;
    y = v * height;

    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = CLAMP(x1 + 1, 0, input->width - 1);
    int y2 = CLAMP(y1 + 1, 0, input->height - 1);

    float fx = x - x1;
    float fy = y - y1;

    PixelT* p1 = NULL, * p2 = NULL, * p3 = NULL, * p4 = NULL;
    PF_PixelPtr row1 = NULL, row2 = NULL;

    switch (format) {
    case PF_PixelFormat_ARGB128:
        row1 = reinterpret_cast<PF_PixelPtr>((reinterpret_cast<char*>(input->data) + y1 * input->rowbytes));
        row2 = reinterpret_cast<PF_PixelPtr>((reinterpret_cast<char*>(input->data) + y2 * input->rowbytes));
        p1 = reinterpret_cast<PixelT*>(row1) + x1;
        p2 = reinterpret_cast<PixelT*>(row1) + x2;
        p3 = reinterpret_cast<PixelT*>(row2) + x1;
        p4 = reinterpret_cast<PixelT*>(row2) + x2;
        break;

    case PF_PixelFormat_ARGB64:
        row1 = reinterpret_cast<PF_PixelPtr>((reinterpret_cast<char*>(input->data) + y1 * input->rowbytes));
        row2 = reinterpret_cast<PF_PixelPtr>((reinterpret_cast<char*>(input->data) + y2 * input->rowbytes));
        p1 = reinterpret_cast<PixelT*>(row1) + x1;
        p2 = reinterpret_cast<PixelT*>(row1) + x2;
        p3 = reinterpret_cast<PixelT*>(row2) + x1;
        p4 = reinterpret_cast<PixelT*>(row2) + x2;
        break;

    case PF_PixelFormat_ARGB32:
    default:
        row1 = reinterpret_cast<PF_PixelPtr>((reinterpret_cast<char*>(input->data) + y1 * input->rowbytes));
        row2 = reinterpret_cast<PF_PixelPtr>((reinterpret_cast<char*>(input->data) + y2 * input->rowbytes));
        p1 = reinterpret_cast<PixelT*>(row1) + x1;
        p2 = reinterpret_cast<PixelT*>(row1) + x2;
        p3 = reinterpret_cast<PixelT*>(row2) + x1;
        p4 = reinterpret_cast<PixelT*>(row2) + x2;
        break;
    }

    if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        result.alpha = (1 - fx) * (1 - fy) * p1->alpha + fx * (1 - fy) * p2->alpha + (1 - fx) * fy * p3->alpha + fx * fy * p4->alpha;
        result.red = (1 - fx) * (1 - fy) * p1->red + fx * (1 - fy) * p2->red + (1 - fx) * fy * p3->red + fx * fy * p4->red;
        result.green = (1 - fx) * (1 - fy) * p1->green + fx * (1 - fy) * p2->green + (1 - fx) * fy * p3->green + fx * fy * p4->green;
        result.blue = (1 - fx) * (1 - fy) * p1->blue + fx * (1 - fy) * p2->blue + (1 - fx) * fy * p3->blue + fx * fy * p4->blue;
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        result.alpha = static_cast<A_u_short>((1 - fx) * (1 - fy) * p1->alpha + fx * (1 - fy) * p2->alpha + (1 - fx) * fy * p3->alpha + fx * fy * p4->alpha);
        result.red = static_cast<A_u_short>((1 - fx) * (1 - fy) * p1->red + fx * (1 - fy) * p2->red + (1 - fx) * fy * p3->red + fx * fy * p4->red);
        result.green = static_cast<A_u_short>((1 - fx) * (1 - fy) * p1->green + fx * (1 - fy) * p2->green + (1 - fx) * fy * p3->green + fx * fy * p4->green);
        result.blue = static_cast<A_u_short>((1 - fx) * (1 - fy) * p1->blue + fx * (1 - fy) * p2->blue + (1 - fx) * fy * p3->blue + fx * fy * p4->blue);
    }
    else {
        result.alpha = static_cast<A_u_char>((1 - fx) * (1 - fy) * p1->alpha + fx * (1 - fy) * p2->alpha + (1 - fx) * fy * p3->alpha + fx * fy * p4->alpha);
        result.red = static_cast<A_u_char>((1 - fx) * (1 - fy) * p1->red + fx * (1 - fy) * p2->red + (1 - fx) * fy * p3->red + fx * fy * p4->red);
        result.green = static_cast<A_u_char>((1 - fx) * (1 - fy) * p1->green + fx * (1 - fy) * p2->green + (1 - fx) * fy * p3->green + fx * fy * p4->green);
        result.blue = static_cast<A_u_char>((1 - fx) * (1 - fy) * p1->blue + fx * (1 - fy) * p2->blue + (1 - fx) * fy * p3->blue + fx * fy * p4->blue);
    }

    return result;
}

template<typename PixelT>
static PF_Err
SwirlFunc(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PixelT* inP,
    PixelT* outP)
{
    PF_Err err = PF_Err_NONE;
    SwirlParams* siP = reinterpret_cast<SwirlParams*>(refcon);

    if (!siP) return PF_Err_BAD_CALLBACK_PARAM;

    float width = static_cast<float>(siP->width);
    float height = static_cast<float>(siP->height);
    float x = static_cast<float>(xL) / width;
    float y = static_cast<float>(yL) / height;

    float centerX = siP->centerX / width;
    float centerY = siP->centerY / height;

    centerY = 1.0f - centerY;

    float dx = x - centerX;
    float dy = y - centerY;

    float convRateX = 1.0f;
    float convRateY = 1.0f;

    if (height > width) {
        convRateY = height / width;
    }
    else {
        convRateX = width / height;
    }

    dx *= convRateX;
    dy *= convRateY;

    float dist = sqrt(dx * dx + dy * dy);

    float srcX = x;
    float srcY = y;

    if (dist < siP->radius) {
        float percent = (siP->radius - dist) / siP->radius;

        float T = siP->strength;
        float A = (T <= 0.5f) ?
            ((T / 0.5f)) :
            (1.0f - ((T - 0.5f) / 0.5f));

        float theta = percent * percent * A * 8.0f * 3.14159f;
        float sinTheta = -sin(theta);
        float cosTheta = cos(theta);

        float newDx = dx * cosTheta - dy * sinTheta;
        float newDy = dx * sinTheta + dy * cosTheta;

        newDx /= convRateX;
        newDy /= convRateY;

        srcX = centerX + newDx;
        srcY = centerY + newDy;
    }

    PF_PixelFormat format = PF_PixelFormat_ARGB32;
    if constexpr (std::is_same_v<PixelT, PF_PixelFloat>)
        format = PF_PixelFormat_ARGB128;
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>)
        format = PF_PixelFormat_ARGB64;

    srcX *= width;
    srcY *= height;

    *outP = SampleBilinear<PixelT>(siP->inputP, srcX, srcY, format, siP->x_tiles, siP->y_tiles, siP->mirror);

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
    AEGP_SuiteHandler suites(in_dataP->pica_basicP);

    SwirlParams siP;
    AEFX_CLR_STRUCT(siP);
    A_long linesL = 0;

    linesL = output->extent_hint.bottom - output->extent_hint.top;

    siP.centerX = params[SWIRL_CENTER]->u.td.x_value / 65536.0f;
    siP.centerY = params[SWIRL_CENTER]->u.td.y_value / 65536.0f;
    siP.strength = params[SWIRL_STRENGTH]->u.fs_d.value;
    siP.radius = params[SWIRL_RADIUS]->u.fs_d.value;

    siP.x_tiles = params[SWIRL_X_TILES]->u.bd.value;
    siP.y_tiles = params[SWIRL_Y_TILES]->u.bd.value;
    siP.mirror = params[SWIRL_MIRROR]->u.bd.value;

    siP.width = output->width;
    siP.height = output->height;
    siP.inputP = &params[SWIRL_INPUT]->u.ld;

    PF_PixelFormat pixelFormat;
    PF_WorldSuite2* wsP = NULL;
    ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
    ERR(wsP->PF_GetPixelFormat(output, &pixelFormat));
    ERR(suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2));

    switch (pixelFormat) {
    case PF_PixelFormat_ARGB128:
        ERR(suites.IterateFloatSuite1()->iterate(
            in_dataP,
            0,
            linesL,
            &params[SWIRL_INPUT]->u.ld,
            NULL,
            (void*)&siP,
            (PF_IteratePixelFloatFunc)SwirlFunc<PF_PixelFloat>,
            output));
        break;

    case PF_PixelFormat_ARGB64:
        ERR(suites.Iterate16Suite1()->iterate(
            in_dataP,
            0,
            linesL,
            &params[SWIRL_INPUT]->u.ld,
            NULL,
            (void*)&siP,
            SwirlFunc<PF_Pixel16>,
            output));
        break;

    case PF_PixelFormat_ARGB32:
    default:
        ERR(suites.Iterate8Suite1()->iterate(
            in_dataP,
            0,
            linesL,
            &params[SWIRL_INPUT]->u.ld,
            NULL,
            (void*)&siP,
            SwirlFunc<PF_Pixel8>,
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
        SwirlParams* infoP = reinterpret_cast<SwirlParams*>(pre_render_dataPV);
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

    SwirlParams* infoP = reinterpret_cast<SwirlParams*>(malloc(sizeof(SwirlParams)));

    if (infoP) {
        PF_ParamDef cur_param;

        ERR(PF_CHECKOUT_PARAM(in_dataP, SWIRL_CENTER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->centerX = cur_param.u.td.x_value / 65536.0f;
        infoP->centerY = cur_param.u.td.y_value / 65536.0f;

        ERR(PF_CHECKOUT_PARAM(in_dataP, SWIRL_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->strength = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, SWIRL_RADIUS, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->radius = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, SWIRL_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->x_tiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, SWIRL_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->y_tiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, SWIRL_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mirror = cur_param.u.bd.value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            SWIRL_INPUT,
            SWIRL_INPUT,
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

typedef struct
{
    int mSrcPitch;
    int mDstPitch;
    int m16f;
    int mWidth;
    int mHeight;
    float mCenterX;
    float mCenterY;
    float mStrength;
    float mRadius;
    int mXTiles;
    int mYTiles;
    int mMirror;
} SwirlKernelParams;

static size_t
RoundUp(size_t inValue, size_t inMultiple)
{
    return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}

size_t DivideRoundUp(size_t inValue, size_t inMultiple)
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
    SwirlParams* infoP)
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

    SwirlKernelParams swirl_params;

    swirl_params.mWidth = input_worldP->width;
    swirl_params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    swirl_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    swirl_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    swirl_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    swirl_params.mCenterX = infoP->centerX;
    swirl_params.mCenterY = infoP->centerY;

    swirl_params.mStrength = infoP->strength;
    swirl_params.mRadius = infoP->radius;
    swirl_params.mXTiles = infoP->x_tiles;
    swirl_params.mYTiles = infoP->y_tiles;
    swirl_params.mMirror = infoP->mirror;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint swirl_param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(int), &swirl_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(int), &swirl_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(int), &swirl_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(int), &swirl_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(int), &swirl_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(float), &swirl_params.mCenterX));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(float), &swirl_params.mCenterY));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(float), &swirl_params.mStrength));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(float), &swirl_params.mRadius));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(int), &swirl_params.mXTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(int), &swirl_params.mYTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swirl_kernel, swirl_param_index++, sizeof(int), &swirl_params.mMirror));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(swirl_params.mWidth, threadBlock[0]), RoundUp(swirl_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->swirl_kernel,
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
        Swirl_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            swirl_params.mSrcPitch,
            swirl_params.mDstPitch,
            swirl_params.m16f,
            swirl_params.mWidth,
            swirl_params.mHeight,
            swirl_params.mCenterX,
            swirl_params.mCenterY,
            swirl_params.mStrength,
            swirl_params.mRadius,
            swirl_params.mXTiles,
            swirl_params.mYTiles,
            swirl_params.mMirror);

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
            dx_gpu_data->mSwirlShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&swirl_params, sizeof(SwirlKernelParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView(
            (ID3D12Resource*)dst_mem,
            swirl_params.mHeight * dst_row_bytes));
        DX_ERR(shaderExecution.SetShaderResourceView(
            (ID3D12Resource*)src_mem,
            swirl_params.mHeight * src_row_bytes));
        DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(swirl_params.mWidth, 16), (UINT)DivideRoundUp(swirl_params.mHeight, 16)));
    }
#endif
#if HAS_METAL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        Handle metal_handle = (Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> swirl_param_buffer = [[device newBufferWithBytes : &swirl_params
            length : sizeof(SwirlKernelParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->swirl_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(swirl_params.mWidth, threadsPerGroup.width), DivideRoundUp(swirl_params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->swirl_pipeline] ;
        [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
        [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
        [computeEncoder setBuffer : swirl_param_buffer offset : 0 atIndex : 2] ;
        [computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;
        [computeEncoder endEncoding] ;
        [commandBuffer commit] ;

        err = NSError2PFErr([commandBuffer error]);
    }
#endif 

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
    SwirlParams* infoP)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    infoP->width = input_worldP->width;
    infoP->height = input_worldP->height;
    infoP->inputP = input_worldP;

    switch (pixel_format) {
    case PF_PixelFormat_ARGB128:
        ERR(suites.IterateFloatSuite1()->iterate(
            in_data,
            0,
            output_worldP->height,
            input_worldP,
            NULL,
            (void*)infoP,
            (PF_IteratePixelFloatFunc)SwirlFunc<PF_PixelFloat>,
            output_worldP));
        break;

    case PF_PixelFormat_ARGB64:
        ERR(suites.Iterate16Suite1()->iterate(
            in_data,
            0,
            output_worldP->height,
            input_worldP,
            NULL,
            (void*)infoP,
            SwirlFunc<PF_Pixel16>,
            output_worldP));
        break;

    case PF_PixelFormat_ARGB32:
    default:
        ERR(suites.Iterate8Suite1()->iterate(
            in_data,
            0,
            output_worldP->height,
            input_worldP,
            NULL,
            (void*)infoP,
            SwirlFunc<PF_Pixel8>,
            output_worldP));
        break;
    }

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

    SwirlParams* infoP = reinterpret_cast<SwirlParams*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, SWIRL_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, SWIRL_INPUT));
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
        STR_NAME,
        "DKT Swirl",
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