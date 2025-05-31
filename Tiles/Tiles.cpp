#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "Tiles.h"
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

extern void Tiles_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float cropF);

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

    PF_ADD_FLOAT_SLIDERX(
        "Crop",
        CROP_MIN_VALUE,
        CROP_MAX_VALUE,
        CROP_MIN_SLIDER,
        CROP_MAX_SLIDER,
        CROP_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        TILES_CROP);

    out_data->num_params = TILES_NUM_PARAMS;

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
    cl_kernel tiles_kernel;
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
    ShaderObjectPtr mTilesShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>tiles_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kTilesKernel_OpenCLString) };
        char const* strings[] = { k16fString, kTilesKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->tiles_kernel = clCreateKernel(program, "TilesKernel", &result);
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
        dx_gpu_data->mTilesShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"TilesKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mTilesShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kTiles_Kernel_MetalString encoding : NSUTF8StringEncoding];
        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

        NSError* error = nil;
        id<MTLLibrary> library = [[device newLibraryWithSource : source options : nil error : &error]autorelease];

        if (!err && !library) {
            err = NSError2PFErr(error);
        }

        PF_Handle metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
        MetalGPUData* metal_data = reinterpret_cast<MetalGPUData*>(*metal_handle);

        if (err == PF_Err_NONE)
        {
            id<MTLFunction> tiles_function = nil;
            NSString* tiles_name = [NSString stringWithCString : "TilesKernel" encoding : NSUTF8StringEncoding];

            tiles_function = [[library newFunctionWithName : tiles_name]autorelease];

            if (!tiles_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->tiles_pipeline = [device newComputePipelineStateWithFunction : tiles_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->tiles_kernel);

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
        dx_gpu_dataP->mTilesShader.reset();

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
static PixelT
SampleBilinear(
    PF_EffectWorld* input,
    float x,
    float y)
{
    PixelT result;

    if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
        result.red = 0;
        result.green = 0;
        result.blue = 0;
        result.alpha = 0;
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        result.red = 0;
        result.green = 0;
        result.blue = 0;
        result.alpha = 0;
    }
    else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        result.red = 0.0f;
        result.green = 0.0f;
        result.blue = 0.0f;
        result.alpha = 0.0f;
    }

    if (x < 0.0f || x >= input->width || y < 0.0f || y >= input->height) {
        return result;
    }

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = fmin(x0 + 1, input->width - 1);
    int y1 = fmin(y0 + 1, input->height - 1);

    float fx = x - x0;
    float fy = y - y0;

    PixelT* p00 = NULL;
    PixelT* p01 = NULL;
    PixelT* p10 = NULL;
    PixelT* p11 = NULL;

    p00 = (PixelT*)((char*)input->data + y0 * input->rowbytes + x0 * sizeof(PixelT));
    p01 = (PixelT*)((char*)input->data + y0 * input->rowbytes + x1 * sizeof(PixelT));
    p10 = (PixelT*)((char*)input->data + y1 * input->rowbytes + x0 * sizeof(PixelT));
    p11 = (PixelT*)((char*)input->data + y1 * input->rowbytes + x1 * sizeof(PixelT));

    float red = (1 - fx) * (1 - fy) * p00->red +
        fx * (1 - fy) * p01->red +
        (1 - fx) * fy * p10->red +
        fx * fy * p11->red;

    float green = (1 - fx) * (1 - fy) * p00->green +
        fx * (1 - fy) * p01->green +
        (1 - fx) * fy * p10->green +
        fx * fy * p11->green;

    float blue = (1 - fx) * (1 - fy) * p00->blue +
        fx * (1 - fy) * p01->blue +
        (1 - fx) * fy * p10->blue +
        fx * fy * p11->blue;

    float alpha = (1 - fx) * (1 - fy) * p00->alpha +
        fx * (1 - fy) * p01->alpha +
        (1 - fx) * fy * p10->alpha +
        fx * fy * p11->alpha;

    if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
        result.red = (A_u_char)fmin(fmax(red + 0.5f, 0.0f), 255.0f);
        result.green = (A_u_char)fmin(fmax(green + 0.5f, 0.0f), 255.0f);
        result.blue = (A_u_char)fmin(fmax(blue + 0.5f, 0.0f), 255.0f);
        result.alpha = (A_u_char)fmin(fmax(alpha + 0.5f, 0.0f), 255.0f);
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        result.red = (A_u_short)fmin(fmax(red + 0.5f, 0.0f), 32767.0f);
        result.green = (A_u_short)fmin(fmax(green + 0.5f, 0.0f), 32767.0f);
        result.blue = (A_u_short)fmin(fmax(blue + 0.5f, 0.0f), 32767.0f);
        result.alpha = (A_u_short)fmin(fmax(alpha + 0.5f, 0.0f), 32767.0f);
    }
    else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        result.red = red;
        result.green = green;
        result.blue = blue;
        result.alpha = alpha;
    }

    return result;
}

template<typename PixelT>
static PF_Err
ProcessCropFunc(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelT* inP,
    PixelT* outP)
{
    PF_Err err = PF_Err_NONE;
    CropInfo* ciP = reinterpret_cast<CropInfo*>(refcon);
    PF_EffectWorld* input = ciP->input;

    if (!ciP || !input) return PF_Err_BAD_CALLBACK_PARAM;

    float centerX = input->width / 2.0f;
    float centerY = input->height / 2.0f;

    float pixelSizeX = 1.0f / input->width;
    float pixelSizeY = 1.0f / input->height;
    float pixelSize = (pixelSizeX + pixelSizeY) / 2.0f;

    float adjustedCropF = ciP->cropF + pixelSize * 4.0f;

    float srcX = centerX + (xL - centerX) / adjustedCropF;
    float srcY = centerY + (yL - centerY) / adjustedCropF;

    *outP = SampleBilinear<PixelT>(input, srcX, srcY);

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

    CropInfo ci;
    AEFX_CLR_STRUCT(ci);
    ci.cropF = params[TILES_CROP]->u.fs_d.value;
    ci.input = &params[TILES_INPUT]->u.ld;

    PF_PixelFormat pixelFormat;
    PF_WorldSuite2* wsP = NULL;
    ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
    ERR(wsP->PF_GetPixelFormat(output, &pixelFormat));
    ERR(suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2));

    A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

    switch (pixelFormat) {
    case PF_PixelFormat_ARGB128:
        ERR(suites.IterateFloatSuite1()->iterate(
            in_dataP,
            0,
            linesL,
            &params[TILES_INPUT]->u.ld,
            NULL,
            (void*)&ci,
            (PF_IteratePixelFloatFunc)ProcessCropFunc<PF_PixelFloat>,
            output));
        break;

    case PF_PixelFormat_ARGB64:
        ERR(suites.Iterate16Suite1()->iterate(
            in_dataP,
            0,
            linesL,
            &params[TILES_INPUT]->u.ld,
            NULL,
            (void*)&ci,
            ProcessCropFunc<PF_Pixel16>,
            output));
        break;

    case PF_PixelFormat_ARGB32:
    default:
        ERR(suites.Iterate8Suite1()->iterate(
            in_dataP,
            0,
            linesL,
            &params[TILES_INPUT]->u.ld,
            NULL,
            (void*)&ci,
            ProcessCropFunc<PF_Pixel8>,
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
        CropInfo* infoP = reinterpret_cast<CropInfo*>(pre_render_dataPV);
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

    CropInfo* infoP = reinterpret_cast<CropInfo*>(malloc(sizeof(CropInfo)));

    if (infoP) {
        PF_ParamDef cur_param;
        ERR(PF_CHECKOUT_PARAM(in_dataP, TILES_CROP, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->cropF = cur_param.u.fs_d.value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            TILES_INPUT,
            TILES_INPUT,
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
    CropInfo* infoP)
{
    PF_Err err = PF_Err_NONE;
    infoP->input = input_worldP;

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
                ProcessCropFunc<PF_PixelFloat>,
                output_worldP);
            break;
        }

        case PF_PixelFormat_ARGB64: {
            AEFX_SuiteScoper<PF_iterate16Suite1> iterate16Suite =
                AEFX_SuiteScoper<PF_iterate16Suite1>(in_data,
                    kPFIterate16Suite,
                    kPFIterate16SuiteVersion2,
                    out_data);
            iterate16Suite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)infoP,
                ProcessCropFunc<PF_Pixel16>,
                output_worldP);
            break;
        }

        case PF_PixelFormat_ARGB32: {
            AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
                AEFX_SuiteScoper<PF_Iterate8Suite1>(in_data,
                    kPFIterate8Suite,
                    kPFIterate8SuiteVersion2,
                    out_data);

            iterate8Suite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)infoP,
                ProcessCropFunc<PF_Pixel8>,
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
    float mCropF;
} TilesParams;

static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    CropInfo* infoP)
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

    TilesParams tiles_params;
    tiles_params.mWidth = input_worldP->width;
    tiles_params.mHeight = input_worldP->height;
    tiles_params.mCropF = infoP->cropF;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    tiles_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    tiles_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    tiles_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->tiles_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->tiles_kernel, param_index++, sizeof(int), &tiles_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->tiles_kernel, param_index++, sizeof(int), &tiles_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->tiles_kernel, param_index++, sizeof(int), &tiles_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->tiles_kernel, param_index++, sizeof(int), &tiles_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->tiles_kernel, param_index++, sizeof(int), &tiles_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->tiles_kernel, param_index++, sizeof(float), &tiles_params.mCropF));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(tiles_params.mWidth, threadBlock[0]), RoundUp(tiles_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->tiles_kernel,
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
        Tiles_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            tiles_params.mSrcPitch,
            tiles_params.mDstPitch,
            tiles_params.m16f,
            tiles_params.mWidth,
            tiles_params.mHeight,
            tiles_params.mCropF);

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
            dx_gpu_data->mTilesShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&tiles_params, sizeof(TilesParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView(
            (ID3D12Resource*)dst_mem,
            tiles_params.mHeight * dst_row_bytes));
        DX_ERR(shaderExecution.SetShaderResourceView(
            (ID3D12Resource*)src_mem,
            tiles_params.mHeight * src_row_bytes));
        DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(tiles_params.mWidth, 16), (UINT)DivideRoundUp(tiles_params.mHeight, 16)));
    }
#endif
#if HAS_METAL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        Handle metal_handle = (Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> tiles_param_buffer = [[device newBufferWithBytes : &tiles_params
            length : sizeof(TilesParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->tiles_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(tiles_params.mWidth, threadsPerGroup.width), DivideRoundUp(tiles_params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->tiles_pipeline] ;
        [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
        [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
        [computeEncoder setBuffer : tiles_param_buffer offset : 0 atIndex : 2] ;
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

    CropInfo* infoP = reinterpret_cast<CropInfo*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, TILES_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, TILES_INPUT));
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
        "Tiles",
        "DKT Tiles",
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