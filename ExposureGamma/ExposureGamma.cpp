#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "ExposureGamma.h"
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

extern void ExposureGamma_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float inExposure,
    float inGamma,
    float inOffset);

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

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Exposure",
        EXPOSURE_MIN_VALUE,
        EXPOSURE_MAX_VALUE,
        EXPOSURE_MIN_SLIDER,
        EXPOSURE_MAX_SLIDER,
        EXPOSURE_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        EXPOSUREGAMMA_EXPOSURE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Gamma",
        GAMMA_MIN_VALUE,
        GAMMA_MAX_VALUE,
        GAMMA_MIN_SLIDER,
        GAMMA_MAX_SLIDER,
        GAMMA_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        EXPOSUREGAMMA_GAMMA);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Offset",
        OFFSET_MIN_VALUE,
        OFFSET_MAX_VALUE,
        OFFSET_MIN_SLIDER,
        OFFSET_MAX_SLIDER,
        OFFSET_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        EXPOSUREGAMMA_OFFSET);

    out_data->num_params = EXPOSUREGAMMA_NUM_PARAMS;

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
    cl_kernel exposure_gamma_kernel;
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
    ShaderObjectPtr mExposureGammaShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>exposure_gamma_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kExposureGammaKernel_OpenCLString) };
        char const* strings[] = { k16fString, kExposureGammaKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->exposure_gamma_kernel = clCreateKernel(program, "ExposureGammaKernel", &result);
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
        dx_gpu_data->mExposureGammaShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"ExposureGammaKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mExposureGammaShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kExposureGammaKernelMetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> exposure_gamma_function = nil;
            NSString* kernel_name = [NSString stringWithCString : "ExposureGammaKernel" encoding : NSUTF8StringEncoding];

            exposure_gamma_function = [[library newFunctionWithName : kernel_name]autorelease];

            if (!exposure_gamma_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->exposure_gamma_pipeline = [device newComputePipelineStateWithFunction : exposure_gamma_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->exposure_gamma_kernel);

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
        dx_gpu_dataP->mExposureGammaShader.reset();

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

        [metal_dataP->exposure_gamma_pipeline release] ;

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
static PF_Err
ProcessExposureGamma(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    PF_Err err = PF_Err_NONE;

    ExposureGammaInfo* infoP = reinterpret_cast<ExposureGammaInfo*>(refcon);

    if (infoP) {
        outP->alpha = inP->alpha;

        PF_FpLong redF, greenF, blueF;
        PF_FpLong alphaF;

        if constexpr (std::is_same<PixelType, PF_Pixel8>::value) {
            redF = (PF_FpLong)inP->red / PF_MAX_CHAN8;
            greenF = (PF_FpLong)inP->green / PF_MAX_CHAN8;
            blueF = (PF_FpLong)inP->blue / PF_MAX_CHAN8;
            alphaF = (PF_FpLong)inP->alpha / PF_MAX_CHAN8;
        }
        else if constexpr (std::is_same<PixelType, PF_Pixel16>::value) {
            redF = (PF_FpLong)inP->red / PF_MAX_CHAN16;
            greenF = (PF_FpLong)inP->green / PF_MAX_CHAN16;
            blueF = (PF_FpLong)inP->blue / PF_MAX_CHAN16;
            alphaF = (PF_FpLong)inP->alpha / PF_MAX_CHAN16;
        }
        else {  
            redF = inP->red;
            greenF = inP->green;
            blueF = inP->blue;
            alphaF = inP->alpha;
        }

        redF += infoP->offset * alphaF;
        greenF += infoP->offset * alphaF;
        blueF += infoP->offset * alphaF;

        if (infoP->gamma != 0) {
            redF = pow(MAX(redF, 0.0), 1.0 / infoP->gamma);
            greenF = pow(MAX(greenF, 0.0), 1.0 / infoP->gamma);
            blueF = pow(MAX(blueF, 0.0), 1.0 / infoP->gamma);
        }

        PF_FpLong exposureFactor = pow(2.0, infoP->exposure);
        redF *= exposureFactor;
        greenF *= exposureFactor;
        blueF *= exposureFactor;

        if constexpr (std::is_same<PixelType, PF_Pixel8>::value) {
            outP->red = (A_u_char)MIN(MAX(redF * PF_MAX_CHAN8, 0), PF_MAX_CHAN8);
            outP->green = (A_u_char)MIN(MAX(greenF * PF_MAX_CHAN8, 0), PF_MAX_CHAN8);
            outP->blue = (A_u_char)MIN(MAX(blueF * PF_MAX_CHAN8, 0), PF_MAX_CHAN8);
        }
        else if constexpr (std::is_same<PixelType, PF_Pixel16>::value) {
            outP->red = (A_u_short)MIN(MAX(redF * PF_MAX_CHAN16, 0), PF_MAX_CHAN16);
            outP->green = (A_u_short)MIN(MAX(greenF * PF_MAX_CHAN16, 0), PF_MAX_CHAN16);
            outP->blue = (A_u_short)MIN(MAX(blueF * PF_MAX_CHAN16, 0), PF_MAX_CHAN16);
        }
        else {  
            outP->red = MIN(MAX(redF, 0), 1);
            outP->green = MIN(MAX(greenF, 0), 1);
            outP->blue = MIN(MAX(blueF, 0), 1);
        }
    }

    return err;
}

static PF_Err
ProcessExposureGamma8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return ProcessExposureGamma<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err
ProcessExposureGamma16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return ProcessExposureGamma<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err
ProcessExposureGammaFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return ProcessExposureGamma<PF_PixelFloat>(refcon, xL, yL, inP, outP);
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
        PF_LayerDef* src = &params[0]->u.ld;
        PF_LayerDef* dest = output;

        const char* srcData = (const char*)src->data;
        char* destData = (char*)dest->data;

        float exposure = params[EXPOSUREGAMMA_EXPOSURE]->u.fs_d.value;
        float gamma = params[EXPOSUREGAMMA_GAMMA]->u.fs_d.value;
        float offset = params[EXPOSUREGAMMA_OFFSET]->u.fs_d.value;

        for (int y = 0; y < output->height; ++y, srcData += src->rowbytes, destData += dest->rowbytes)
        {
            for (int x = 0; x < output->width; ++x)
            {
                float v = ((const float*)srcData)[x * 4 + 0];
                float u = ((const float*)srcData)[x * 4 + 1];
                float y = ((const float*)srcData)[x * 4 + 2];
                float a = ((const float*)srcData)[x * 4 + 3];

                y += offset * a;

                if (gamma != 0.0f) {
                    y = pow(MAX(y, 0.0f), 1.0f / gamma);
                }

                float exposureFactor = pow(2.0f, exposure);
                y *= exposureFactor;

                y = MIN(MAX(y, 0.0f), 1.0f);

                ((float*)destData)[x * 4 + 0] = v;
                ((float*)destData)[x * 4 + 1] = u;
                ((float*)destData)[x * 4 + 2] = y;
                ((float*)destData)[x * 4 + 3] = a;
            }
        }
    }

    return err;
}

static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        ExposureGammaInfo* infoP = reinterpret_cast<ExposureGammaInfo*>(pre_render_dataPV);
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

    ExposureGammaInfo* infoP = reinterpret_cast<ExposureGammaInfo*>(malloc(sizeof(ExposureGammaInfo)));

    if (infoP) {
        PF_ParamDef cur_param;
        ERR(PF_CHECKOUT_PARAM(in_dataP, EXPOSUREGAMMA_EXPOSURE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->exposure = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, EXPOSUREGAMMA_GAMMA, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->gamma = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, EXPOSUREGAMMA_OFFSET, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->offset = cur_param.u.fs_d.value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            EXPOSUREGAMMA_INPUT,
            EXPOSUREGAMMA_INPUT,
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
    ExposureGammaInfo* infoP)
{
    PF_Err err = PF_Err_NONE;

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
                (void*)infoP,
                ProcessExposureGammaFloat,
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
                (void*)infoP,
                ProcessExposureGamma16,
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
                (void*)infoP,
                ProcessExposureGamma8,
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
    unsigned int mWidth;
    unsigned int mHeight;
    float mExposure;
    float mGamma;
    float mOffset;
} ExposureGammaParams;


static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    ExposureGammaInfo* infoP)
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

    ExposureGammaParams eg_params;

    eg_params.mWidth = input_worldP->width;
    eg_params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    eg_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    eg_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    eg_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    eg_params.mExposure = infoP->exposure;
    eg_params.mGamma = infoP->gamma;
    eg_params.mOffset = infoP->offset;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(int), &eg_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(int), &eg_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(int), &eg_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(int), &eg_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(int), &eg_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(float), &eg_params.mExposure));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(float), &eg_params.mGamma));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->exposure_gamma_kernel, param_index++, sizeof(float), &eg_params.mOffset));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(eg_params.mWidth, threadBlock[0]), RoundUp(eg_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->exposure_gamma_kernel,
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
        ExposureGamma_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            eg_params.mSrcPitch,
            eg_params.mDstPitch,
            eg_params.m16f,
            eg_params.mWidth,
            eg_params.mHeight,
            eg_params.mExposure,
            eg_params.mGamma,
            eg_params.mOffset);

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
            dx_gpu_data->mExposureGammaShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&eg_params, sizeof(ExposureGammaParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView(
            (ID3D12Resource*)dst_mem,
            eg_params.mHeight * dst_row_bytes));
        DX_ERR(shaderExecution.SetShaderResourceView(
            (ID3D12Resource*)src_mem,
            eg_params.mHeight * src_row_bytes));
        DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(eg_params.mWidth, 16), (UINT)DivideRoundUp(eg_params.mHeight, 16)));
    }
#endif
#if HAS_METAL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        PF_Handle metal_handle = (PF_Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &eg_params
            length : sizeof(ExposureGammaParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->exposure_gamma_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(eg_params.mWidth, threadsPerGroup.width), DivideRoundUp(eg_params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->exposure_gamma_pipeline] ;
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
    PF_Err err = PF_Err_NONE,
        err2 = PF_Err_NONE;

    PF_EffectWorld* input_worldP = NULL,
        * output_worldP = NULL;

    ExposureGammaInfo* infoP = reinterpret_cast<ExposureGammaInfo*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, EXPOSUREGAMMA_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, EXPOSUREGAMMA_INPUT));
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
        "Exposure/Gamma",  
        "DKT Exposure Gamma",   
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

