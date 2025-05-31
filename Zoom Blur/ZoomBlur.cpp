#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "ZoomBlur.h"
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

extern void ZoomBlur_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float strength,
    float centerX,
    float centerY,
    int adaptive);

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
    PF_ADD_FLOAT_SLIDERX(STR_STRENGTH_PARAM_NAME,
        ZOOMBLUR_STRENGTH_MIN,
        ZOOMBLUR_STRENGTH_MAX,
        ZOOMBLUR_STRENGTH_MIN,
        ZOOMBLUR_STRENGTH_MAX,
        ZOOMBLUR_STRENGTH_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        ZOOMBLUR_STRENGTH);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT(STR_CENTER_PARAM_NAME,
        0,
        0,
        0,
        ZOOMBLUR_CENTER);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(STR_ADAPTIVE_PARAM_NAME,
        "Adaptive",
        TRUE,
        0,
        ZOOMBLUR_ADAPTIVE);

    out_data->num_params = ZOOMBLUR_NUM_PARAMS;

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
    cl_kernel zoomblur_kernel;
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
    ShaderObjectPtr mZoomBlurShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>zoomblur_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kZoomBlurKernel_OpenCLString) };
        char const* strings[] = { k16fString, kZoomBlurKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->zoomblur_kernel = clCreateKernel(program, "ZoomBlurKernel", &result);
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
        dx_gpu_data->mZoomBlurShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"ZoomBlurKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mZoomBlurShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kZoomBlur_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> zoomblur_function = nil;
            NSString* zoomblur_name = [NSString stringWithCString : "ZoomBlurKernel" encoding : NSUTF8StringEncoding];

            zoomblur_function = [[library newFunctionWithName : zoomblur_name]autorelease];

            if (!zoomblur_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->zoomblur_pipeline = [device newComputePipelineStateWithFunction : zoomblur_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->zoomblur_kernel);

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
        dx_gpu_dataP->mZoomBlurShader.reset();

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

        [metal_dataP->zoomblur_pipeline release] ;

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
ZoomBlurFunc(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    PF_Err err = PF_Err_NONE;
    ZoomBlurInfo* info = (ZoomBlurInfo*)refcon;

    if (!info) {
        return PF_Err_BAD_CALLBACK_PARAM;
    }

    *outP = *inP;

    PF_FpLong width = (PF_FpLong)info->width;
    PF_FpLong height = (PF_FpLong)info->height;

    PF_FpLong centerXFixed = (PF_FpLong)info->center.x / 65536.0;
    PF_FpLong centerYFixed = (PF_FpLong)info->center.y / 65536.0;

    PF_FpLong centerX = 0.5 + centerXFixed / width;
    PF_FpLong centerY = 0.5 - centerYFixed / height;

    PF_FpLong uvX = (PF_FpLong)xL / width;
    PF_FpLong uvY = (PF_FpLong)yL / height;

    PF_FpLong vX = uvX - centerX;
    PF_FpLong vY = uvY - centerY;

    PF_FpLong dist = sqrt(vX * vX + vY * vY);

    PF_FpLong texelSizeX = 1.0 / width;
    PF_FpLong texelSizeY = 1.0 / height;
    PF_FpLong texelSize = MIN(texelSizeX, texelSizeY);

    PF_FpLong speed = info->strength / 2.0 / texelSize;

    if (info->adaptive) {
        speed *= dist;
    }
    else {
        PF_FpLong smoothEdge = texelSize * 5.0;
        PF_FpLong t = CLAMP(dist / smoothEdge, 0.0, 1.0);
        speed *= t * t * (3.0 - 2.0 * t);    
    }

    A_long numSamples = (A_long)CLAMP(speed, 1.01, 100.01);

    PF_FpLong normX = 0.0;
    PF_FpLong normY = 0.0;

    if (dist > 0.0) {
        normX = vX / dist;
        normY = vY / dist;
    }

    PF_FpLong aspectRatioX = 1.0;
    PF_FpLong aspectRatioY = 1.0;

    if (height > width) {
        aspectRatioX *= width / height;
    }
    else {
        aspectRatioY *= height / width;
    }

    PF_FpLong sumA = outP->alpha;      
    PF_FpLong sumR = outP->red;
    PF_FpLong sumG = outP->green;
    PF_FpLong sumB = outP->blue;
    A_long validSamples = 1;        

    for (A_long i = 1; i < numSamples; i++) {
        PF_FpLong sampleOffset = ((PF_FpLong)i / (PF_FpLong)(numSamples - 1) - 0.5);

        PF_FpLong offsetX = normX * sampleOffset * speed * texelSize * aspectRatioX;
        PF_FpLong offsetY = normY * sampleOffset * speed * texelSize * aspectRatioY;

        PF_FpLong sampleX = xL - offsetX * width;
        PF_FpLong sampleY = yL - offsetY * height;

        if (sampleX >= 0 && sampleX < width && sampleY >= 0 && sampleY < height) {
            A_long pixel_index = ((A_long)sampleY * info->rowbytes) + ((A_long)sampleX * sizeof(PixelType));

            PixelType* srcP = (PixelType*)((char*)info->src_data + pixel_index);

            sumA += srcP->alpha;
            sumR += srcP->red;
            sumG += srcP->green;
            sumB += srcP->blue;
            validSamples++;
        }
    }

    if (validSamples > 0) {
        PF_FpLong inv_validSamples = 1.0 / (PF_FpLong)validSamples;
        outP->alpha = sumA * inv_validSamples;
        outP->red = sumR * inv_validSamples;
        outP->green = sumG * inv_validSamples;
        outP->blue = sumB * inv_validSamples;
    }

    return err;
}

static PF_Err
Render(
    PF_InData* in_data,
    PF_OutData* out_dataP,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    ZoomBlurInfo info;
    AEFX_CLR_STRUCT(info);

    info.center.x = params[ZOOMBLUR_CENTER]->u.td.x_value;
    info.center.y = params[ZOOMBLUR_CENTER]->u.td.y_value;
    info.strength = params[ZOOMBLUR_STRENGTH]->u.fs_d.value;
    info.adaptive = params[ZOOMBLUR_ADAPTIVE]->u.bd.value;

    info.width = params[ZOOMBLUR_INPUT]->u.ld.width;
    info.height = params[ZOOMBLUR_INPUT]->u.ld.height;
    info.src_data = params[ZOOMBLUR_INPUT]->u.ld.data;
    info.rowbytes = params[ZOOMBLUR_INPUT]->u.ld.rowbytes;

    if (info.strength <= 0.001) {
        err = PF_COPY(&params[ZOOMBLUR_INPUT]->u.ld, output, NULL, NULL);
    }
    else {
        double bytesPerPixel = (double)info.rowbytes / (double)info.width;
        bool is16bit = false;
        bool is32bit = false;

        if (bytesPerPixel >= 16.0) {        
            is32bit = true;
        }
        else if (bytesPerPixel >= 8.0) {       
            is16bit = true;
        }

        A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

        if (is32bit) {
            ERR(suites.IterateFloatSuite1()->iterate(
                in_data,
                0,                                  
                linesL,                             
                &params[ZOOMBLUR_INPUT]->u.ld,      
                NULL,                                   
                (void*)&info,                         
                ZoomBlurFunc<PF_PixelFloat>,         
                output));
        }
        else if (is16bit) {
            ERR(suites.Iterate16Suite1()->iterate(
                in_data,
                0,                                  
                linesL,                             
                &params[ZOOMBLUR_INPUT]->u.ld,      
                NULL,                                   
                (void*)&info,                         
                ZoomBlurFunc<PF_Pixel16>,            
                output));
        }
        else {
            ERR(suites.Iterate8Suite1()->iterate(
                in_data,
                0,                                  
                linesL,                             
                &params[ZOOMBLUR_INPUT]->u.ld,      
                NULL,                                   
                (void*)&info,                         
                ZoomBlurFunc<PF_Pixel8>,             
                output));
        }
    }

    return err;
}

static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        ZoomBlurInfo* infoP = reinterpret_cast<ZoomBlurInfo*>(pre_render_dataPV);
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

    ZoomBlurInfo* infoP = reinterpret_cast<ZoomBlurInfo*>(malloc(sizeof(ZoomBlurInfo)));

    if (infoP) {
        PF_ParamDef cur_param;
        ERR(PF_CHECKOUT_PARAM(in_dataP, ZOOMBLUR_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->strength = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, ZOOMBLUR_CENTER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->center.x = cur_param.u.td.x_value;
        infoP->center.y = cur_param.u.td.y_value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, ZOOMBLUR_ADAPTIVE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->adaptive = cur_param.u.bd.value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            ZOOMBLUR_INPUT,
            ZOOMBLUR_INPUT,
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
    ZoomBlurInfo* infoP)
{
    PF_Err err = PF_Err_NONE;

    if (!err) {
        infoP->width = input_worldP->width;
        infoP->height = input_worldP->height;
        infoP->src_data = input_worldP->data;
        infoP->rowbytes = input_worldP->rowbytes;

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
                ZoomBlurFunc<PF_PixelFloat>,
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
                ZoomBlurFunc<PF_Pixel16>,
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
                ZoomBlurFunc<PF_Pixel8>,
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

static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    ZoomBlurInfo* infoP)
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

    ZoomBlurParams params;
    params.mWidth = input_worldP->width;
    params.mHeight = input_worldP->height;
    params.mSrcPitch = input_worldP->rowbytes / bytes_per_pixel;
    params.mDstPitch = output_worldP->rowbytes / bytes_per_pixel;
    params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
    params.mStrength = infoP->strength;

    PF_FpLong centerXFixed = (PF_FpLong)infoP->center.x / 65536.0;
    PF_FpLong centerYFixed = (PF_FpLong)infoP->center.y / 65536.0;
    params.mCenterX = 0.5 + centerXFixed / params.mWidth;
    params.mCenterY = 0.5 - centerYFixed / params.mHeight;
    params.mAdaptive = infoP->adaptive;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(int), &params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(int), &params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(int), &params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(int), &params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(int), &params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(float), &params.mStrength));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(float), &params.mCenterX));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(float), &params.mCenterY));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->zoomblur_kernel, param_index++, sizeof(int), &params.mAdaptive));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->zoomblur_kernel,
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
        ZoomBlur_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            params.mSrcPitch,
            params.mDstPitch,
            params.m16f,
            params.mWidth,
            params.mHeight,
            params.mStrength,
            params.mCenterX,
            params.mCenterY,
            params.mAdaptive);

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
            dx_gpu_data->mZoomBlurShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(ZoomBlurParams)));
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

        PF_Handle metal_handle = (PF_Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &params
            length : sizeof(ZoomBlurParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->zoomblur_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(params.mWidth, threadsPerGroup.width), DivideRoundUp(params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->zoomblur_pipeline] ;
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

    ZoomBlurInfo* infoP = reinterpret_cast<ZoomBlurInfo*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, ZOOMBLUR_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
            kPFWorldSuite,
            kPFWorldSuiteVersion2,
            out_data);
        PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
        ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

        if (!isGPU && infoP->strength <= 0.001) {
            ERR(PF_COPY(input_worldP, output_worldP, NULL, NULL));
        }
        else if (isGPU) {
            ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
        }
        else {
            ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
        }

        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, ZOOMBLUR_INPUT));
    }
    else {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}


static PF_Err
SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    PF_RenderRequest req = extra->input->output_request;
    req.preserve_rgb_of_zero_alpha = TRUE;

    PF_CheckoutResult checkout;
    ERR(extra->cb->checkout_layer(in_data->effect_ref,
        ZOOMBLUR_INPUT,
        ZOOMBLUR_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &checkout));

    if (!err) {
        extra->output->max_result_rect = checkout.max_result_rect;

        extra->output->result_rect = checkout.result_rect;

        extra->output->solid = checkout.solid;
        extra->output->pre_render_data = NULL;
        extra->output->flags = 0;     
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
        "Zoom Blur",  
        "DKT Zoom Blur",   
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
        switch (cmd) {
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