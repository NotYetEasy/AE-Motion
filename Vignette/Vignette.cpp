#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "Vignette.h"
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

extern void Vignette_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float scale,
    float roundness,
    float feather,
    float strength,
    float tint,
    float colorR,
    float colorG,
    float colorB,
    int punchout);

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
        "Size",
        VIGNETTE_SCALE_MIN,
        VIGNETTE_SCALE_MAX,
        VIGNETTE_SCALE_MIN,
        VIGNETTE_SCALE_MAX,
        VIGNETTE_SCALE_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        VIGNETTE_SCALE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Roundness",
        VIGNETTE_ROUNDNESS_MIN,
        VIGNETTE_ROUNDNESS_MAX,
        VIGNETTE_ROUNDNESS_MIN,
        VIGNETTE_ROUNDNESS_MAX,
        VIGNETTE_ROUNDNESS_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        VIGNETTE_ROUNDNESS);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Feather",
        VIGNETTE_FEATHER_MIN,
        VIGNETTE_FEATHER_MAX,
        VIGNETTE_FEATHER_MIN,
        VIGNETTE_FEATHER_MAX,
        VIGNETTE_FEATHER_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        VIGNETTE_FEATHER);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Strength",
        VIGNETTE_STRENGTH_MIN,
        VIGNETTE_STRENGTH_MAX,
        VIGNETTE_STRENGTH_MIN,
        VIGNETTE_STRENGTH_MAX,
        VIGNETTE_STRENGTH_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        VIGNETTE_STRENGTH);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Tint",
        VIGNETTE_TINT_MIN,
        VIGNETTE_TINT_MAX,
        VIGNETTE_TINT_MIN,
        VIGNETTE_TINT_MAX,
        VIGNETTE_TINT_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        VIGNETTE_TINT);

    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(
        "Color",
        0, 0, 0,
        VIGNETTE_COLOR);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "Punch Out",
        "Punch out center instead of darkening edges",
        FALSE,
        0,
        VIGNETTE_PUNCHOUT);

    out_data->num_params = VIGNETTE_NUM_PARAMS;

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
    cl_kernel vignette_kernel;
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
    ShaderObjectPtr mVignetteShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>vignette_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kVignetteKernel_OpenCLString) };
        char const* strings[] = { k16fString, kVignetteKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->vignette_kernel = clCreateKernel(program, "VignetteKernel", &result);
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
        dx_gpu_data->mVignetteShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"VignetteKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mVignetteShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kVignette_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> vignette_function = nil;
            NSString* vignette_name = [NSString stringWithCString : "VignetteKernel" encoding : NSUTF8StringEncoding];

            vignette_function = [[library newFunctionWithName : vignette_name]autorelease];

            if (!vignette_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->vignette_pipeline = [device newComputePipelineStateWithFunction : vignette_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->vignette_kernel);

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
        dx_gpu_dataP->mVignetteShader.reset();

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
PixelT SampleBilinear(
    PF_EffectWorld* inputP,
    float x,
    float y)
{
    x = CLAMP(x, 0.0f, static_cast<float>(inputP->width - 1));
    y = CLAMP(y, 0.0f, static_cast<float>(inputP->height - 1));

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = MIN(x0 + 1, inputP->width - 1);
    int y1 = MIN(y0 + 1, inputP->height - 1);

    float fx = x - x0;
    float fy = y - y0;

    PixelT* baseP = reinterpret_cast<PixelT*>(inputP->data);
    PixelT* p00 = baseP + y0 * inputP->rowbytes / sizeof(PixelT) + x0;
    PixelT* p01 = baseP + y0 * inputP->rowbytes / sizeof(PixelT) + x1;
    PixelT* p10 = baseP + y1 * inputP->rowbytes / sizeof(PixelT) + x0;
    PixelT* p11 = baseP + y1 * inputP->rowbytes / sizeof(PixelT) + x1;

    PixelT result;

    if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
        result.alpha = static_cast<A_u_char>((1.0f - fx) * (1.0f - fy) * p00->alpha +
            fx * (1.0f - fy) * p01->alpha +
            (1.0f - fx) * fy * p10->alpha +
            fx * fy * p11->alpha);

        result.red = static_cast<A_u_char>((1.0f - fx) * (1.0f - fy) * p00->red +
            fx * (1.0f - fy) * p01->red +
            (1.0f - fx) * fy * p10->red +
            fx * fy * p11->red);

        result.green = static_cast<A_u_char>((1.0f - fx) * (1.0f - fy) * p00->green +
            fx * (1.0f - fy) * p01->green +
            (1.0f - fx) * fy * p10->green +
            fx * fy * p11->green);

        result.blue = static_cast<A_u_char>((1.0f - fx) * (1.0f - fy) * p00->blue +
            fx * (1.0f - fy) * p01->blue +
            (1.0f - fx) * fy * p10->blue +
            fx * fy * p11->blue);
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        result.alpha = static_cast<A_u_short>((1.0f - fx) * (1.0f - fy) * p00->alpha +
            fx * (1.0f - fy) * p01->alpha +
            (1.0f - fx) * fy * p10->alpha +
            fx * fy * p11->alpha);

        result.red = static_cast<A_u_short>((1.0f - fx) * (1.0f - fy) * p00->red +
            fx * (1.0f - fy) * p01->red +
            (1.0f - fx) * fy * p10->red +
            fx * fy * p11->red);

        result.green = static_cast<A_u_short>((1.0f - fx) * (1.0f - fy) * p00->green +
            fx * (1.0f - fy) * p01->green +
            (1.0f - fx) * fy * p10->green +
            fx * fy * p11->green);

        result.blue = static_cast<A_u_short>((1.0f - fx) * (1.0f - fy) * p00->blue +
            fx * (1.0f - fy) * p01->blue +
            (1.0f - fx) * fy * p10->blue +
            fx * fy * p11->blue);
    }
    else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        result.alpha = (1.0f - fx) * (1.0f - fy) * p00->alpha +
            fx * (1.0f - fy) * p01->alpha +
            (1.0f - fx) * fy * p10->alpha +
            fx * fy * p11->alpha;

        result.red = (1.0f - fx) * (1.0f - fy) * p00->red +
            fx * (1.0f - fy) * p01->red +
            (1.0f - fx) * fy * p10->red +
            fx * fy * p11->red;

        result.green = (1.0f - fx) * (1.0f - fy) * p00->green +
            fx * (1.0f - fy) * p01->green +
            (1.0f - fx) * fy * p10->green +
            fx * fy * p11->green;

        result.blue = (1.0f - fx) * (1.0f - fy) * p00->blue +
            fx * (1.0f - fy) * p01->blue +
            (1.0f - fx) * fy * p10->blue +
            fx * fy * p11->blue;
    }

    return result;
}

float smoothstep(float edge0, float edge1, float x) {
    float t = CLAMP((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

template<typename PixelT>
static PF_Err
VignetteFunc(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelT* inP,
    PixelT* outP)
{
    PF_Err err = PF_Err_NONE;
    VignetteParams* viP = reinterpret_cast<VignetteParams*>(refcon);

    float width = static_cast<float>(viP->mWidth);
    float height = static_cast<float>(viP->mHeight);

    float normX = xL / width;
    float normY = yL / height;

    float stX = (normX - 0.5f) * 2.0f;
    float stY = (normY - 0.5f) * 2.0f;

    float scale = viP->mScale + (viP->mFeather / 4.0f);
    stX /= scale;
    stY /= scale;

    float n = viP->mRoundness + 1.0f;
    float d = powf(fabsf(stX), n) + powf(fabsf(stY), n);

    float p = smoothstep(1.0f - viP->mFeather, 1.0f, d);

    outP->alpha = inP->alpha;

    float r, g, b;
    float colorR = static_cast<float>(viP->mColor.red) / 255.0f;
    float colorG = static_cast<float>(viP->mColor.green) / 255.0f;
    float colorB = static_cast<float>(viP->mColor.blue) / 255.0f;

    if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
        r = static_cast<float>(inP->red) / 255.0f;
        g = static_cast<float>(inP->green) / 255.0f;
        b = static_cast<float>(inP->blue) / 255.0f;

        r = r * r;
        g = g * g;
        b = b * b;

        r = r * (1.0f - viP->mTint) + colorR * viP->mTint;
        g = g * (1.0f - viP->mTint) + colorG * viP->mTint;
        b = b * (1.0f - viP->mTint) + colorB * viP->mTint;

        if (viP->mPunchout) {
            outP->red = static_cast<A_u_char>(inP->red * (1.0f - p * viP->mStrength));
            outP->green = static_cast<A_u_char>(inP->green * (1.0f - p * viP->mStrength));
            outP->blue = static_cast<A_u_char>(inP->blue * (1.0f - p * viP->mStrength));
            outP->alpha = static_cast<A_u_char>(inP->alpha * (1.0f - p * viP->mStrength));
        }
        else {
            outP->red = static_cast<A_u_char>(inP->red * (1.0f - p * viP->mStrength) + r * 255.0f * (p * viP->mStrength));
            outP->green = static_cast<A_u_char>(inP->green * (1.0f - p * viP->mStrength) + g * 255.0f * (p * viP->mStrength));
            outP->blue = static_cast<A_u_char>(inP->blue * (1.0f - p * viP->mStrength) + b * 255.0f * (p * viP->mStrength));
        }
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        r = static_cast<float>(inP->red) / 32768.0f;
        g = static_cast<float>(inP->green) / 32768.0f;
        b = static_cast<float>(inP->blue) / 32768.0f;

        r = r * r;
        g = g * g;
        b = b * b;

        r = r * (1.0f - viP->mTint) + colorR * viP->mTint;
        g = g * (1.0f - viP->mTint) + colorG * viP->mTint;
        b = b * (1.0f - viP->mTint) + colorB * viP->mTint;

        if (viP->mPunchout) {
            outP->red = static_cast<A_u_short>(inP->red * (1.0f - p * viP->mStrength));
            outP->green = static_cast<A_u_short>(inP->green * (1.0f - p * viP->mStrength));
            outP->blue = static_cast<A_u_short>(inP->blue * (1.0f - p * viP->mStrength));
            outP->alpha = static_cast<A_u_short>(inP->alpha * (1.0f - p * viP->mStrength));
        }
        else {
            outP->red = static_cast<A_u_short>(inP->red * (1.0f - p * viP->mStrength) + r * 32768.0f * (p * viP->mStrength));
            outP->green = static_cast<A_u_short>(inP->green * (1.0f - p * viP->mStrength) + g * 32768.0f * (p * viP->mStrength));
            outP->blue = static_cast<A_u_short>(inP->blue * (1.0f - p * viP->mStrength) + b * 32768.0f * (p * viP->mStrength));
        }
    }
    else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        r = inP->red * inP->red;
        g = inP->green * inP->green;
        b = inP->blue * inP->blue;

        r = r * (1.0f - viP->mTint) + colorR * viP->mTint;
        g = g * (1.0f - viP->mTint) + colorG * viP->mTint;
        b = b * (1.0f - viP->mTint) + colorB * viP->mTint;

        if (viP->mPunchout) {
            outP->red = inP->red * (1.0f - p * viP->mStrength);
            outP->green = inP->green * (1.0f - p * viP->mStrength);
            outP->blue = inP->blue * (1.0f - p * viP->mStrength);
            outP->alpha = inP->alpha * (1.0f - p * viP->mStrength);
        }
        else {
            outP->red = inP->red * (1.0f - p * viP->mStrength) + r * (p * viP->mStrength);
            outP->green = inP->green * (1.0f - p * viP->mStrength) + g * (p * viP->mStrength);
            outP->blue = inP->blue * (1.0f - p * viP->mStrength) + b * (p * viP->mStrength);
        }
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

    VignetteParams viP;
    viP.mScale = params[VIGNETTE_SCALE]->u.fs_d.value;
    viP.mRoundness = params[VIGNETTE_ROUNDNESS]->u.fs_d.value;
    viP.mFeather = params[VIGNETTE_FEATHER]->u.fs_d.value;
    viP.mStrength = params[VIGNETTE_STRENGTH]->u.fs_d.value;
    viP.mTint = params[VIGNETTE_TINT]->u.fs_d.value;
    viP.mColor = params[VIGNETTE_COLOR]->u.cd.value;
    viP.mPunchout = params[VIGNETTE_PUNCHOUT]->u.bd.value;
    viP.mWidth = output->width;
    viP.mHeight = output->height;

    AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_dataP,
        kPFWorldSuite,
        kPFWorldSuiteVersion2,
        out_dataP);

    PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
    ERR(world_suite->PF_GetPixelFormat(output, &pixel_format));

    switch (pixel_format) {
    case PF_PixelFormat_ARGB128: {
        AEFX_SuiteScoper<PF_iterateFloatSuite2> iterateFloatSuite =
            AEFX_SuiteScoper<PF_iterateFloatSuite2>(in_dataP,
                kPFIterateFloatSuite,
                kPFIterateFloatSuiteVersion2,
                out_dataP);
        iterateFloatSuite->iterate(in_dataP,
            0,
            output->height,
            &params[VIGNETTE_INPUT]->u.ld,
            NULL,
            (void*)&viP,
            VignetteFunc,
            output);
        break;
    }
    case PF_PixelFormat_ARGB64: {
        AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
            AEFX_SuiteScoper<PF_iterate16Suite2>(in_dataP,
                kPFIterate16Suite,
                kPFIterate16SuiteVersion2,
                out_dataP);
        iterate16Suite->iterate(in_dataP,
            0,
            output->height,
            &params[VIGNETTE_INPUT]->u.ld,
            NULL,
            (void*)&viP,
            VignetteFunc,
            output);
        break;
    }
    case PF_PixelFormat_ARGB32: {
        AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
            AEFX_SuiteScoper<PF_Iterate8Suite2>(in_dataP,
                kPFIterate8Suite,
                kPFIterate8SuiteVersion2,
                out_dataP);
        iterate8Suite->iterate(in_dataP,
            0,
            output->height,
            &params[VIGNETTE_INPUT]->u.ld,
            NULL,
            (void*)&viP,
            VignetteFunc,
            output);
        break;
    }
    default:
        err = PF_Err_BAD_CALLBACK_PARAM;
        break;
    }

    return err;
}

static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        VignetteParams* infoP = reinterpret_cast<VignetteParams*>(pre_render_dataPV);
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

    VignetteParams* infoP = reinterpret_cast<VignetteParams*>(malloc(sizeof(VignetteParams)));

    if (infoP) {
        PF_ParamDef cur_param;
        ERR(PF_CHECKOUT_PARAM(in_dataP, VIGNETTE_SCALE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mScale = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, VIGNETTE_ROUNDNESS, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mRoundness = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, VIGNETTE_FEATHER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mFeather = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, VIGNETTE_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mStrength = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, VIGNETTE_TINT, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mTint = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, VIGNETTE_COLOR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mColor = cur_param.u.cd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, VIGNETTE_PUNCHOUT, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mPunchout = cur_param.u.bd.value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            VIGNETTE_INPUT,
            VIGNETTE_INPUT,
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
    VignetteParams* infoP)
{
    PF_Err err = PF_Err_NONE;

    infoP->mWidth = output_worldP->width;
    infoP->mHeight = output_worldP->height;

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
                VignetteFunc,
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
                VignetteFunc,
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
                VignetteFunc,
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
    float mRoundness;
    float mFeather;
    float mStrength;
    float mTint;
    float mColorR;
    float mColorG;
    float mColorB;
    int mPunchout;
} VignetteGPUParams;

static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    VignetteParams* infoP)
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

    VignetteGPUParams vignette_params;

    vignette_params.mWidth = input_worldP->width;
    vignette_params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    vignette_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    vignette_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    vignette_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    vignette_params.mScale = infoP->mScale;
    vignette_params.mRoundness = infoP->mRoundness;
    vignette_params.mFeather = infoP->mFeather;
    vignette_params.mStrength = infoP->mStrength;
    vignette_params.mTint = infoP->mTint;
    vignette_params.mColorR = static_cast<float>(infoP->mColor.red) / 255.0f;
    vignette_params.mColorG = static_cast<float>(infoP->mColor.green) / 255.0f;
    vignette_params.mColorB = static_cast<float>(infoP->mColor.blue) / 255.0f;
    vignette_params.mPunchout = infoP->mPunchout ? 1 : 0;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(int), &vignette_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(int), &vignette_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(int), &vignette_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(int), &vignette_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(int), &vignette_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(float), &vignette_params.mScale));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(float), &vignette_params.mRoundness));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(float), &vignette_params.mFeather));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(float), &vignette_params.mStrength));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(float), &vignette_params.mTint));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(float), &vignette_params.mColorR));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(float), &vignette_params.mColorG));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(float), &vignette_params.mColorB));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vignette_kernel, param_index++, sizeof(int), &vignette_params.mPunchout));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(vignette_params.mWidth, threadBlock[0]), RoundUp(vignette_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->vignette_kernel,
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

        Vignette_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            vignette_params.mSrcPitch,
            vignette_params.mDstPitch,
            vignette_params.m16f,
            vignette_params.mWidth,
            vignette_params.mHeight,
            vignette_params.mScale,
            vignette_params.mRoundness,
            vignette_params.mFeather,
            vignette_params.mStrength,
            vignette_params.mTint,
            vignette_params.mColorR,
            vignette_params.mColorG,
            vignette_params.mColorB,
            vignette_params.mPunchout);

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
            dx_gpu_data->mVignetteShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&vignette_params, sizeof(VignetteGPUParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView(
            (ID3D12Resource*)dst_mem,
            vignette_params.mHeight * dst_row_bytes));
        DX_ERR(shaderExecution.SetShaderResourceView(
            (ID3D12Resource*)src_mem,
            vignette_params.mHeight * src_row_bytes));
        DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(vignette_params.mWidth, 16), (UINT)DivideRoundUp(vignette_params.mHeight, 16)));
    }
#endif
#if HAS_METAL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        Handle metal_handle = (Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> vignette_param_buffer = [[device newBufferWithBytes : &vignette_params
            length : sizeof(VignetteGPUParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->vignette_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(vignette_params.mWidth, threadsPerGroup.width), DivideRoundUp(vignette_params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->vignette_pipeline] ;
        [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
        [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
        [computeEncoder setBuffer : vignette_param_buffer offset : 0 atIndex : 2] ;
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

    VignetteParams* infoP = reinterpret_cast<VignetteParams*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, VIGNETTE_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, VIGNETTE_INPUT));
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
        "DKT Vignette",
        "DKT Vignette",
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