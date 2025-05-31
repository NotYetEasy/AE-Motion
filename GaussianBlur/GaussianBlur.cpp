#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#define NOMINMAX
#include "GaussianBlur.h"
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

extern void GaussianBlur_Horizontal_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float strength);

extern void GaussianBlur_Vertical_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float strength);

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
        "Strength",
        GAUSSIANBLUR_STRENGTH_MIN,
        GAUSSIANBLUR_STRENGTH_MAX,
        GAUSSIANBLUR_STRENGTH_MIN_SLIDER,
        GAUSSIANBLUR_STRENGTH_MAX_SLIDER,
        GAUSSIANBLUR_STRENGTH_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        GAUSSIANBLUR_STRENGTH);

    out_data->num_params = GAUSSIANBLUR_NUM_PARAMS;

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
    cl_kernel horizontal_kernel;
    cl_kernel vertical_kernel;
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
    ShaderObjectPtr mHorizontalShader;
    ShaderObjectPtr mVerticalShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>horizontal_pipeline;
    id<MTLComputePipelineState>vertical_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kGaussianBlurKernel_OpenCLString) };
        char const* strings[] = { k16fString, kGaussianBlurKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->horizontal_kernel = clCreateKernel(program, "GaussianBlurHorizontalKernel", &result);
            CL_ERR(result);
        }

        if (!err) {
            cl_gpu_data->vertical_kernel = clCreateKernel(program, "GaussianBlurVerticalKernel", &result);
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
        dx_gpu_data->mHorizontalShader = std::make_shared<ShaderObject>();
        dx_gpu_data->mVerticalShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"GaussianBlurHorizontalKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mHorizontalShader));

        DX_ERR(GetShaderPath(L"GaussianBlurVerticalKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mVerticalShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kGaussianBlur_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> horizontal_function = nil;
            id<MTLFunction> vertical_function = nil;
            NSString* horizontal_name = [NSString stringWithCString : "GaussianBlurHorizontalKernel" encoding : NSUTF8StringEncoding];
            NSString* vertical_name = [NSString stringWithCString : "GaussianBlurVerticalKernel" encoding : NSUTF8StringEncoding];

            horizontal_function = [[library newFunctionWithName : horizontal_name]autorelease];
            vertical_function = [[library newFunctionWithName : vertical_name]autorelease];

            if (!horizontal_function || !vertical_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->horizontal_pipeline = [device newComputePipelineStateWithFunction : horizontal_function error : &error];
                err = NSError2PFErr(error);
            }

            if (!err) {
                metal_data->vertical_pipeline = [device newComputePipelineStateWithFunction : vertical_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->horizontal_kernel);
        (void)clReleaseKernel(cl_gpu_dataP->vertical_kernel);

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
        dx_gpu_dataP->mHorizontalShader.reset();
        dx_gpu_dataP->mVerticalShader.reset();

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
static PixelT SampleBilinear(PF_LayerDef* world, float x, float y) {
    float clampedX = std::max(0.0f, std::min(static_cast<float>(world->width - 1), x));
    float clampedY = std::max(0.0f, std::min(static_cast<float>(world->height - 1), y));

    int x0 = static_cast<int>(clampedX);
    int y0 = static_cast<int>(clampedY);
    int x1 = std::min(x0 + 1, world->width - 1);
    int y1 = std::min(y0 + 1, world->height - 1);

    float fracX = clampedX - x0;
    float fracY = clampedY - y0;

    if (x0 == world->width - 1) {
        x1 = x0;
        fracX = 0.0f;
    }

    if (y0 == world->height - 1) {
        y1 = y0;
        fracY = 0.0f;
    }

    PixelT* p00 = nullptr;
    PixelT* p01 = nullptr;
    PixelT* p10 = nullptr;
    PixelT* p11 = nullptr;

    A_long rowStride = 0;

    if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
        rowStride = world->rowbytes / sizeof(PF_Pixel8);
        PF_Pixel8* baseAddr = reinterpret_cast<PF_Pixel8*>(world->data);
        p00 = reinterpret_cast<PixelT*>(baseAddr + y0 * rowStride + x0);
        p01 = reinterpret_cast<PixelT*>(baseAddr + y0 * rowStride + x1);
        p10 = reinterpret_cast<PixelT*>(baseAddr + y1 * rowStride + x0);
        p11 = reinterpret_cast<PixelT*>(baseAddr + y1 * rowStride + x1);
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        rowStride = world->rowbytes / sizeof(PF_Pixel16);
        PF_Pixel16* baseAddr = reinterpret_cast<PF_Pixel16*>(world->data);
        p00 = reinterpret_cast<PixelT*>(baseAddr + y0 * rowStride + x0);
        p01 = reinterpret_cast<PixelT*>(baseAddr + y0 * rowStride + x1);
        p10 = reinterpret_cast<PixelT*>(baseAddr + y1 * rowStride + x0);
        p11 = reinterpret_cast<PixelT*>(baseAddr + y1 * rowStride + x1);
    }
    else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        rowStride = world->rowbytes / sizeof(PF_PixelFloat);
        PF_PixelFloat* baseAddr = reinterpret_cast<PF_PixelFloat*>(world->data);
        p00 = reinterpret_cast<PixelT*>(baseAddr + y0 * rowStride + x0);
        p01 = reinterpret_cast<PixelT*>(baseAddr + y0 * rowStride + x1);
        p10 = reinterpret_cast<PixelT*>(baseAddr + y1 * rowStride + x0);
        p11 = reinterpret_cast<PixelT*>(baseAddr + y1 * rowStride + x1);
    }

    if (!p00 || !p01 || !p10 || !p11) {
        PixelT result;
        if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
            result.alpha = 1.0f;
            result.red = result.green = result.blue = 0.0f;
        }
        else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
            result.alpha = PF_MAX_CHAN16;
            result.red = result.green = result.blue = 0;
        }
        else {
            result.alpha = PF_MAX_CHAN8;
            result.red = result.green = result.blue = 0;
        }
        return result;
    }

    PixelT result;

    if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        float w00 = (1.0f - fracX) * (1.0f - fracY);
        float w01 = fracX * (1.0f - fracY);
        float w10 = (1.0f - fracX) * fracY;
        float w11 = fracX * fracY;

        result.alpha = w00 * p00->alpha + w01 * p01->alpha + w10 * p10->alpha + w11 * p11->alpha;
        result.red = w00 * p00->red + w01 * p01->red + w10 * p10->red + w11 * p11->red;
        result.green = w00 * p00->green + w01 * p01->green + w10 * p10->green + w11 * p11->green;
        result.blue = w00 * p00->blue + w01 * p01->blue + w10 * p10->blue + w11 * p11->blue;
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        float w00 = (1.0f - fracX) * (1.0f - fracY);
        float w01 = fracX * (1.0f - fracY);
        float w10 = (1.0f - fracX) * fracY;
        float w11 = fracX * fracY;

        result.alpha = static_cast<A_u_short>(w00 * p00->alpha + w01 * p01->alpha + w10 * p10->alpha + w11 * p11->alpha + 0.5f);
        result.red = static_cast<A_u_short>(w00 * p00->red + w01 * p01->red + w10 * p10->red + w11 * p11->red + 0.5f);
        result.green = static_cast<A_u_short>(w00 * p00->green + w01 * p01->green + w10 * p10->green + w11 * p11->green + 0.5f);
        result.blue = static_cast<A_u_short>(w00 * p00->blue + w01 * p01->blue + w10 * p10->blue + w11 * p11->blue + 0.5f);
    }
    else {
        float w00 = (1.0f - fracX) * (1.0f - fracY);
        float w01 = fracX * (1.0f - fracY);
        float w10 = (1.0f - fracX) * fracY;
        float w11 = fracX * fracY;

        result.alpha = static_cast<A_u_char>(w00 * p00->alpha + w01 * p01->alpha + w10 * p10->alpha + w11 * p11->alpha + 0.5f);
        result.red = static_cast<A_u_char>(w00 * p00->red + w01 * p01->red + w10 * p10->red + w11 * p11->red + 0.5f);
        result.green = static_cast<A_u_char>(w00 * p00->green + w01 * p01->green + w10 * p10->green + w11 * p11->green + 0.5f);
        result.blue = static_cast<A_u_char>(w00 * p00->blue + w01 * p01->blue + w10 * p10->blue + w11 * p11->blue + 0.5f);
    }

    return result;
}

template<typename PixelT>
static PF_Err
HorizontalBlurPass(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelT* inP,
    PixelT* outP)
{
    PF_Err err = PF_Err_NONE;
    BlurInfo* biP = reinterpret_cast<BlurInfo*>(refcon);

    if (!biP) return PF_Err_BAD_CALLBACK_PARAM;

    if (biP->strengthF <= 0.0001) {
        outP->alpha = inP->alpha;
        outP->red = inP->red;
        outP->green = inP->green;
        outP->blue = inP->blue;
        return err;
    }

    float previewSize = sqrtf(static_cast<float>(biP->comp_width * biP->comp_width +
        biP->comp_height * biP->comp_height));
    float downsampleFactor = (float)biP->downsample_x.num / (float)biP->downsample_x.den;
    float kernelSize = (biP->strengthF / 2.0f * 1.14f / 4.0f / 3.0f * previewSize) * downsampleFactor;

    float numBlurPixelsPerSide = std::max(1.0f, kernelSize);

    const float PI = 3.14159265f;
    float adjSigma = numBlurPixelsPerSide / 2.14596602f;

    float incrementalGaussianX = 1.0f / (sqrtf(2.0f * PI) * adjSigma);
    float incrementalGaussianY = expf(-0.5f / (adjSigma * adjSigma));
    float incrementalGaussianZ = incrementalGaussianY * incrementalGaussianY;

    float avgAlpha = 0.0f;
    float avgRed = 0.0f;
    float avgGreen = 0.0f;
    float avgBlue = 0.0f;
    float coefficientSum = 0.0f;

    avgAlpha += inP->alpha * incrementalGaussianX;
    avgRed += inP->red * incrementalGaussianX;
    avgGreen += inP->green * incrementalGaussianX;
    avgBlue += inP->blue * incrementalGaussianX;
    coefficientSum += incrementalGaussianX;

    for (float i = 1.0f; i <= numBlurPixelsPerSide; i += 2.0f) {
        float offset0 = i;
        float offset1 = i + 1.0f;

        incrementalGaussianX *= incrementalGaussianY;
        incrementalGaussianY *= incrementalGaussianZ;
        float weight0 = incrementalGaussianX;
        coefficientSum += (2.0f * weight0);

        incrementalGaussianX *= incrementalGaussianY;
        incrementalGaussianY *= incrementalGaussianZ;
        float weight1 = incrementalGaussianX;
        coefficientSum += (2.0f * weight1);

        float weightL = weight0 + weight1;
        float offsetL = (offset0 * weight0 + offset1 * weight1) / weightL;

        float leftX = xL - offsetL;
        float rightX = xL + offsetL;

        PixelT leftPix = SampleBilinear<PixelT>(biP->inputImg, leftX, static_cast<float>(yL));
        PixelT rightPix = SampleBilinear<PixelT>(biP->inputImg, rightX, static_cast<float>(yL));

        avgAlpha += (leftPix.alpha * weightL + rightPix.alpha * weightL);
        avgRed += (leftPix.red * weightL + rightPix.red * weightL);
        avgGreen += (leftPix.green * weightL + rightPix.green * weightL);
        avgBlue += (leftPix.blue * weightL + rightPix.blue * weightL);
    }

    if (coefficientSum > 0) {
        if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
            outP->alpha = avgAlpha / coefficientSum;
            outP->red = avgRed / coefficientSum;
            outP->green = avgGreen / coefficientSum;
            outP->blue = avgBlue / coefficientSum;
        }
        else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
            outP->alpha = static_cast<A_u_short>(avgAlpha / coefficientSum + 0.5f);
            outP->red = static_cast<A_u_short>(avgRed / coefficientSum + 0.5f);
            outP->green = static_cast<A_u_short>(avgGreen / coefficientSum + 0.5f);
            outP->blue = static_cast<A_u_short>(avgBlue / coefficientSum + 0.5f);
        }
        else {
            outP->alpha = static_cast<A_u_char>(avgAlpha / coefficientSum + 0.5f);
            outP->red = static_cast<A_u_char>(avgRed / coefficientSum + 0.5f);
            outP->green = static_cast<A_u_char>(avgGreen / coefficientSum + 0.5f);
            outP->blue = static_cast<A_u_char>(avgBlue / coefficientSum + 0.5f);
        }
    }
    else {
        outP->alpha = inP->alpha;
        outP->red = inP->red;
        outP->green = inP->green;
        outP->blue = inP->blue;
    }

    return err;
}

template<typename PixelT>
static PF_Err
VerticalBlurPass(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelT* inP,
    PixelT* outP)
{
    PF_Err err = PF_Err_NONE;
    BlurInfo* biP = reinterpret_cast<BlurInfo*>(refcon);

    if (!biP) return PF_Err_BAD_CALLBACK_PARAM;

    if (biP->strengthF <= 0.0001) {
        outP->alpha = inP->alpha;
        outP->red = inP->red;
        outP->green = inP->green;
        outP->blue = inP->blue;
        return err;
    }

    float previewSize = sqrtf(static_cast<float>(biP->comp_width * biP->comp_width +
        biP->comp_height * biP->comp_height));
    float downsampleFactor = (float)biP->downsample_x.num / (float)biP->downsample_x.den;
    float kernelSize = (biP->strengthF / 2.0f * 1.14f / 4.0f / 3.0f * previewSize) * downsampleFactor;

    float numBlurPixelsPerSide = std::max(1.0f, kernelSize);

    const float PI = 3.14159265f;
    float adjSigma = numBlurPixelsPerSide / 2.14596602f;

    float incrementalGaussianX = 1.0f / (sqrtf(2.0f * PI) * adjSigma);
    float incrementalGaussianY = expf(-0.5f / (adjSigma * adjSigma));
    float incrementalGaussianZ = incrementalGaussianY * incrementalGaussianY;

    float avgAlpha = 0.0f;
    float avgRed = 0.0f;
    float avgGreen = 0.0f;
    float avgBlue = 0.0f;
    float coefficientSum = 0.0f;

    avgAlpha += inP->alpha * incrementalGaussianX;
    avgRed += inP->red * incrementalGaussianX;
    avgGreen += inP->green * incrementalGaussianX;
    avgBlue += inP->blue * incrementalGaussianX;
    coefficientSum += incrementalGaussianX;

    for (float i = 1.0f; i <= numBlurPixelsPerSide; i += 2.0f) {
        float offset0 = i;
        float offset1 = i + 1.0f;

        incrementalGaussianX *= incrementalGaussianY;
        incrementalGaussianY *= incrementalGaussianZ;
        float weight0 = incrementalGaussianX;
        coefficientSum += (2.0f * weight0);

        incrementalGaussianX *= incrementalGaussianY;
        incrementalGaussianY *= incrementalGaussianZ;
        float weight1 = incrementalGaussianX;
        coefficientSum += (2.0f * weight1);

        float weightL = weight0 + weight1;
        float offsetL = (offset0 * weight0 + offset1 * weight1) / weightL;

        float topY = yL - offsetL;
        float bottomY = yL + offsetL;

        PixelT topPix = SampleBilinear<PixelT>(biP->hblurImg, static_cast<float>(xL), topY);
        PixelT bottomPix = SampleBilinear<PixelT>(biP->hblurImg, static_cast<float>(xL), bottomY);

        avgAlpha += (topPix.alpha * weightL + bottomPix.alpha * weightL);
        avgRed += (topPix.red * weightL + bottomPix.red * weightL);
        avgGreen += (topPix.green * weightL + bottomPix.green * weightL);
        avgBlue += (topPix.blue * weightL + bottomPix.blue * weightL);
    }

    if (coefficientSum > 0) {
        if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
            outP->alpha = avgAlpha / coefficientSum;
            outP->red = avgRed / coefficientSum;
            outP->green = avgGreen / coefficientSum;
            outP->blue = avgBlue / coefficientSum;
        }
        else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
            outP->alpha = static_cast<A_u_short>(avgAlpha / coefficientSum + 0.5f);
            outP->red = static_cast<A_u_short>(avgRed / coefficientSum + 0.5f);
            outP->green = static_cast<A_u_short>(avgGreen / coefficientSum + 0.5f);
            outP->blue = static_cast<A_u_short>(avgBlue / coefficientSum + 0.5f);
        }
        else {
            outP->alpha = static_cast<A_u_char>(avgAlpha / coefficientSum + 0.5f);
            outP->red = static_cast<A_u_char>(avgRed / coefficientSum + 0.5f);
            outP->green = static_cast<A_u_char>(avgGreen / coefficientSum + 0.5f);
            outP->blue = static_cast<A_u_char>(avgBlue / coefficientSum + 0.5f);
        }
    }
    else {
        outP->alpha = inP->alpha;
        outP->red = inP->red;
        outP->green = inP->green;
        outP->blue = inP->blue;
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
    AEGP_SuiteHandler suites(in_dataP->pica_basicP);

    BlurInfo bi;
    AEFX_CLR_STRUCT(bi);

    bi.strengthF = params[GAUSSIANBLUR_STRENGTH]->u.fs_d.value;
    bi.inputImg = &params[GAUSSIANBLUR_INPUT]->u.ld;
    bi.comp_width = in_dataP->width;
    bi.comp_height = in_dataP->height;
    bi.downsample_x = in_dataP->downsample_x;
    bi.downsample_y = in_dataP->downsample_y;

    PF_EffectWorld hblurImg;
    ERR(suites.WorldSuite1()->new_world(
        in_dataP->effect_ref,
        output->width,
        output->height,
        PF_NewWorldFlag_NONE,
        &hblurImg));

    bi.hblurImg = &hblurImg;

    if (!err) {
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
                hblurImg.height,
                &params[GAUSSIANBLUR_INPUT]->u.ld,
                NULL,
                (void*)&bi,
                (PF_IteratePixelFloatFunc)HorizontalBlurPass<PF_PixelFloat>,
                &hblurImg));
            break;

        case PF_PixelFormat_ARGB64:
            ERR(suites.Iterate16Suite1()->iterate(
                in_dataP,
                0,
                hblurImg.height,
                &params[GAUSSIANBLUR_INPUT]->u.ld,
                NULL,
                (void*)&bi,
                HorizontalBlurPass<PF_Pixel16>,
                &hblurImg));
            break;

        case PF_PixelFormat_ARGB32:
        default:
            ERR(suites.Iterate8Suite1()->iterate(
                in_dataP,
                0,
                hblurImg.height,
                &params[GAUSSIANBLUR_INPUT]->u.ld,
                NULL,
                (void*)&bi,
                HorizontalBlurPass<PF_Pixel8>,
                &hblurImg));
            break;
        }

        PF_WorldTransformSuite1* transform_suite = NULL;
        ERR(suites.Pica()->AcquireSuite(kPFWorldTransformSuite,
            kPFWorldTransformSuiteVersion1,
            (const void**)&transform_suite));

        if (!err && transform_suite) {
            ERR(transform_suite->copy(
                in_dataP->effect_ref,
                &hblurImg,
                output,
                NULL,
                NULL
            ));

            suites.Pica()->ReleaseSuite(kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1);
        }

        ERR(suites.WorldSuite1()->dispose_world(in_dataP->effect_ref, &hblurImg));
    }

    return err;
}


static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        BlurInfo* infoP = reinterpret_cast<BlurInfo*>(pre_render_dataPV);
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

    BlurInfo* infoP = reinterpret_cast<BlurInfo*>(malloc(sizeof(BlurInfo)));

    if (infoP) {
        PF_ParamDef cur_param;
        ERR(PF_CHECKOUT_PARAM(in_dataP, GAUSSIANBLUR_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->strengthF = cur_param.u.fs_d.value;
        infoP->mStrength = cur_param.u.fs_d.value;          
        infoP->comp_width = in_dataP->width;
        infoP->comp_height = in_dataP->height;
        infoP->downsample_x = in_dataP->downsample_x;
        infoP->downsample_y = in_dataP->downsample_y;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            GAUSSIANBLUR_INPUT,
            GAUSSIANBLUR_INPUT,
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
    BlurInfo* infoP)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    PF_EffectWorld downImg;
    PF_EffectWorld hblurImg;

    if (infoP->strengthF <= 0.0001) {
        PF_WorldTransformSuite1* transform_suite = NULL;
        ERR(suites.Pica()->AcquireSuite(kPFWorldTransformSuite,
            kPFWorldTransformSuiteVersion1,
            (const void**)&transform_suite));

        if (!err && transform_suite) {
            ERR(transform_suite->copy(
                in_data->effect_ref,
                input_worldP,
                output_worldP,
                NULL,
                NULL
            ));

            suites.Pica()->ReleaseSuite(kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1);
        }
    }
    else {
        PF_WorldSuite2* wsP = NULL;
        ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));

        A_long downsample_width = input_worldP->width / 2;
        A_long downsample_height = input_worldP->height / 2;

        ERR(wsP->PF_NewWorld(
            in_data->effect_ref,
            downsample_width,
            downsample_height,
            true,
            pixel_format,
            &downImg));

        ERR(wsP->PF_NewWorld(
            in_data->effect_ref,
            downsample_width,
            downsample_height,
            true,
            pixel_format,
            &hblurImg));

        ERR(suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2));

        if (!err) {
            PF_WorldTransformSuite1* transform_suite = NULL;
            ERR(suites.Pica()->AcquireSuite(kPFWorldTransformSuite,
                kPFWorldTransformSuiteVersion1,
                (const void**)&transform_suite));

            if (!err && transform_suite) {
                ERR(transform_suite->copy_hq(
                    in_data->effect_ref,
                    input_worldP,
                    &downImg,
                    NULL,
                    NULL
                ));

                suites.Pica()->ReleaseSuite(kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1);
            }

            infoP->inputImg = &downImg;
            infoP->hblurImg = &hblurImg;

            switch (pixel_format) {
            case PF_PixelFormat_ARGB128:
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,
                    downImg.height,
                    &downImg,
                    NULL,
                    (void*)infoP,
                    (PF_IteratePixelFloatFunc)HorizontalBlurPass<PF_PixelFloat>,
                    &hblurImg));
                break;

            case PF_PixelFormat_ARGB64:
                ERR(suites.Iterate16Suite1()->iterate(
                    in_data,
                    0,
                    downImg.height,
                    &downImg,
                    NULL,
                    (void*)infoP,
                    HorizontalBlurPass<PF_Pixel16>,
                    &hblurImg));
                break;

            case PF_PixelFormat_ARGB32:
            default:
                ERR(suites.Iterate8Suite1()->iterate(
                    in_data,
                    0,
                    downImg.height,
                    &downImg,
                    NULL,
                    (void*)infoP,
                    HorizontalBlurPass<PF_Pixel8>,
                    &hblurImg));
                break;
            }

            PF_EffectWorld vblurImg;

            ERR(wsP->PF_NewWorld(
                in_data->effect_ref,
                downsample_width,
                downsample_height,
                true,
                pixel_format,
                &vblurImg));

            infoP->hblurImg = &hblurImg;

            switch (pixel_format) {
            case PF_PixelFormat_ARGB128:
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,
                    hblurImg.height,
                    &hblurImg,
                    NULL,
                    (void*)infoP,
                    (PF_IteratePixelFloatFunc)VerticalBlurPass<PF_PixelFloat>,
                    &vblurImg));
                break;

            case PF_PixelFormat_ARGB64:
                ERR(suites.Iterate16Suite1()->iterate(
                    in_data,
                    0,
                    hblurImg.height,
                    &hblurImg,
                    NULL,
                    (void*)infoP,
                    VerticalBlurPass<PF_Pixel16>,
                    &vblurImg));
                break;

            case PF_PixelFormat_ARGB32:
            default:
                ERR(suites.Iterate8Suite1()->iterate(
                    in_data,
                    0,
                    hblurImg.height,
                    &hblurImg,
                    NULL,
                    (void*)infoP,
                    VerticalBlurPass<PF_Pixel8>,
                    &vblurImg));
                break;
            }

            ERR(suites.Pica()->AcquireSuite(kPFWorldTransformSuite,
                kPFWorldTransformSuiteVersion1,
                (const void**)&transform_suite));

            if (!err && transform_suite) {
                ERR(transform_suite->copy_hq(
                    in_data->effect_ref,
                    &vblurImg,
                    output_worldP,
                    NULL,
                    NULL
                ));

                suites.Pica()->ReleaseSuite(kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1);
            }

            ERR(suites.WorldSuite1()->dispose_world(in_data->effect_ref, &vblurImg));
            ERR(suites.WorldSuite1()->dispose_world(in_data->effect_ref, &hblurImg));
            ERR(suites.WorldSuite1()->dispose_world(in_data->effect_ref, &downImg));
        }
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
    float mStrength;
} GaussianBlurParams;

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
    BlurInfo* infoP)
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

    PF_EffectWorld* intermediate_buffer;

    ERR(gpu_suite->CreateGPUWorld(in_dataP->effect_ref,
        extraP->input->device_index,
        input_worldP->width,
        input_worldP->height,
        input_worldP->pix_aspect_ratio,
        in_dataP->field,
        pixel_format,
        false,
        &intermediate_buffer));

    void* src_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

    void* dst_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

    void* im_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, intermediate_buffer, &im_mem));

    GaussianBlurParams horizontal_params;
    GaussianBlurParams vertical_params;

    horizontal_params.mWidth = vertical_params.mWidth = input_worldP->width;
    horizontal_params.mHeight = vertical_params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long tmp_row_bytes = intermediate_buffer->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    horizontal_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    horizontal_params.mDstPitch = tmp_row_bytes / bytes_per_pixel;
    horizontal_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
    horizontal_params.mStrength = infoP->mStrength;

    vertical_params.mSrcPitch = tmp_row_bytes / bytes_per_pixel;
    vertical_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    vertical_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
    vertical_params.mStrength = infoP->mStrength;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_im_mem = (cl_mem)im_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint horizontal_param_index = 0;
        cl_uint vertical_param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->horizontal_kernel, horizontal_param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->horizontal_kernel, horizontal_param_index++, sizeof(cl_mem), &cl_im_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->horizontal_kernel, horizontal_param_index++, sizeof(int), &horizontal_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->horizontal_kernel, horizontal_param_index++, sizeof(int), &horizontal_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->horizontal_kernel, horizontal_param_index++, sizeof(int), &horizontal_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->horizontal_kernel, horizontal_param_index++, sizeof(int), &horizontal_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->horizontal_kernel, horizontal_param_index++, sizeof(int), &horizontal_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->horizontal_kernel, horizontal_param_index++, sizeof(float), &horizontal_params.mStrength));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(horizontal_params.mWidth, threadBlock[0]), RoundUp(horizontal_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->horizontal_kernel,
            2,
            0,
            grid,
            threadBlock,
            0,
            0,
            0));

        CL_ERR(clSetKernelArg(cl_gpu_dataP->vertical_kernel, vertical_param_index++, sizeof(cl_mem), &cl_im_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vertical_kernel, vertical_param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vertical_kernel, vertical_param_index++, sizeof(int), &vertical_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vertical_kernel, vertical_param_index++, sizeof(int), &vertical_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vertical_kernel, vertical_param_index++, sizeof(int), &vertical_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vertical_kernel, vertical_param_index++, sizeof(int), &vertical_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vertical_kernel, vertical_param_index++, sizeof(int), &vertical_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->vertical_kernel, vertical_param_index++, sizeof(float), &vertical_params.mStrength));

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->vertical_kernel,
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

        GaussianBlur_Horizontal_CUDA(
            (const float*)src_mem,
            (float*)im_mem,
            horizontal_params.mSrcPitch,
            horizontal_params.mDstPitch,
            horizontal_params.m16f,
            horizontal_params.mWidth,
            horizontal_params.mHeight,
            horizontal_params.mStrength);

        if (cudaPeekAtLastError() != cudaSuccess) {
            err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }

        GaussianBlur_Vertical_CUDA(
            (const float*)im_mem,
            (float*)dst_mem,
            vertical_params.mSrcPitch,
            vertical_params.mDstPitch,
            vertical_params.m16f,
            vertical_params.mWidth,
            vertical_params.mHeight,
            vertical_params.mStrength);

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
                dx_gpu_data->mHorizontalShader,
                3);

            DX_ERR(shaderExecution.SetParamBuffer(&horizontal_params, sizeof(GaussianBlurParams)));
            DX_ERR(shaderExecution.SetUnorderedAccessView(
                (ID3D12Resource*)im_mem,
                horizontal_params.mHeight * tmp_row_bytes));
            DX_ERR(shaderExecution.SetShaderResourceView(
                (ID3D12Resource*)src_mem,
                horizontal_params.mHeight * src_row_bytes));
            DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(horizontal_params.mWidth, 16), (UINT)DivideRoundUp(horizontal_params.mHeight, 16)));
        }

        if (!err)
        {
            DXShaderExecution shaderExecution(
                dx_gpu_data->mContext,
                dx_gpu_data->mVerticalShader,
                3);

            DX_ERR(shaderExecution.SetParamBuffer(&vertical_params, sizeof(GaussianBlurParams)));
            DX_ERR(shaderExecution.SetUnorderedAccessView(
                (ID3D12Resource*)dst_mem,
                vertical_params.mHeight * dst_row_bytes));
            DX_ERR(shaderExecution.SetShaderResourceView(
                (ID3D12Resource*)im_mem,
                vertical_params.mHeight * tmp_row_bytes));
            DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(vertical_params.mWidth, 16), (UINT)DivideRoundUp(vertical_params.mHeight, 16)));
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
        id<MTLBuffer> horizontal_param_buffer = [[device newBufferWithBytes : &horizontal_params
            length : sizeof(GaussianBlurParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLBuffer> vertical_param_buffer = [[device newBufferWithBytes : &vertical_params
            length : sizeof(GaussianBlurParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> im_metal_buffer = (id<MTLBuffer>)im_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup1 = { [metal_dataP->horizontal_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups1 = { DivideRoundUp(horizontal_params.mWidth, threadsPerGroup1.width), DivideRoundUp(horizontal_params.mHeight, threadsPerGroup1.height), 1 };

        MTLSize threadsPerGroup2 = { [metal_dataP->vertical_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups2 = { DivideRoundUp(vertical_params.mWidth, threadsPerGroup2.width), DivideRoundUp(vertical_params.mHeight, threadsPerGroup2.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->horizontal_pipeline] ;
        [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
        [computeEncoder setBuffer : im_metal_buffer offset : 0 atIndex : 1] ;
        [computeEncoder setBuffer : horizontal_param_buffer offset : 0 atIndex : 2] ;
        [computeEncoder dispatchThreadgroups : numThreadgroups1 threadsPerThreadgroup : threadsPerGroup1] ;

        err = NSError2PFErr([commandBuffer error]);

        if (!err) {
            [computeEncoder setComputePipelineState : metal_dataP->vertical_pipeline] ;
            [computeEncoder setBuffer : im_metal_buffer offset : 0 atIndex : 0] ;
            [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
            [computeEncoder setBuffer : vertical_param_buffer offset : 0 atIndex : 2] ;
            [computeEncoder dispatchThreadgroups : numThreadgroups2 threadsPerThreadgroup : threadsPerGroup2] ;
            [computeEncoder endEncoding] ;
            [commandBuffer commit] ;

            err = NSError2PFErr([commandBuffer error]);
        }
    }
#endif 

    ERR(gpu_suite->DisposeGPUWorld(in_dataP->effect_ref, intermediate_buffer));
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

    BlurInfo* infoP = reinterpret_cast<BlurInfo*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, GAUSSIANBLUR_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, GAUSSIANBLUR_INPUT));
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
        "DKT Gaussian Blur",
        "DKT Gaussian Blur",
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