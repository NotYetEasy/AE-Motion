#include "DirectionalBlurKernel.cu"

#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "DirectionalBlur.h"
#include <iostream>

// brings in M_PI on Windows
#define _USE_MATH_DEFINES
#include <math.h>

inline PF_Err CL2Err(cl_int cl_result) {
    if (cl_result == CL_SUCCESS) {
        return PF_Err_NONE;
    }
    else {
        // set a breakpoint here to pick up OpenCL errors.
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))

// CUDA kernel; see SDK_DirectionalBlur.cu.
extern void DirectionalBlur_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float strength,
    float angle);

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

    // For Premiere - declare supported pixel formats
    if (in_dataP->appl_id == 'PrMr') {

        AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
            AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_dataP,
                kPFPixelFormatSuite,
                kPFPixelFormatSuiteVersion1,
                out_data);

        //	Add the pixel formats we support in order of preference.
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
    PF_Err			err = PF_Err_NONE;
    PF_ParamDef		def;

    AEFX_CLR_STRUCT(def);

    // Strength
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        STR_STRENGTH_PARAM_NAME,
        DBLUR_STRENGTH_MIN,
        DBLUR_STRENGTH_MAX,
        DBLUR_STRENGTH_MIN,
        DBLUR_STRENGTH_MAX,
        DBLUR_STRENGTH_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        STRENGTH_DISK_ID);

    // Angle
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE(
        STR_ANGLE_PARAM_NAME,
        0,
        ANGLE_DISK_ID);

    out_data->num_params = DBLUR_NUM_PARAMS;

    return err;
}

#if HAS_METAL
PF_Err NSError2PFErr(NSError* inError)
{
    if (inError)
    {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;  //For debugging, uncomment above line and set breakpoint here
    }
    return PF_Err_NONE;
}
#endif //HAS_METAL

// GPU data initialized at GPU setup and used during render.
struct OpenCLGPUData
{
    cl_kernel directional_blur_kernel;
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
    ShaderObjectPtr mDirectionalBlurShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>directional_blur_pipeline;
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

    // Load and compile the kernel - a real plugin would cache binaries to disk

    if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
        // Nothing to do here. CUDA Kernel statically linked
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
    else if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {

        PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(OpenCLGPUData));
        OpenCLGPUData* cl_gpu_data = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_int result = CL_SUCCESS;

        char const* k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";

        size_t sizes[] = { strlen(k16fString), strlen(kDirectionalBlurKernel_OpenCLString) };
        char const* strings[] = { k16fString, kDirectionalBlurKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->directional_blur_kernel = clCreateKernel(program, "DirectionalBlurKernel", &result);
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

        // Create objects
        dx_gpu_data->mContext = std::make_shared<DXContext>();
        dx_gpu_data->mDirectionalBlurShader = std::make_shared<ShaderObject>();

        // Create the DXContext
        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"DirectionalBlurKernel", csoPath, sigPath));

        // Load the shader
        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mDirectionalBlurShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        //Create a library from source
        NSString* source = [NSString stringWithCString : kSDK_DirectionalBlur_Kernel_MetalString encoding : NSUTF8StringEncoding];
        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

        NSError* error = nil;
        id<MTLLibrary> library = [[device newLibraryWithSource : source options : nil error : &error]autorelease];

        // An error code is set for Metal compile warnings, so use nil library as the error signal
        if (!err && !library) {
            err = NSError2PFErr(error);
        }

        // For debugging only. This will contain Metal compile warnings and erorrs.
        NSString* getError = error.localizedDescription;

        PF_Handle metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
        MetalGPUData* metal_data = reinterpret_cast<MetalGPUData*>(*metal_handle);

        //Create pipeline state from function extracted from library
        if (err == PF_Err_NONE)
        {
            id<MTLFunction> directional_blur_function = nil;
            NSString* directional_blur_name = [NSString stringWithCString : "DirectionalBlurKernel" encoding : NSUTF8StringEncoding];

            directional_blur_function = [[library newFunctionWithName : directional_blur_name]autorelease];

            if (!directional_blur_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->directional_blur_pipeline = [device newComputePipelineStateWithFunction : directional_blur_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->directional_blur_kernel);

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

        // Note: DirectX: If deferred execution is implemented, a GPU sync is
        // necessary before the plugin shutdown.
        dx_gpu_dataP->mContext.reset();
        dx_gpu_dataP->mDirectionalBlurShader.reset();

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

template <typename PixelType>
static inline void SampleBilinear(
    const void* src_data,
    const A_long rowbytes,
    const A_long width,
    const A_long height,
    const float x,
    const float y,
    PixelType* outP)
{
    const int x0 = static_cast<int>(floor(x));
    const int y0 = static_cast<int>(floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float fx = x - x0;
    const float fy = y - y0;

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w10 = fx * (1.0f - fy);
    const float w11 = fx * fy;

    float r00 = 0, g00 = 0, b00 = 0, a00 = 0;
    float r01 = 0, g01 = 0, b01 = 0, a01 = 0;
    float r10 = 0, g10 = 0, b10 = 0, a10 = 0;
    float r11 = 0, g11 = 0, b11 = 0, a11 = 0;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        const PF_PixelFloat* base = static_cast<const PF_PixelFloat*>(src_data);
        const A_long stride = rowbytes / sizeof(PF_PixelFloat);

        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            const PF_PixelFloat* p00 = &base[y0 * stride + x0];
            r00 = p00->red;
            g00 = p00->green;
            b00 = p00->blue;
            a00 = p00->alpha;
        }

        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
            const PF_PixelFloat* p01 = &base[y1 * stride + x0];
            r01 = p01->red;
            g01 = p01->green;
            b01 = p01->blue;
            a01 = p01->alpha;
        }

        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
            const PF_PixelFloat* p10 = &base[y0 * stride + x1];
            r10 = p10->red;
            g10 = p10->green;
            b10 = p10->blue;
            a10 = p10->alpha;
        }

        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            const PF_PixelFloat* p11 = &base[y1 * stride + x1];
            r11 = p11->red;
            g11 = p11->green;
            b11 = p11->blue;
            a11 = p11->alpha;
        }

        outP->red = r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11;
        outP->green = g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11;
        outP->blue = b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11;
        outP->alpha = a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        const PF_Pixel16* base = static_cast<const PF_Pixel16*>(src_data);
        const A_long stride = rowbytes / sizeof(PF_Pixel16);

        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            const PF_Pixel16* p00 = &base[y0 * stride + x0];
            r00 = p00->red;
            g00 = p00->green;
            b00 = p00->blue;
            a00 = p00->alpha;
        }

        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
            const PF_Pixel16* p01 = &base[y1 * stride + x0];
            r01 = p01->red;
            g01 = p01->green;
            b01 = p01->blue;
            a01 = p01->alpha;
        }

        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
            const PF_Pixel16* p10 = &base[y0 * stride + x1];
            r10 = p10->red;
            g10 = p10->green;
            b10 = p10->blue;
            a10 = p10->alpha;
        }

        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            const PF_Pixel16* p11 = &base[y1 * stride + x1];
            r11 = p11->red;
            g11 = p11->green;
            b11 = p11->blue;
            a11 = p11->alpha;
        }

        outP->red = static_cast<A_u_short>(r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11 + 0.5f);
        outP->green = static_cast<A_u_short>(g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11 + 0.5f);
        outP->blue = static_cast<A_u_short>(b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11 + 0.5f);
        outP->alpha = static_cast<A_u_short>(a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11 + 0.5f);
    }
    else {
        const PF_Pixel8* base = static_cast<const PF_Pixel8*>(src_data);
        const A_long stride = rowbytes / sizeof(PF_Pixel8);

        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            const PF_Pixel8* p00 = &base[y0 * stride + x0];
            r00 = p00->red;
            g00 = p00->green;
            b00 = p00->blue;
            a00 = p00->alpha;
        }

        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
            const PF_Pixel8* p01 = &base[y1 * stride + x0];
            r01 = p01->red;
            g01 = p01->green;
            b01 = p01->blue;
            a01 = p01->alpha;
        }

        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
            const PF_Pixel8* p10 = &base[y0 * stride + x1];
            r10 = p10->red;
            g10 = p10->green;
            b10 = p10->blue;
            a10 = p10->alpha;
        }

        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            const PF_Pixel8* p11 = &base[y1 * stride + x1];
            r11 = p11->red;
            g11 = p11->green;
            b11 = p11->blue;
            a11 = p11->alpha;
        }

        outP->red = static_cast<A_u_char>(r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11 + 0.5f);
        outP->green = static_cast<A_u_char>(g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11 + 0.5f);
        outP->blue = static_cast<A_u_char>(b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11 + 0.5f);
        outP->alpha = static_cast<A_u_char>(a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11 + 0.5f);
    }
}

template <typename PixelType>
static PF_Err DirectionalBlurGeneric(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    PF_Err err = PF_Err_NONE;
    BlurInfo* biP = reinterpret_cast<BlurInfo*>(refcon);

    if (!biP) {
        *outP = *inP;
        return PF_Err_BAD_CALLBACK_PARAM;
    }

    const float width = static_cast<float>(biP->width);
    const float height = static_cast<float>(biP->height);
    const float strength = static_cast<float>(biP->strength);
    const float angle_rad = static_cast<float>(-biP->angle * PF_RAD_PER_DEGREE);

    if (strength <= 0.001f) {
        outP->alpha = inP->alpha;
        outP->red = inP->red;
        outP->green = inP->green;
        outP->blue = inP->blue;
        return PF_Err_NONE;
    }

    const float velocity_x = cos(angle_rad) * strength;
    const float velocity_y = -sin(angle_rad) * strength;


    const float adjusted_velocity_x = velocity_x * (width / height);

    const float texelSize_x = 1.0f / width;
    const float texelSize_y = 1.0f / height;

    const float speed_x = adjusted_velocity_x / texelSize_x;
    const float speed_y = velocity_y / texelSize_y;
    const float speed = sqrtf(speed_x * speed_x + speed_y * speed_y);

    // Limit the number of samples based on blur strength
    const int nSamples = static_cast<int>(MAX(MIN(speed, 100.01f), 1.01f));

    const float normX = static_cast<float>(xL) / width;
    const float normY = static_cast<float>(yL) / height;

    float accumR = 0.0f;
    float accumG = 0.0f;
    float accumB = 0.0f;
    float accumA = 0.0f;
    float totalWeight = 0.0f;

    // First sample at the current pixel position
    PixelType currentPixel;
    SampleBilinear<PixelType>(biP->src, biP->rowbytes, biP->width, biP->height,
        static_cast<float>(xL), static_cast<float>(yL), &currentPixel);

    accumR += static_cast<float>(currentPixel.red);
    accumG += static_cast<float>(currentPixel.green);
    accumB += static_cast<float>(currentPixel.blue);
    accumA += static_cast<float>(currentPixel.alpha);
    totalWeight += 1.0f;

 
    const float inv_nSamples_minus_1 = 1.0f / static_cast<float>(nSamples - 1);

    for (int i = 1; i < nSamples; i++) {
   
        const float t = static_cast<float>(i) * inv_nSamples_minus_1 - 0.5f;
        const float sample_norm_x = normX - adjusted_velocity_x * t;
        const float sample_norm_y = normY - velocity_y * t;

        // Check if sample is inside texture bounds
        if (sample_norm_x >= 0.0f && sample_norm_x <= 1.0f &&
            sample_norm_y >= 0.0f && sample_norm_y <= 1.0f) {

            const float sample_x = sample_norm_x * width;
            const float sample_y = sample_norm_y * height;

            PixelType sample;
            SampleBilinear<PixelType>(biP->src, biP->rowbytes, biP->width, biP->height,
                sample_x, sample_y, &sample);

            accumR += static_cast<float>(sample.red);
            accumG += static_cast<float>(sample.green);
            accumB += static_cast<float>(sample.blue);
            accumA += static_cast<float>(sample.alpha);
            totalWeight += 1.0f;
        }
    }

    // Properly handle the case where some samples were outside the texture
    const float finalWeight = (totalWeight > 0.0f) ? (1.0f / totalWeight) : 1.0f;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        outP->red = accumR * finalWeight;
        outP->green = accumG * finalWeight;
        outP->blue = accumB * finalWeight;
        outP->alpha = accumA * finalWeight;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        outP->red = static_cast<A_u_short>(accumR * finalWeight + 0.5f);
        outP->green = static_cast<A_u_short>(accumG * finalWeight + 0.5f);
        outP->blue = static_cast<A_u_short>(accumB * finalWeight + 0.5f);
        outP->alpha = static_cast<A_u_short>(accumA * finalWeight + 0.5f);
    }
    else {
        outP->red = static_cast<A_u_char>(accumR * finalWeight + 0.5f);
        outP->green = static_cast<A_u_char>(accumG * finalWeight + 0.5f);
        outP->blue = static_cast<A_u_char>(accumB * finalWeight + 0.5f);
        outP->alpha = static_cast<A_u_char>(accumA * finalWeight + 0.5f);
    }

    return err;
}




static PF_Err DirectionalBlur8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return DirectionalBlurGeneric<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err DirectionalBlur16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return DirectionalBlurGeneric<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err DirectionalBlurFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return DirectionalBlurGeneric<PF_PixelFloat>(refcon, xL, yL, inP, outP);
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

        float strength = params[DBLUR_STRENGTH]->u.fs_d.value;
        float angle = params[DBLUR_ANGLE]->u.ad.value / 65536.0f;

        // Handle Premiere Pro's VUYA format (similar to our CPU implementation)
        // This is a simplified version for Premiere - in a real plugin you would
        // implement a more sophisticated version using the directional blur algorithm
        for (int y = 0; y < output->height; ++y) {
            for (int x = 0; x < output->width; ++x) {
                ((float*)destData)[x * 4 + 0] = ((const float*)srcData)[x * 4 + 0];
                ((float*)destData)[x * 4 + 1] = ((const float*)srcData)[x * 4 + 1];
                ((float*)destData)[x * 4 + 2] = ((const float*)srcData)[x * 4 + 2];
                ((float*)destData)[x * 4 + 3] = ((const float*)srcData)[x * 4 + 3];
            }
            srcData += src->rowbytes;
            destData += dest->rowbytes;
        }
    }
    else {
        BlurInfo bi;
        AEFX_CLR_STRUCT(bi);

        PF_LayerDef* input_layer = &params[DBLUR_INPUT]->u.ld;
        A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

        bi.strength = params[DBLUR_STRENGTH]->u.fs_d.value;
        bi.angle = params[DBLUR_ANGLE]->u.ad.value / 65536.0f;
        bi.width = input_layer->width;
        bi.height = input_layer->height;
        bi.src = input_layer->data;
        bi.rowbytes = input_layer->rowbytes;

        const double bytesPerPixel = static_cast<double>(bi.rowbytes) / static_cast<double>(bi.width);

        if (bytesPerPixel >= 16.0) {
            AEFX_SuiteScoper<PF_iterateFloatSuite1> iterateFloatSuite =
                AEFX_SuiteScoper<PF_iterateFloatSuite1>(in_dataP,
                    kPFIterateFloatSuite,
                    kPFIterateFloatSuiteVersion1,
                    out_dataP);
            ERR(iterateFloatSuite->iterate(
                in_dataP, 0, linesL, input_layer, NULL, &bi, DirectionalBlurFloat, output));
        }
        else if (bytesPerPixel >= 8.0) {
            AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
                AEFX_SuiteScoper<PF_iterate16Suite2>(in_dataP,
                    kPFIterate16Suite,
                    kPFIterate16SuiteVersion2,
                    out_dataP);
            ERR(iterate16Suite->iterate(
                in_dataP, 0, linesL, input_layer, NULL, &bi, DirectionalBlur16, output));
        }
        else {
            AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
                AEFX_SuiteScoper<PF_Iterate8Suite2>(in_dataP,
                    kPFIterate8Suite,
                    kPFIterate8SuiteVersion2,
                    out_dataP);
            ERR(iterate8Suite->iterate(
                in_dataP, 0, linesL, input_layer, NULL, &bi, DirectionalBlur8, output));
        }
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

    // Signal that GPU rendering is possible
    extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

    BlurInfo* infoP = reinterpret_cast<BlurInfo*>(malloc(sizeof(BlurInfo)));

    if (infoP) {
        // Query parameters to pass from PreRender to Render with pre_render_data
        PF_ParamDef strength_param, angle_param;
        AEFX_CLR_STRUCT(strength_param);
        AEFX_CLR_STRUCT(angle_param);

        ERR(PF_CHECKOUT_PARAM(in_dataP, DBLUR_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &strength_param));
        ERR(PF_CHECKOUT_PARAM(in_dataP, DBLUR_ANGLE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &angle_param));

        infoP->strength = strength_param.u.fs_d.value;
        infoP->angle = angle_param.u.ad.value / 65536.0f;

        // Checkout the input layer first to get its dimensions
        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            DBLUR_INPUT,
            DBLUR_INPUT,
            &req,
            in_dataP->current_time,
            in_dataP->time_step,
            in_dataP->time_scale,
            &in_result));

        // Calculate expanded output area based on blur parameters
        float angle_rad = -infoP->angle * PF_RAD_PER_DEGREE;
        float velocity_x = cos(angle_rad) * infoP->strength;
        float velocity_y = -sin(angle_rad) * infoP->strength;

        float width = static_cast<float>(in_dataP->width);
        float height = static_cast<float>(in_dataP->height);
        float adjusted_velocity_x = velocity_x * (width / height);

        // Calculate how much to expand the rectangle for the blur
        float expansion_x = fabs(adjusted_velocity_x) * 100.0f;
        float expansion_y = fabs(velocity_y) * 100.0f;

        // CRITICAL: Use the input's result_rect as the base, not the output request
        // This ensures we're positioned correctly within the composition
        PF_Rect expanded_rect = in_result.result_rect;

        // Expand the rect to accommodate the blur
        expanded_rect.left -= static_cast<A_long>(expansion_x);
        expanded_rect.top -= static_cast<A_long>(expansion_y);
        expanded_rect.right += static_cast<A_long>(expansion_x);
        expanded_rect.bottom += static_cast<A_long>(expansion_y);

        // Store the pre-render data
        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        // Set the max_result_rect to our expanded rect
        extraP->output->max_result_rect = expanded_rect;

        // CRITICAL: Set the result_rect to match the input's result_rect
        // This preserves the layer's position within the composition
        extraP->output->result_rect = in_result.result_rect;

        // Tell AE we're returning extra pixels beyond what was requested
        extraP->output->flags |= PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;

        ERR(PF_CHECKIN_PARAM(in_dataP, &strength_param));
        ERR(PF_CHECKIN_PARAM(in_dataP, &angle_param));
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
    PF_PixelFormat			pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    BlurInfo* infoP)
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

            // Set up the blur info for the iterate function
            infoP->width = input_worldP->width;
            infoP->height = input_worldP->height;
            infoP->src = input_worldP->data;
            infoP->rowbytes = input_worldP->rowbytes;

            iterateFloatSuite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)infoP,
                DirectionalBlurFloat,
                output_worldP);
            break;
        }

        case PF_PixelFormat_ARGB64: {
            AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
                AEFX_SuiteScoper<PF_iterate16Suite2>(in_data,
                    kPFIterate16Suite,
                    kPFIterate16SuiteVersion2,
                    out_data);

            // Set up the blur info for the iterate function
            infoP->width = input_worldP->width;
            infoP->height = input_worldP->height;
            infoP->src = input_worldP->data;
            infoP->rowbytes = input_worldP->rowbytes;

            iterate16Suite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)infoP,
                DirectionalBlur16,
                output_worldP);
            break;
        }

        case PF_PixelFormat_ARGB32: {
            AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
                AEFX_SuiteScoper<PF_Iterate8Suite2>(in_data,
                    kPFIterate8Suite,
                    kPFIterate8SuiteVersion2,
                    out_data);

            // Set up the blur info for the iterate function
            infoP->width = input_worldP->width;
            infoP->height = input_worldP->height;
            infoP->src = input_worldP->data;
            infoP->rowbytes = input_worldP->rowbytes;

            iterate8Suite->iterate(in_data,
                0,
                output_worldP->height,
                input_worldP,
                NULL,
                (void*)infoP,
                DirectionalBlur8,
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
    float mStrength;
    float mAngle;
} DirectionalBlurParams;

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

    void* src_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

    void* dst_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

    // Read parameters
    DirectionalBlurParams blur_params;

    // Use the dimensions from the actual input world
    blur_params.mWidth = input_worldP->width;
    blur_params.mHeight = input_worldP->height;
    blur_params.mStrength = infoP->strength;
    blur_params.mAngle = infoP->angle;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    blur_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    blur_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    blur_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);


    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        // Set the arguments
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(int), &blur_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(int), &blur_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(int), &blur_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(int), &blur_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(int), &blur_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(float), &blur_params.mStrength));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->directional_blur_kernel, param_index++, sizeof(float), &blur_params.mAngle));

        // Launch the kernel
        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(blur_params.mWidth, threadBlock[0]), RoundUp(blur_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->directional_blur_kernel,
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
        DirectionalBlur_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            blur_params.mSrcPitch,
            blur_params.mDstPitch,
            blur_params.m16f,
            blur_params.mWidth,
            blur_params.mHeight,
            blur_params.mStrength,
            blur_params.mAngle);

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

        // Execute DirectionalBlur
        DXShaderExecution shaderExecution(
            dx_gpu_data->mContext,
            dx_gpu_data->mDirectionalBlurShader,
            3);

        // Note: The order of elements in the param structure should be identical to the order expected by the shader
        DX_ERR(shaderExecution.SetParamBuffer(&blur_params, sizeof(DirectionalBlurParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView(
            (ID3D12Resource*)dst_mem,
            blur_params.mHeight * dst_row_bytes));
        DX_ERR(shaderExecution.SetShaderResourceView(
            (ID3D12Resource*)src_mem,
            blur_params.mHeight * src_row_bytes));
        DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(blur_params.mWidth, 16), (UINT)DivideRoundUp(blur_params.mHeight, 16)));
    }
#endif
#if HAS_METAL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        Handle metal_handle = (Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        // Set the arguments
        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &blur_params
            length : sizeof(DirectionalBlurParams)
            options : MTLResourceStorageModeManaged]autorelease];

        // Launch the command
        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->directional_blur_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(blur_params.mWidth, threadsPerGroup.width), DivideRoundUp(blur_params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->directional_blur_pipeline] ;
        [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
        [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
        [computeEncoder setBuffer : param_buffer offset : 0 atIndex : 2] ;
        [computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;
        [computeEncoder endEncoding] ;
        [commandBuffer commit] ;

        err = NSError2PFErr([commandBuffer error]);
    }
#endif //HAS_METAL

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

    // Parameters can be queried during render. In this example, we pass them from PreRender as an example of using pre_render_data.
    BlurInfo* infoP = reinterpret_cast<BlurInfo*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, DBLUR_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        // Save original dimensions for position calculation
        infoP->width = input_worldP->width;
        infoP->height = input_worldP->height;
        infoP->src = input_worldP->data;
        infoP->rowbytes = input_worldP->rowbytes;

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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, DBLUR_INPUT));
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
        STR_NAME, // Name
        "DKT Directional Blur", // Match Name
        "DKT Effects", // Category
        AE_RESERVED_INFO, // Reserved Info
        "EffectMain",	// Entry point
        "https://www.adobe.com");	// support URL

    return result;
}

PF_Err
EffectMain(
    PF_Cmd			cmd,
    PF_InData* in_dataP,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err		err = PF_Err_NONE;

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
        // Never EVER throw exceptions into AE.
        err = thrown_err;
    }
    return err;
}
