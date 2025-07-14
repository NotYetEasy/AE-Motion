#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "FractalWarp.h"
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


extern void FractalWarp_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float positionX,
    float positionY,
    float parallaxX,
    float parallaxY,
    float magnitude,
    float detail,
    float lacunarity,
    int screenSpace,
    int octaves,
    int x_tiles,
    int y_tiles,
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
    PF_Err    err = PF_Err_NONE;

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
    PF_Err            err = PF_Err_NONE;
    PF_ParamDef        def;

    AEFX_CLR_STRUCT(def);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT(
        "Position",
        0, 0,
        0,
        POSITION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POINT(
        "Parallax",
        0, 0,
        0,
        PARALLAX_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Magnitude",
        FRACTALWARP_MAGNITUDE_MIN,
        FRACTALWARP_MAGNITUDE_MAX,
        -0.05,
        0.05,
        FRACTALWARP_MAGNITUDE_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        MAGNITUDE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Detail",
        FRACTALWARP_DETAIL_MIN,
        FRACTALWARP_DETAIL_MAX,
        FRACTALWARP_DETAIL_MIN,
        FRACTALWARP_DETAIL_MAX,
        FRACTALWARP_DETAIL_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        DETAIL_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(
        "Lacunarity",
        FRACTALWARP_LACUNARITY_MIN,
        FRACTALWARP_LACUNARITY_MAX,
        FRACTALWARP_LACUNARITY_MIN,
        FRACTALWARP_LACUNARITY_MAX,
        FRACTALWARP_LACUNARITY_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        LACUNARITY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "Screen Space",
        "",
        FALSE,
        0,
        SCREENSPACE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER(
        "Octaves",
        FRACTALWARP_OCTAVES_MIN,
        FRACTALWARP_OCTAVES_MAX,
        FRACTALWARP_OCTAVES_MIN,
        FRACTALWARP_OCTAVES_MAX,
        FRACTALWARP_OCTAVES_DFLT,
        OCTAVES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_START;
    PF_STRCPY(def.name, "Tiles");
    def.flags = PF_ParamFlag_START_COLLAPSED;
    PF_ADD_PARAM(in_data, -1, &def);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "X Tiles",
        "",
        FALSE,
        0,
        X_TILES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "Y Tiles",
        "",
        FALSE,
        0,
        Y_TILES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX(
        "Mirror",
        "",
        FALSE,
        0,
        MIRROR_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    PF_ADD_PARAM(in_data, -1, &def);

    out_data->num_params = FRACTALWARP_NUM_PARAMS;

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
    cl_kernel fractalwarp_kernel;
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
    ShaderObjectPtr mFractalWarpShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState> fractalwarp_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kFractalWarpKernel_OpenCLString) };
        char const* strings[] = { k16fString, kFractalWarpKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->fractalwarp_kernel = clCreateKernel(program, "FractalWarpKernel", &result);
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
        dx_gpu_data->mFractalWarpShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"FractalWarpKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mFractalWarpShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kFractalWarpKernelMetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> fractalwarp_function = nil;
            NSString* fractalwarp_name = [NSString stringWithCString : "FractalWarpKernel" encoding : NSUTF8StringEncoding];

            fractalwarp_function = [[library newFunctionWithName : fractalwarp_name]autorelease];

            if (!fractalwarp_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->fractalwarp_pipeline = [device newComputePipelineStateWithFunction : fractalwarp_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->fractalwarp_kernel);

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
        dx_gpu_dataP->mFractalWarpShader.reset();

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
static PixelType
SampleBilinear(PixelType* src, PF_FpLong x, PF_FpLong y, A_long width, A_long height, A_long rowbytes,
    bool x_tiles, bool y_tiles, bool mirror) {

    bool outsideBounds = false;

    if (x_tiles) {
        if (mirror) {
            float intPart;
            float fracPart = modff(fabsf(x / width), &intPart);
            int isOdd = (int)intPart & 1;
            x = isOdd ? (1.0f - fracPart) * width : fracPart * width;
        }
        else {
            x = fmodf(fmodf(x, width) + width, width);
        }
    }
    else {
        if (x < 0 || x >= width) {
            outsideBounds = true;
        }
    }

    if (y_tiles) {
        if (mirror) {
            float intPart;
            float fracPart = modff(fabsf(y / height), &intPart);
            int isOdd = (int)intPart & 1;
            y = isOdd ? (1.0f - fracPart) * height : fracPart * height;
        }
        else {
            y = fmodf(fmodf(y, height) + height, height);
        }
    }
    else {
        if (y < 0 || y >= height) {
            outsideBounds = true;
        }
    }

    if (outsideBounds) {
        PixelType transparent;
        if (std::is_same<PixelType, PF_PixelFloat>::value) {
            transparent.alpha = 0.0f;
            transparent.red = 0.0f;
            transparent.green = 0.0f;
            transparent.blue = 0.0f;
        }
        else if (std::is_same<PixelType, PF_Pixel16>::value) {
            transparent.alpha = 0;
            transparent.red = 0;
            transparent.green = 0;
            transparent.blue = 0;
        }
        else {
            transparent.alpha = 0;
            transparent.red = 0;
            transparent.green = 0;
            transparent.blue = 0;
        }
        return transparent;
    }

    x = MAX(0, MIN(width - 1.001f, x));
    y = MAX(0, MIN(height - 1.001f, y));

    A_long x0 = static_cast<A_long>(x);
    A_long y0 = static_cast<A_long>(y);
    A_long x1 = MIN(x0 + 1, width - 1);
    A_long y1 = MIN(y0 + 1, height - 1);

    PF_FpLong fx = x - x0;
    PF_FpLong fy = y - y0;

    PixelType* p00 = (PixelType*)((char*)src + y0 * rowbytes) + x0;
    PixelType* p01 = (PixelType*)((char*)src + y0 * rowbytes) + x1;
    PixelType* p10 = (PixelType*)((char*)src + y1 * rowbytes) + x0;
    PixelType* p11 = (PixelType*)((char*)src + y1 * rowbytes) + x1;

    PixelType result;
    if (std::is_same<PixelType, PF_PixelFloat>::value) {
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
    else if (std::is_same<PixelType, PF_Pixel16>::value) {
        result.alpha = static_cast<A_u_short>(
            (1.0f - fx) * (1.0f - fy) * p00->alpha +
            fx * (1.0f - fy) * p01->alpha +
            (1.0f - fx) * fy * p10->alpha +
            fx * fy * p11->alpha + 0.5f);

        result.red = static_cast<A_u_short>(
            (1.0f - fx) * (1.0f - fy) * p00->red +
            fx * (1.0f - fy) * p01->red +
            (1.0f - fx) * fy * p10->red +
            fx * fy * p11->red + 0.5f);

        result.green = static_cast<A_u_short>(
            (1.0f - fx) * (1.0f - fy) * p00->green +
            fx * (1.0f - fy) * p01->green +
            (1.0f - fx) * fy * p10->green +
            fx * fy * p11->green + 0.5f);

        result.blue = static_cast<A_u_short>(
            (1.0f - fx) * (1.0f - fy) * p00->blue +
            fx * (1.0f - fy) * p01->blue +
            (1.0f - fx) * fy * p10->blue +
            fx * fy * p11->blue + 0.5f);
    }
    else {
        result.alpha = static_cast<A_u_char>(
            (1.0f - fx) * (1.0f - fy) * p00->alpha +
            fx * (1.0f - fy) * p01->alpha +
            (1.0f - fx) * fy * p10->alpha +
            fx * fy * p11->alpha + 0.5f);

        result.red = static_cast<A_u_char>(
            (1.0f - fx) * (1.0f - fy) * p00->red +
            fx * (1.0f - fy) * p01->red +
            (1.0f - fx) * fy * p10->red +
            fx * fy * p11->red + 0.5f);

        result.green = static_cast<A_u_char>(
            (1.0f - fx) * (1.0f - fy) * p00->green +
            fx * (1.0f - fy) * p01->green +
            (1.0f - fx) * fy * p10->green +
            fx * fy * p11->green + 0.5f);

        result.blue = static_cast<A_u_char>(
            (1.0f - fx) * (1.0f - fy) * p00->blue +
            fx * (1.0f - fy) * p01->blue +
            (1.0f - fx) * fy * p10->blue +
            fx * fy * p11->blue + 0.5f);
    }

    return result;
}

inline float Mix(float a, float b, float t) {
    double a_d = (double)a;
    double b_d = (double)b;
    double t_d = (double)t;
    double result_d = a_d * (1.0 - t_d) + b_d * t_d;
    return (float)result_d;
}

static float Fract(float x) {
    double x_d = (double)x;
    double result_d = x_d - floor(x_d);
    return (float)result_d;
}

static float Random(float x, float y) {
    float sin_input = x * 12.9898f + y * 78.233f;

    double sin_input_d = (double)sin_input;
    double sin_result_d = sin(sin_input_d);
    double multiplied_d = sin_result_d * 43758.5453123;

    double fract_d = multiplied_d - floor(multiplied_d);
    return (float)fract_d;
}

static float Noise(float x, float y) {
    double x_d = (double)x;
    double y_d = (double)y;

    double i = floor(x_d);
    double j = floor(y_d);
    double f = x_d - i;
    double g = y_d - j;

    float a = Random((float)i, (float)j);
    float b = Random((float)(i + 1.0), (float)j);
    float c = Random((float)i, (float)(j + 1.0));
    float d = Random((float)(i + 1.0), (float)(j + 1.0));

    double a_d = (double)a;
    double b_d = (double)b;
    double c_d = (double)c;
    double d_d = (double)d;

    double u = f * f * (3.0 - 2.0 * f);
    double v = g * g * (3.0 - 2.0 * g);

    double mix1 = a_d * (1.0 - u) + b_d * u;
    double mix2 = c_d * (1.0 - u) + d_d * u;
    double result_d = mix1 * (1.0 - v) + mix2 * v;

    return (float)result_d;
}

static float FBM(float x, float y, float px, float py, int octaveCount, float intensity) {
    double x_d = (double)x;
    double y_d = (double)y;
    double px_d = (double)px;
    double py_d = (double)py;
    double intensity_d = (double)intensity;

    double value = 0.0;
    double amplitude = 0.5;

    for (int i = 0; i < octaveCount; i++) {
        float noise_val = Noise((float)x_d, (float)y_d);
        double noise_val_d = (double)noise_val;

        value += amplitude * noise_val_d;
        x_d = x_d * 2.0 + px_d;
        y_d = y_d * 2.0 + py_d;
        amplitude *= intensity_d;
    }

    return (float)value;
}




template <typename PixelType>
static PF_Err
FractalWarpFunc(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    PF_Err err = PF_Err_NONE;
    FractalWarpParams* params = reinterpret_cast<FractalWarpParams*>(refcon);

    if (params) {
        float adjusted_x = static_cast<float>(xL);
        float adjusted_y = static_cast<float>(yL);

        const float width = static_cast<float>(params->width);
        const float height = static_cast<float>(params->height);
        const float aspectRatio = width / height;

        float st_x, st_y;
        if (params->screenSpace) {
            st_x = adjusted_x / width;
            st_y = 1.0f - (adjusted_y / height);
            st_x *= aspectRatio;
        }
        else {
            st_x = adjusted_x / width;
            st_y = 1.0f - (adjusted_y / height);
            st_x *= aspectRatio;
        }

        float position_offset_x = params->positionX / 1000.0f * -1.0f;
        float position_offset_y = params->positionY / 1000.0f * 1.0f;

        st_x += position_offset_x;
        st_y += position_offset_y;

        float parallax_x = params->parallaxX / 200.0f * -1.0f;
        float parallax_y = params->parallaxY / 200.0f * 1.0f;

        float fbm_x_coord = ((st_x - 0.5f) * 3.0f * params->detail) + 0.5f;
        float fbm_y_coord = ((st_y - 0.5f) * 3.0f * params->detail) + 0.5f;

        const float dx = FBM(
            fbm_x_coord,
            fbm_y_coord,
            parallax_x, parallax_y,
            params->octaves, params->lacunarity
        );

        const float dy = FBM(
            ((st_x + 25.3f - 0.5f) * 3.0f * params->detail) + 0.5f,
            ((st_y + 12.9f - 0.5f) * 3.0f * params->detail) + 0.5f,
            parallax_x, parallax_y,
            params->octaves, params->lacunarity
        );

        float sample_x, sample_y;
        if (params->screenSpace) {
            sample_x = adjusted_x / width;
            sample_y = 1.0f - (adjusted_y / height);
        }
        else {
            sample_x = adjusted_x / width;
            sample_y = 1.0f - (adjusted_y / height);
        }

        sample_x += (dx - 0.5f) * params->magnitude;
        sample_y += (dy - 0.5f) * params->magnitude;

        sample_x *= width;
        sample_y = (1.0f - sample_y) * height;

        *outP = SampleBilinear(static_cast<PixelType*>(params->inputP),
            sample_x, sample_y,
            params->width, params->height,
            params->rowbytes,
            params->x_tiles,
            params->y_tiles,
            params->mirror);
    }

    return err;
}


static PF_Err
LegacyRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    FractalWarpParams fwParams;
    memset(&fwParams, 0, sizeof(FractalWarpParams));

    PF_LayerDef* input_layer = &params[FRACTALWARP_INPUT]->u.ld;

    fwParams.positionX = params[FRACTALWARP_POSITION]->u.td.x_value / 65536.0f;
    fwParams.positionY = params[FRACTALWARP_POSITION]->u.td.y_value / 65536.0f;

    fwParams.parallaxX = params[FRACTALWARP_PARALLAX]->u.td.x_value / 65536.0f;
    fwParams.parallaxY = params[FRACTALWARP_PARALLAX]->u.td.y_value / 65536.0f;

    fwParams.magnitude = params[FRACTALWARP_MAGNITUDE]->u.fs_d.value;

    fwParams.detail = params[FRACTALWARP_DETAIL]->u.fs_d.value;

    fwParams.lacunarity = params[FRACTALWARP_LACUNARITY]->u.fs_d.value;

    fwParams.screenSpace = params[FRACTALWARP_SCREENSPACE]->u.bd.value;

    fwParams.octaves = params[FRACTALWARP_OCTAVES]->u.sd.value;

    fwParams.x_tiles = params[FRACTALWARP_X_TILES]->u.bd.value;
    fwParams.y_tiles = params[FRACTALWARP_Y_TILES]->u.bd.value;
    fwParams.mirror = params[FRACTALWARP_MIRROR]->u.bd.value;

    fwParams.width = output->width;
    fwParams.height = output->height;
    fwParams.inputP = input_layer->data;
    fwParams.rowbytes = input_layer->rowbytes;

    A_long linesL = output->height;

    A_long pixels_per_row = input_layer->width;
    A_long bytes_per_pixel = input_layer->rowbytes / pixels_per_row;

    bool is_8bit = (bytes_per_pixel <= 4);
    bool is_16bit = (bytes_per_pixel > 4 && bytes_per_pixel <= 8);
    bool is_32bit = (bytes_per_pixel > 8);

    if (is_32bit) {
        err = suites.IterateFloatSuite1()->iterate(
            in_data,
            0,
            linesL,
            input_layer,
            NULL,
            (void*)&fwParams,
            FractalWarpFunc<PF_PixelFloat>,
            output);
    }
    else if (is_16bit) {
        err = suites.Iterate16Suite2()->iterate(
            in_data,
            0,
            linesL,
            input_layer,
            NULL,
            (void*)&fwParams,
            FractalWarpFunc<PF_Pixel16>,
            output);
    }
    else {
        err = suites.Iterate8Suite2()->iterate(
            in_data,
            0,
            linesL,
            input_layer,
            NULL,
            (void*)&fwParams,
            FractalWarpFunc<PF_Pixel8>,
            output);
    }

    return err;
}


static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        FractalWarpParams* paramsP = reinterpret_cast<FractalWarpParams*>(pre_render_dataPV);
        free(paramsP);
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

    FractalWarpParams* paramsP = reinterpret_cast<FractalWarpParams*>(malloc(sizeof(FractalWarpParams)));

    if (paramsP) {
        PF_ParamDef cur_param;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_POSITION, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->positionX = cur_param.u.td.x_value / 65536.0f;
        paramsP->positionY = cur_param.u.td.y_value / 65536.0f;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_PARALLAX, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->parallaxX = cur_param.u.td.x_value / 65536.0f;
        paramsP->parallaxY = cur_param.u.td.y_value / 65536.0f;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_MAGNITUDE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->magnitude = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_DETAIL, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->detail = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_LACUNARITY, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->lacunarity = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_SCREENSPACE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->screenSpace = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_OCTAVES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->octaves = cur_param.u.sd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->x_tiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->y_tiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, FRACTALWARP_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        paramsP->mirror = cur_param.u.bd.value;

        extraP->output->pre_render_data = paramsP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            FRACTALWARP_INPUT,
            FRACTALWARP_INPUT,
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
    FractalWarpParams* paramsP)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    FractalWarpParams cpuParams;
    cpuParams.positionX = paramsP->positionX;
    cpuParams.positionY = paramsP->positionY;
    cpuParams.parallaxX = paramsP->parallaxX;
    cpuParams.parallaxY = paramsP->parallaxY;
    cpuParams.magnitude = paramsP->magnitude;
    cpuParams.detail = paramsP->detail;
    cpuParams.lacunarity = paramsP->lacunarity;
    cpuParams.screenSpace = paramsP->screenSpace;
    cpuParams.octaves = paramsP->octaves;
    cpuParams.x_tiles = paramsP->x_tiles;
    cpuParams.y_tiles = paramsP->y_tiles;
    cpuParams.mirror = paramsP->mirror;
    cpuParams.width = input_worldP->width;
    cpuParams.height = input_worldP->height;
    cpuParams.inputP = input_worldP->data;
    cpuParams.rowbytes = input_worldP->rowbytes;

    A_long linesL = output_worldP->height;

    switch (pixel_format) {
    case PF_PixelFormat_ARGB128: {
        err = suites.IterateFloatSuite1()->iterate(
            in_data,
            0,
            linesL,
            input_worldP,
            NULL,
            (void*)&cpuParams,
            FractalWarpFunc<PF_PixelFloat>,
            output_worldP);
        break;
    }

    case PF_PixelFormat_ARGB64: {
        err = suites.Iterate16Suite1()->iterate(
            in_data,
            0,
            linesL,
            input_worldP,
            NULL,
            (void*)&cpuParams,
            FractalWarpFunc<PF_Pixel16>,
            output_worldP);
        break;
    }

    case PF_PixelFormat_ARGB32: {
        err = suites.Iterate8Suite1()->iterate(
            in_data,
            0,
            linesL,
            input_worldP,
            NULL,
            (void*)&cpuParams,
            FractalWarpFunc<PF_Pixel8>,
            output_worldP);
        break;
    }

    default:
        err = PF_Err_BAD_CALLBACK_PARAM;
        break;
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
    float mPositionX;
    float mPositionY;
    float mParallaxX;
    float mParallaxY;
    float mMagnitude;
    float mDetail;
    float mLacunarity;
    int mScreenSpace;
    int mOctaves;
    int mXTiles;
    int mYTiles;
    int mMirror;
} FractalWarpGPUParams;


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
    FractalWarpParams* paramsP)
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

    FractalWarpGPUParams kernel_params;
    kernel_params.mWidth = input_worldP->width;
    kernel_params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    kernel_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    kernel_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    kernel_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    kernel_params.mPositionX = paramsP->positionX;
    kernel_params.mPositionY = paramsP->positionY;
    kernel_params.mParallaxX = paramsP->parallaxX;
    kernel_params.mParallaxY = paramsP->parallaxY;
    kernel_params.mMagnitude = paramsP->magnitude;
    kernel_params.mDetail = paramsP->detail;
    kernel_params.mLacunarity = paramsP->lacunarity;
    kernel_params.mScreenSpace = paramsP->screenSpace;
    kernel_params.mOctaves = paramsP->octaves;
    kernel_params.mXTiles = paramsP->x_tiles;
    kernel_params.mYTiles = paramsP->y_tiles;
    kernel_params.mMirror = paramsP->mirror;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(float), &kernel_params.mPositionX));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(float), &kernel_params.mPositionY));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(float), &kernel_params.mParallaxX));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(float), &kernel_params.mParallaxY));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(float), &kernel_params.mMagnitude));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(float), &kernel_params.mDetail));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(float), &kernel_params.mLacunarity));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mScreenSpace));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mOctaves));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mXTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mYTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->fractalwarp_kernel, param_index++, sizeof(int), &kernel_params.mMirror));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(kernel_params.mWidth, threadBlock[0]), RoundUp(kernel_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->fractalwarp_kernel,
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
        FractalWarp_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            kernel_params.mSrcPitch,
            kernel_params.mDstPitch,
            kernel_params.m16f,
            kernel_params.mWidth,
            kernel_params.mHeight,
            kernel_params.mPositionX,
            kernel_params.mPositionY,
            kernel_params.mParallaxX,
            kernel_params.mParallaxY,
            kernel_params.mMagnitude,
            kernel_params.mDetail,
            kernel_params.mLacunarity,
            kernel_params.mScreenSpace,
            kernel_params.mOctaves,
            kernel_params.mXTiles,
            kernel_params.mYTiles,
            kernel_params.mMirror);

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
                dx_gpu_data->mFractalWarpShader,
                3);

            DX_ERR(shaderExecution.SetParamBuffer(&kernel_params, sizeof(FractalWarpGPUParams)));
            DX_ERR(shaderExecution.SetUnorderedAccessView(
                (ID3D12Resource*)dst_mem,
                kernel_params.mHeight * dst_row_bytes));
            DX_ERR(shaderExecution.SetShaderResourceView(
                (ID3D12Resource*)src_mem,
                kernel_params.mHeight * src_row_bytes));
            DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(kernel_params.mWidth, 16), (UINT)DivideRoundUp(kernel_params.mHeight, 16)));
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
        id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &kernel_params
            length : sizeof(FractalWarpGPUParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->fractalwarp_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(kernel_params.mWidth, threadsPerGroup.width), DivideRoundUp(kernel_params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->fractalwarp_pipeline] ;
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

    FractalWarpParams* paramsP = reinterpret_cast<FractalWarpParams*>(extraP->input->pre_render_data);

    if (paramsP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, FRACTALWARP_INPUT, &input_worldP)));
        ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

        AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
            kPFWorldSuite,
            kPFWorldSuiteVersion2,
            out_data);
        PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
        ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

        if (isGPU) {
            ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, paramsP));
        }
        else {
            ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, paramsP));
        }
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, FRACTALWARP_INPUT));
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
        "Fractal Warp",
        "DKT FractalWarp",
        "DKT Effects",
        AE_RESERVED_INFO,
        "EffectMain",
        "https://www.example.com");

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
            err = LegacyRender(in_dataP, out_data, params, output);
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

