#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "WaveWarp.h"
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


extern void WaveWarp_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float phase,
    float direction,
    float spacing,
    float magnitude,
    float warpAngle,
    float damping,
    float dampingSpace,
    float dampingOrigin,
    int screenSpace,
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

    PF_ADD_FLOAT_SLIDERX("Phase",
        WAVEWARP_PHASE_MIN,
        WAVEWARP_PHASE_MAX,
        0,
        5,
        WAVEWARP_PHASE_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        WAVEWARP_PHASE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Angle", WAVEWARP_ANGLE_DFLT, WAVEWARP_ANGLE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Spacing",
        WAVEWARP_SPACING_MIN,
        WAVEWARP_SPACING_MAX,
        WAVEWARP_SPACING_MIN,
        WAVEWARP_SPACING_MAX,
        WAVEWARP_SPACING_DFLT,
        PF_Precision_TENTHS,
        0,
        0,
        WAVEWARP_SPACING);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Magnitude",
        WAVEWARP_MAGNITUDE_MIN,
        WAVEWARP_MAGNITUDE_MAX,
        WAVEWARP_MAGNITUDE_MIN,
        WAVEWARP_MAGNITUDE_MAX,
        WAVEWARP_MAGNITUDE_DFLT,
        PF_Precision_TENTHS,
        0,
        0,
        WAVEWARP_MAGNITUDE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Warp Angle",
        -180,
        180,
        -180,
        180,
        WAVEWARP_WARPANGLE_DFLT,
        PF_Precision_INTEGER,
        0,
        0,
        WAVEWARP_WARPANGLE);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_START;
    PF_STRCPY(def.name, "Damping");
    def.flags = PF_ParamFlag_START_COLLAPSED;
    PF_ADD_PARAM(in_data, -1, &def);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Magnitude",
        WAVEWARP_DAMPING_MIN,
        WAVEWARP_DAMPING_MAX,
        WAVEWARP_DAMPING_MIN,
        WAVEWARP_DAMPING_MAX,
        WAVEWARP_DAMPING_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        WAVEWARP_DAMPING);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Spacing",
        WAVEWARP_DAMPINGSPACE_MIN,
        WAVEWARP_DAMPINGSPACE_MAX,
        WAVEWARP_DAMPINGSPACE_MIN,
        WAVEWARP_DAMPINGSPACE_MAX,
        WAVEWARP_DAMPINGSPACE_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        WAVEWARP_DAMPINGSPACE);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Anchor",
        WAVEWARP_DAMPINGORIGIN_MIN,
        WAVEWARP_DAMPINGORIGIN_MAX,
        WAVEWARP_DAMPINGORIGIN_MIN,
        WAVEWARP_DAMPINGORIGIN_MAX,
        WAVEWARP_DAMPINGORIGIN_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        WAVEWARP_DAMPINGORIGIN);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Screen Space",
        "Screen Space",
        FALSE,
        0,
        WAVEWARP_SCREENSPACE);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    PF_ADD_PARAM(in_data, -1, &def);

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
        WAVEWARP_XTILES);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Y Tiles",
        "",
        FALSE,
        0,
        WAVEWARP_YTILES);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Mirror",
        "",
        FALSE,
        0,
        WAVEWARP_MIRROR);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    PF_ADD_PARAM(in_data, -1, &def);

    out_data->num_params = WAVEWARP_NUM_PARAMS;

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
    cl_kernel wavewarp_kernel;
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
    ShaderObjectPtr mWaveWarpShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState>wavewarp_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kWaveWarpKernel_OpenCLString) };
        char const* strings[] = { k16fString, kWaveWarpKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->wavewarp_kernel = clCreateKernel(program, "WaveWarpKernel", &result);
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
        dx_gpu_data->mWaveWarpShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"WaveWarpKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mWaveWarpShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kWaveWarpKernel_MetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> wavewarp_function = nil;
            NSString* wavewarp_name = [NSString stringWithCString : "WaveWarpKernel" encoding : NSUTF8StringEncoding];

            wavewarp_function = [[library newFunctionWithName : wavewarp_name]autorelease];

            if (!wavewarp_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->wavewarp_pipeline = [device newComputePipelineStateWithFunction : wavewarp_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->wavewarp_kernel);

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
        dx_gpu_dataP->mWaveWarpShader.reset();

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

        [metal_dataP->wavewarp_pipeline release] ;

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
PixelType SampleBilinear(PixelType* src, PF_FpLong x, PF_FpLong y, A_long width, A_long height, A_long rowbytes, bool xTile, bool yTile, bool mirror) {
    PF_FpLong tiled_x = x;
    PF_FpLong tiled_y = y;

    if (xTile) {
        if (mirror) {
            PF_FpLong intPart;
            PF_FpLong fracPart = modf(fabs(tiled_x / width), &intPart);
            int isOdd = (int)intPart & 1;
            tiled_x = isOdd ? (width - fracPart * width) : (fracPart * width);
        }
        else {
            tiled_x = fmod(tiled_x, width);
            if (tiled_x < 0) tiled_x += width;
        }
    }
    else if (tiled_x < 0 || tiled_x >= width) {
        PixelType result;
        if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
            result.alpha = result.red = result.green = result.blue = 0.0f;
        }
        else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
            result.alpha = result.red = result.green = result.blue = 0;
        }
        else {
            result.alpha = result.red = result.green = result.blue = 0;
        }
        return result;
    }

    if (yTile) {
        if (mirror) {
            PF_FpLong intPart;
            PF_FpLong fracPart = modf(fabs(tiled_y / height), &intPart);
            int isOdd = (int)intPart & 1;
            tiled_y = isOdd ? (height - fracPart * height) : (fracPart * height);
        }
        else {
            tiled_y = fmod(tiled_y, height);
            if (tiled_y < 0) tiled_y += height;
        }
    }
    else if (tiled_y < 0 || tiled_y >= height) {
        PixelType result;
        if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
            result.alpha = result.red = result.green = result.blue = 0.0f;
        }
        else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
            result.alpha = result.red = result.green = result.blue = 0;
        }
        else {
            result.alpha = result.red = result.green = result.blue = 0;
        }
        return result;
    }

    A_long x0 = (A_long)tiled_x;
    A_long y0 = (A_long)tiled_y;
    A_long x1 = (x0 + 1) % width;       
    A_long y1 = (y0 + 1) % height;      

    PF_FpLong fx = tiled_x - x0;
    PF_FpLong fy = tiled_y - y0;

    PixelType* p00 = (PixelType*)((char*)src + y0 * rowbytes + x0 * sizeof(PixelType));
    PixelType* p01 = (PixelType*)((char*)src + y0 * rowbytes + x1 * sizeof(PixelType));
    PixelType* p10 = (PixelType*)((char*)src + y1 * rowbytes + x0 * sizeof(PixelType));
    PixelType* p11 = (PixelType*)((char*)src + y1 * rowbytes + x1 * sizeof(PixelType));

    PixelType result;
    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        result.alpha = (PF_FpShort)(
            p00->alpha * (1 - fx) * (1 - fy) +
            p01->alpha * fx * (1 - fy) +
            p10->alpha * (1 - fx) * fy +
            p11->alpha * fx * fy);

        result.red = (PF_FpShort)(
            p00->red * (1 - fx) * (1 - fy) +
            p01->red * fx * (1 - fy) +
            p10->red * (1 - fx) * fy +
            p11->red * fx * fy);

        result.green = (PF_FpShort)(
            p00->green * (1 - fx) * (1 - fy) +
            p01->green * fx * (1 - fy) +
            p10->green * (1 - fx) * fy +
            p11->green * fx * fy);

        result.blue = (PF_FpShort)(
            p00->blue * (1 - fx) * (1 - fy) +
            p01->blue * fx * (1 - fy) +
            p10->blue * (1 - fx) * fy +
            p11->blue * fx * fy);
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        result.alpha = (A_u_short)(
            p00->alpha * (1 - fx) * (1 - fy) +
            p01->alpha * fx * (1 - fy) +
            p10->alpha * (1 - fx) * fy +
            p11->alpha * fx * fy);

        result.red = (A_u_short)(
            p00->red * (1 - fx) * (1 - fy) +
            p01->red * fx * (1 - fy) +
            p10->red * (1 - fx) * fy +
            p11->red * fx * fy);

        result.green = (A_u_short)(
            p00->green * (1 - fx) * (1 - fy) +
            p01->green * fx * (1 - fy) +
            p10->green * (1 - fx) * fy +
            p11->green * fx * fy);

        result.blue = (A_u_short)(
            p00->blue * (1 - fx) * (1 - fy) +
            p01->blue * fx * (1 - fy) +
            p10->blue * (1 - fx) * fy +
            p11->blue * fx * fy);
    }
    else {
        result.alpha = (A_u_char)(
            p00->alpha * (1 - fx) * (1 - fy) +
            p01->alpha * fx * (1 - fy) +
            p10->alpha * (1 - fx) * fy +
            p11->alpha * fx * fy);

        result.red = (A_u_char)(
            p00->red * (1 - fx) * (1 - fy) +
            p01->red * fx * (1 - fy) +
            p10->red * (1 - fx) * fy +
            p11->red * fx * fy);

        result.green = (A_u_char)(
            p00->green * (1 - fx) * (1 - fy) +
            p01->green * fx * (1 - fy) +
            p10->green * (1 - fx) * fy +
            p11->green * fx * fy);

        result.blue = (A_u_char)(
            p00->blue * (1 - fx) * (1 - fy) +
            p01->blue * fx * (1 - fy) +
            p10->blue * (1 - fx) * fy +
            p11->blue * fx * fy);
    }

    return result;
}

#define WW_MIN(a, b) ((a) < (b) ? (a) : (b))
#define WW_MAX(a, b) ((a) > (b) ? (a) : (b))


template <typename PixelType>
static PF_Err
WaveWarpFunc(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PixelType* inP,
    PixelType* outP)
{
    PF_Err      err = PF_Err_NONE;

    WaveWarpInfo* wiP = reinterpret_cast<WaveWarpInfo*>(refcon);

    if (wiP) {
        PF_FpLong adjusted_x = xL;
        PF_FpLong adjusted_y = yL;

        PF_FpLong x_norm = adjusted_x / wiP->width;
        PF_FpLong y_norm = adjusted_y / wiP->height;

        PF_FpLong direction_deg = -wiP->direction;
        PF_FpLong warpangle_deg = -wiP->offset;

        PF_FpLong a1 = direction_deg * 0.0174533;
        PF_FpLong a2 = (direction_deg + warpangle_deg) * 0.0174533;

        PF_FpLong st_x, st_y;
        if (wiP->screenSpace) {
            st_x = x_norm;
            st_y = y_norm;
        }
        else {
            st_x = x_norm;
            st_y = y_norm;
        }

        PF_FpLong raw_v_x = cos(a1);
        PF_FpLong raw_v_y = -sin(a1);

        PF_FpLong raw_p = (st_x * raw_v_x) + (st_y * raw_v_y);

        PF_FpLong space_damp = 1.0;
        if (wiP->dampingSpace < 0.0) {
            space_damp = 1.0 - (WW_MIN(fabs(raw_p - wiP->dampingOrigin), 1.0) * (0.0 - wiP->dampingSpace));
        }
        else if (wiP->dampingSpace > 0.0) {
            space_damp = 1.0 - ((1.0 - WW_MIN(fabs(raw_p - wiP->dampingOrigin), 1.0)) * wiP->dampingSpace);
        }

        PF_FpLong space = wiP->spacing * space_damp;

        PF_FpLong v_x = cos(a1) * space;
        PF_FpLong v_y = -sin(a1) * space;

        PF_FpLong p = (st_x * v_x) + (st_y * v_y);

        PF_FpLong ddist = fabs(p / space);

        PF_FpLong damp = 1.0;
        if (wiP->damping < 0.0) {
            damp = 1.0 - (WW_MIN(fabs(ddist - wiP->dampingOrigin), 1.0) * (0.0 - wiP->damping));
        }
        else if (wiP->damping > 0.0) {
            damp = 1.0 - ((1.0 - WW_MIN(fabs(ddist - wiP->dampingOrigin), 1.0)) * wiP->damping);
        }

        PF_FpLong offs_x = cos(a2) * (wiP->magnitude * damp) / 100.0;
        PF_FpLong offs_y = -sin(a2) * (wiP->magnitude * damp) / 100.0;

        PF_FpLong wave = sin(p + wiP->phase * 6.28318);
        offs_x *= wave;
        offs_y *= wave;

        PF_FpLong sample_x_norm = x_norm + offs_x;
        PF_FpLong sample_y_norm = y_norm + offs_y;

        PF_FpLong sample_x_f = sample_x_norm * wiP->width;
        PF_FpLong sample_y_f = sample_y_norm * wiP->height;

        PixelType* srcP = (PixelType*)(wiP->srcData);

        *outP = SampleBilinear<PixelType>(srcP, sample_x_f, sample_y_f, wiP->width, wiP->height, wiP->rowbytes,
            wiP->xTiles, wiP->yTiles, wiP->mirror);
    }
    else {
        *outP = *inP;
    }

    return err;
}



static PF_Err
WaveWarpFunc8(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return WaveWarpFunc<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err
WaveWarpFunc16(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return WaveWarpFunc<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err
WaveWarpFunc32(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return WaveWarpFunc<PF_PixelFloat>(refcon, xL, yL, inP, outP);
}



static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        WaveWarpParams* infoP = reinterpret_cast<WaveWarpParams*>(pre_render_dataPV);
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

    WaveWarpParams* infoP = reinterpret_cast<WaveWarpParams*>(malloc(sizeof(WaveWarpParams)));

    if (infoP) {
        PF_ParamDef cur_param;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_PHASE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->phase = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_ANGLE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->direction = ((float)cur_param.u.ad.value / 65536.0f) / M_PI;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_SPACING, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->spacing = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_MAGNITUDE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->magnitude = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_WARPANGLE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->warpAngle = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_DAMPING, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->damping = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_DAMPINGSPACE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->dampingSpace = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_DAMPINGORIGIN, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->dampingOrigin = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_SCREENSPACE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->screenSpace = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_XTILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->xTiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_YTILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->yTiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, WAVEWARP_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mirror = cur_param.u.bd.value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            WAVEWARP_INPUT,
            WAVEWARP_INPUT,
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
    WaveWarpParams* infoP)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    WaveWarpInfo info;
    AEFX_CLR_STRUCT(info);

    info.phase = infoP->phase;
    info.direction = infoP->direction;
    info.spacing = infoP->spacing;
    info.magnitude = infoP->magnitude;
    info.offset = infoP->warpAngle;
    info.damping = infoP->damping;
    info.dampingSpace = infoP->dampingSpace;
    info.dampingOrigin = infoP->dampingOrigin;
    info.screenSpace = infoP->screenSpace;
    info.xTiles = infoP->xTiles;
    info.yTiles = infoP->yTiles;
    info.mirror = infoP->mirror;

    info.width = input_worldP->width;
    info.height = input_worldP->height;
    info.rowbytes = input_worldP->rowbytes;
    info.srcData = input_worldP->data;
    info.in_data = in_data;

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
                (void*)&info,
                WaveWarpFunc32,
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
                (void*)&info,
                WaveWarpFunc16,
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
                (void*)&info,
                WaveWarpFunc8,
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
    float mPhase;
    float mDirection;
    float mSpacing;
    float mMagnitude;
    float mWarpAngle;
    float mDamping;
    float mDampingSpace;
    float mDampingOrigin;
    int mScreenSpace;
    int mXTiles;
    int mYTiles;
    int mMirror;
} WaveWarpGPUParams;

static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    WaveWarpParams* infoP)
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

    WaveWarpGPUParams params;
    params.mWidth = input_worldP->width;
    params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    params.mPhase = infoP->phase;
    params.mDirection = infoP->direction;
    params.mSpacing = infoP->spacing;
    params.mMagnitude = infoP->magnitude;
    params.mWarpAngle = infoP->warpAngle;
    params.mDamping = infoP->damping;
    params.mDampingSpace = infoP->dampingSpace;
    params.mDampingOrigin = infoP->dampingOrigin;
    params.mScreenSpace = infoP->screenSpace;
    params.mXTiles = infoP->xTiles;
    params.mYTiles = infoP->yTiles;
    params.mMirror = infoP->mirror;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(int), &params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(int), &params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(int), &params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(unsigned int), &params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(unsigned int), &params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(float), &params.mPhase));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(float), &params.mDirection));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(float), &params.mSpacing));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(float), &params.mMagnitude));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(float), &params.mWarpAngle));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(float), &params.mDamping));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(float), &params.mDampingSpace));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(float), &params.mDampingOrigin));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(int), &params.mScreenSpace));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(int), &params.mXTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(int), &params.mYTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->wavewarp_kernel, param_index++, sizeof(int), &params.mMirror));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->wavewarp_kernel,
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
        WaveWarp_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            params.mSrcPitch,
            params.mDstPitch,
            params.m16f,
            params.mWidth,
            params.mHeight,
            params.mPhase,
            params.mDirection,
            params.mSpacing,
            params.mMagnitude,
            params.mWarpAngle,
            params.mDamping,
            params.mDampingSpace,
            params.mDampingOrigin,
            params.mScreenSpace,
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
            dx_gpu_data->mWaveWarpShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(WaveWarpGPUParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView(
            (ID3D12Resource*)dst_mem,
            params.mHeight * dst_row_bytes));
        DX_ERR(shaderExecution.SetShaderResourceView(
            (ID3D12Resource*)src_mem,
            params.mHeight * src_row_bytes));
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
            length : sizeof(WaveWarpGPUParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->wavewarp_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(params.mWidth, threadsPerGroup.width), DivideRoundUp(params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->wavewarp_pipeline] ;
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

    WaveWarpParams* infoP = reinterpret_cast<WaveWarpParams*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, WAVEWARP_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, WAVEWARP_INPUT));
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
    AEGP_SuiteHandler suites(in_dataP->pica_basicP);

    WaveWarpInfo info;
    AEFX_CLR_STRUCT(info);

    info.phase = params[WAVEWARP_PHASE]->u.fs_d.value;
    info.direction = params[WAVEWARP_ANGLE]->u.fs_d.value;
    info.spacing = params[WAVEWARP_SPACING]->u.fs_d.value;
    info.magnitude = params[WAVEWARP_MAGNITUDE]->u.fs_d.value;
    info.offset = params[WAVEWARP_WARPANGLE]->u.fs_d.value;
    info.damping = params[WAVEWARP_DAMPING]->u.fs_d.value;
    info.dampingSpace = params[WAVEWARP_DAMPINGSPACE]->u.fs_d.value;
    info.dampingOrigin = params[WAVEWARP_DAMPINGORIGIN]->u.fs_d.value;
    info.screenSpace = params[WAVEWARP_SCREENSPACE]->u.bd.value;

    info.xTiles = params[WAVEWARP_XTILES]->u.bd.value;
    info.yTiles = params[WAVEWARP_YTILES]->u.bd.value;
    info.mirror = params[WAVEWARP_MIRROR]->u.bd.value;

    info.width = output->width;
    info.height = output->height;
    info.rowbytes = output->rowbytes;
    info.srcData = params[WAVEWARP_INPUT]->u.ld.data;
    info.in_data = in_dataP;

    A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

    A_long bytes_per_pixel = output->rowbytes / output->width;
    bool is_16bit = (bytes_per_pixel > 4 && bytes_per_pixel <= 8);
    bool is_32bit = (bytes_per_pixel > 8);

    if (is_32bit) {
        ERR(suites.IterateFloatSuite1()->iterate(
            in_dataP,
            0,
            linesL,
            &params[WAVEWARP_INPUT]->u.ld,
            NULL,
            (void*)&info,
            WaveWarpFunc32,
            output));
    }
    else if (is_16bit) {
        ERR(suites.Iterate16Suite2()->iterate(
            in_dataP,
            0,
            linesL,
            &params[WAVEWARP_INPUT]->u.ld,
            NULL,
            (void*)&info,
            WaveWarpFunc16,
            output));
    }
    else {
        ERR(suites.Iterate8Suite2()->iterate(
            in_dataP,
            0,
            linesL,
            &params[WAVEWARP_INPUT]->u.ld,
            NULL,
            (void*)&info,
            WaveWarpFunc8,
            output));
    }

    return err;
}


extern "C" DllExport
PF_Err PluginDataEntryFunction(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;

    result = PF_REGISTER_EFFECT(
        inPtr,
        inPluginDataCallBackPtr,
        "Wave Warp",            
        "DKT Wave Warp",              
        "DKT Effects",         
        AE_RESERVED_INFO
    );

    return result;
}


PF_Err
EffectMain(
    PF_Cmd          cmd,
    PF_InData* in_dataP,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err      err = PF_Err_NONE;

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

