
#define NOMINMAX
#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "RasterTransform.h"
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

extern void RasterTransform_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float scale,
    float angle,
    float offsetX,
    float offsetY,
    int maskToLayer,
    float alpha,
    float fill,
    int sample,
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

    PF_ADD_FLOAT_SLIDERX(STR_SCALE_PARAM,
        TRANSFORM_SCALE_MIN,
        TRANSFORM_SCALE_MAX,
        TRANSFORM_SCALE_MIN,
        TRANSFORM_SCALE_MAX,
        TRANSFORM_SCALE_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        SCALE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_ANGLE(STR_ANGLE_PARAM,
        TRANSFORM_ANGLE_DFLT,
        ANGLE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_POINT(STR_OFFSET_PARAM,
        0,       
        0,       
        0,       
        OFFSET_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_CHECKBOX(STR_MASK_PARAM,
        "",
        FALSE,
        0,
        MASK_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_FLOAT_SLIDERX(STR_ALPHA_PARAM,
        TRANSFORM_ALPHA_MIN,
        TRANSFORM_ALPHA_MAX,
        TRANSFORM_ALPHA_MIN,
        TRANSFORM_ALPHA_MAX,
        TRANSFORM_ALPHA_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        ALPHA_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_FLOAT_SLIDERX(STR_FILL_PARAM,
        TRANSFORM_FILL_MIN,
        TRANSFORM_FILL_MAX,
        TRANSFORM_FILL_MIN,
        TRANSFORM_FILL_MAX,
        TRANSFORM_FILL_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        FILL_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_POPUP(STR_SAMPLE_PARAM,
        2,     
        TRANSFORM_SAMPLE_DFLT + 1,     
        STR_NEAREST " | " STR_LINEAR,       
        SAMPLE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_START;
    PF_STRCPY(def.name, "Tiles");
    def.flags = PF_ParamFlag_START_COLLAPSED;
    PF_ADD_PARAM(in_data, TRANSFORM_TILES_GROUP_START, &def);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("X Tiles",
        "",
        TRANSFORM_X_TILES_DFLT,
        0,
        TRANSFORM_X_TILES);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Y Tiles",
        "",
        TRANSFORM_Y_TILES_DFLT,
        0,
        TRANSFORM_Y_TILES);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Mirror",
        "",
        TRANSFORM_MIRROR_DFLT,
        0,
        TRANSFORM_MIRROR);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    PF_ADD_PARAM(in_data, TRANSFORM_TILES_GROUP_END, &def);

    out_data->num_params = TRANSFORM_NUM_PARAMS;

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
    cl_kernel transform_kernel;
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
    ShaderObjectPtr mTransformShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState> transform_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kRasterTransformKernel_OpenCLString) };
        char const* strings[] = { k16fString, kRasterTransformKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->transform_kernel = clCreateKernel(program, "RasterTransformKernel", &result);
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
        dx_gpu_data->mTransformShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"RasterTransformKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mTransformShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kRasterTransform_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> transform_function = nil;
            NSString* transform_name = [NSString stringWithCString : "RasterTransformKernel" encoding : NSUTF8StringEncoding];

            transform_function = [[library newFunctionWithName : transform_name]autorelease];

            if (!transform_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->transform_pipeline = [device newComputePipelineStateWithFunction : transform_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->transform_kernel);

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
        dx_gpu_dataP->mTransformShader.reset();

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
static PF_Err
RasterTransformFunc(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelT* inP,
    PixelT* outP)
{
    PF_Err err = PF_Err_NONE;
    TransformInfo* tiP = reinterpret_cast<TransformInfo*>(refcon);

    if (!tiP) return PF_Err_BAD_CALLBACK_PARAM;

    float width = static_cast<float>(tiP->input_width);
    float height = static_cast<float>(tiP->input_height);

    float st_x = static_cast<float>(xL) / width;
    float st_y = static_cast<float>(yL) / height;

    st_x -= tiP->offset_x / 500.0f;
    st_y -= tiP->offset_y / 500.0f;

    st_x -= 0.5f;
    st_y -= 0.5f;

    st_x *= width / height;

    float angle_rad = tiP->angle * 0.0174533f;
    float cos_angle = cos(angle_rad);
    float sin_angle = sin(angle_rad);

    float rotated_x = st_x * cos_angle - st_y * sin_angle;
    float rotated_y = st_x * sin_angle + st_y * cos_angle;
    st_x = rotated_x;
    st_y = rotated_y;

    st_x /= tiP->scale;
    st_y /= tiP->scale;

    st_x /= width / height;

    st_x += 0.5f;
    st_y += 0.5f;

    float sample_x = st_x * width;
    float sample_y = st_y * height;

    PixelT transformedPixel;
    PixelT basePixel;

    if (tiP->maskToLayer || tiP->fill > 0.0001f) {
        basePixel = *inP;
    }

    if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
        transformedPixel.red = 0;
        transformedPixel.green = 0;
        transformedPixel.blue = 0;
        transformedPixel.alpha = 0;
    }
    else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
        transformedPixel.red = 0;
        transformedPixel.green = 0;
        transformedPixel.blue = 0;
        transformedPixel.alpha = 0;
    }
    else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
        transformedPixel.red = 0.0f;
        transformedPixel.green = 0.0f;
        transformedPixel.blue = 0.0f;
        transformedPixel.alpha = 0.0f;
    }

    bool outsideBounds = false;

    if (tiP->x_tiles) {
        if (tiP->mirror) {
            float intPart;
            float fracPart = modff(fabsf(sample_x / width), &intPart);
            int isOdd = (int)intPart & 1;
            sample_x = isOdd ? (1.0f - fracPart) * width : fracPart * width;
        }
        else {
            sample_x = fmodf(fmodf(sample_x, width) + width, width);
        }
    }
    else {
        if (sample_x < 0 || sample_x >= width) {
            outsideBounds = true;
        }
    }

    if (tiP->y_tiles) {
        if (tiP->mirror) {
            float intPart;
            float fracPart = modff(fabsf(sample_y / height), &intPart);
            int isOdd = (int)intPart & 1;
            sample_y = isOdd ? (1.0f - fracPart) * height : fracPart * height;
        }
        else {
            sample_y = fmodf(fmodf(sample_y, height) + height, height);
        }
    }
    else {
        if (sample_y < 0 || sample_y >= height) {
            outsideBounds = true;
        }
    }

    if (!outsideBounds) {
        if (tiP->sample == TRANSFORM_SAMPLE_NEAREST) {
            A_long rounded_x = static_cast<A_long>(sample_x + 0.5f);
            A_long rounded_y = static_cast<A_long>(sample_y + 0.5f);

            rounded_x = std::max<A_long>(0L, std::min<A_long>(rounded_x, static_cast<A_long>(width) - 1));
            rounded_y = std::max<A_long>(0L, std::min<A_long>(rounded_y, static_cast<A_long>(height) - 1));

            char* pixel_ptr = (char*)tiP->input_worldP->data +
                rounded_y * tiP->input_worldP->rowbytes +
                rounded_x * sizeof(PixelT);
            transformedPixel = *reinterpret_cast<PixelT*>(pixel_ptr);
        }
        else {
            A_long x1 = static_cast<A_long>(sample_x);
            A_long y1 = static_cast<A_long>(sample_y);
            A_long x2 = x1 + 1;
            A_long y2 = y1 + 1;

            x1 = std::max<A_long>(0L, std::min<A_long>(x1, static_cast<A_long>(width) - 1));
            y1 = std::max<A_long>(0L, std::min<A_long>(y1, static_cast<A_long>(height) - 1));
            x2 = std::max<A_long>(0L, std::min<A_long>(x2, static_cast<A_long>(width) - 1));
            y2 = std::max<A_long>(0L, std::min<A_long>(y2, static_cast<A_long>(height) - 1));

            float fx = sample_x - x1;
            float fy = sample_y - y1;

            PixelT p11 = *reinterpret_cast<PixelT*>((char*)tiP->input_worldP->data +
                y1 * tiP->input_worldP->rowbytes +
                x1 * sizeof(PixelT));
            PixelT p12 = *reinterpret_cast<PixelT*>((char*)tiP->input_worldP->data +
                y1 * tiP->input_worldP->rowbytes +
                x2 * sizeof(PixelT));
            PixelT p21 = *reinterpret_cast<PixelT*>((char*)tiP->input_worldP->data +
                y2 * tiP->input_worldP->rowbytes +
                x1 * sizeof(PixelT));
            PixelT p22 = *reinterpret_cast<PixelT*>((char*)tiP->input_worldP->data +
                y2 * tiP->input_worldP->rowbytes +
                x2 * sizeof(PixelT));

            if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
                transformedPixel.red = static_cast<A_u_char>(
                    (1 - fx) * (1 - fy) * p11.red + fx * (1 - fy) * p12.red + (1 - fx) * fy * p21.red + fx * fy * p22.red);
                transformedPixel.green = static_cast<A_u_char>(
                    (1 - fx) * (1 - fy) * p11.green + fx * (1 - fy) * p12.green + (1 - fx) * fy * p21.green + fx * fy * p22.green);
                transformedPixel.blue = static_cast<A_u_char>(
                    (1 - fx) * (1 - fy) * p11.blue + fx * (1 - fy) * p12.blue + (1 - fx) * fy * p21.blue + fx * fy * p22.blue);
                transformedPixel.alpha = static_cast<A_u_char>(
                    (1 - fx) * (1 - fy) * p11.alpha + fx * (1 - fy) * p12.alpha + (1 - fx) * fy * p21.alpha + fx * fy * p22.alpha);
            }
            else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
                transformedPixel.red = static_cast<A_u_short>(
                    (1 - fx) * (1 - fy) * p11.red + fx * (1 - fy) * p12.red + (1 - fx) * fy * p21.red + fx * fy * p22.red);
                transformedPixel.green = static_cast<A_u_short>(
                    (1 - fx) * (1 - fy) * p11.green + fx * (1 - fy) * p12.green + (1 - fx) * fy * p21.green + fx * fy * p22.green);
                transformedPixel.blue = static_cast<A_u_short>(
                    (1 - fx) * (1 - fy) * p11.blue + fx * (1 - fy) * p12.blue + (1 - fx) * fy * p21.blue + fx * fy * p22.blue);
                transformedPixel.alpha = static_cast<A_u_short>(
                    (1 - fx) * (1 - fy) * p11.alpha + fx * (1 - fy) * p12.alpha + (1 - fx) * fy * p21.alpha + fx * fy * p22.alpha);
            }
            else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
                transformedPixel.red =
                    (1 - fx) * (1 - fy) * p11.red + fx * (1 - fy) * p12.red + (1 - fx) * fy * p21.red + fx * fy * p22.red;
                transformedPixel.green =
                    (1 - fx) * (1 - fy) * p11.green + fx * (1 - fy) * p12.green + (1 - fx) * fy * p21.green + fx * fy * p22.green;
                transformedPixel.blue =
                    (1 - fx) * (1 - fy) * p11.blue + fx * (1 - fy) * p12.blue + (1 - fx) * fy * p21.blue + fx * fy * p22.blue;
                transformedPixel.alpha =
                    (1 - fx) * (1 - fy) * p11.alpha + fx * (1 - fy) * p12.alpha + (1 - fx) * fy * p21.alpha + fx * fy * p22.alpha;
            }
        }
    }

    if (tiP->maskToLayer) {
        if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
            float baseA = basePixel.alpha / 255.0f;
            float fillF = tiP->fill;
            float alphaF = tiP->alpha;

            outP->red = static_cast<A_u_char>(basePixel.red * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF / 255.0f) +
                transformedPixel.red * baseA * alphaF);
            outP->green = static_cast<A_u_char>(basePixel.green * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF / 255.0f) +
                transformedPixel.green * baseA * alphaF);
            outP->blue = static_cast<A_u_char>(basePixel.blue * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF / 255.0f) +
                transformedPixel.blue * baseA * alphaF);
            outP->alpha = static_cast<A_u_char>(basePixel.alpha);
        }
        else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
            float baseA = basePixel.alpha / 32768.0f;
            float fillF = tiP->fill;
            float alphaF = tiP->alpha;

            outP->red = static_cast<A_u_short>(basePixel.red * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF / 32768.0f) +
                transformedPixel.red * baseA * alphaF);
            outP->green = static_cast<A_u_short>(basePixel.green * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF / 32768.0f) +
                transformedPixel.green * baseA * alphaF);
            outP->blue = static_cast<A_u_short>(basePixel.blue * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF / 32768.0f) +
                transformedPixel.blue * baseA * alphaF);
            outP->alpha = static_cast<A_u_short>(basePixel.alpha);
        }
        else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
            float baseA = basePixel.alpha;
            float fillF = tiP->fill;
            float alphaF = tiP->alpha;

            outP->red = basePixel.red * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF) +
                transformedPixel.red * baseA * alphaF;
            outP->green = basePixel.green * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF) +
                transformedPixel.green * baseA * alphaF;
            outP->blue = basePixel.blue * fillF * (1.0f - transformedPixel.alpha * baseA * alphaF) +
                transformedPixel.blue * baseA * alphaF;
            outP->alpha = basePixel.alpha;
        }
    }
    else if (tiP->fill > 0.0001f) {
        if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
            float fillF = tiP->fill;
            float alphaF = tiP->alpha;

            outP->red = static_cast<A_u_char>(basePixel.red * fillF * (1.0f - transformedPixel.alpha * alphaF / 255.0f) +
                transformedPixel.red * alphaF);
            outP->green = static_cast<A_u_char>(basePixel.green * fillF * (1.0f - transformedPixel.alpha * alphaF / 255.0f) +
                transformedPixel.green * alphaF);
            outP->blue = static_cast<A_u_char>(basePixel.blue * fillF * (1.0f - transformedPixel.alpha * alphaF / 255.0f) +
                transformedPixel.blue * alphaF);
            outP->alpha = static_cast<A_u_char>(basePixel.alpha * fillF * (1.0f - transformedPixel.alpha * alphaF / 255.0f) +
                transformedPixel.alpha * alphaF);
        }
        else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
            float fillF = tiP->fill;
            float alphaF = tiP->alpha;

            outP->red = static_cast<A_u_short>(basePixel.red * fillF * (1.0f - transformedPixel.alpha * alphaF / 32768.0f) +
                transformedPixel.red * alphaF);
            outP->green = static_cast<A_u_short>(basePixel.green * fillF * (1.0f - transformedPixel.alpha * alphaF / 32768.0f) +
                transformedPixel.green * alphaF);
            outP->blue = static_cast<A_u_short>(basePixel.blue * fillF * (1.0f - transformedPixel.alpha * alphaF / 32768.0f) +
                transformedPixel.blue * alphaF);
            outP->alpha = static_cast<A_u_short>(basePixel.alpha * fillF * (1.0f - transformedPixel.alpha * alphaF / 32768.0f) +
                transformedPixel.alpha * alphaF);
        }
        else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
            float fillF = tiP->fill;
            float alphaF = tiP->alpha;

            outP->red = basePixel.red * fillF * (1.0f - transformedPixel.alpha * alphaF) +
                transformedPixel.red * alphaF;
            outP->green = basePixel.green * fillF * (1.0f - transformedPixel.alpha * alphaF) +
                transformedPixel.green * alphaF;
            outP->blue = basePixel.blue * fillF * (1.0f - transformedPixel.alpha * alphaF) +
                transformedPixel.blue * alphaF;
            outP->alpha = basePixel.alpha * fillF * (1.0f - transformedPixel.alpha * alphaF) +
                transformedPixel.alpha * alphaF;
        }
    }
    else {
        if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
            float alphaF = tiP->alpha;
            outP->red = static_cast<A_u_char>(transformedPixel.red * alphaF);
            outP->green = static_cast<A_u_char>(transformedPixel.green * alphaF);
            outP->blue = static_cast<A_u_char>(transformedPixel.blue * alphaF);
            outP->alpha = static_cast<A_u_char>(transformedPixel.alpha * alphaF);
        }
        else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
            float alphaF = tiP->alpha;
            outP->red = static_cast<A_u_short>(transformedPixel.red * alphaF);
            outP->green = static_cast<A_u_short>(transformedPixel.green * alphaF);
            outP->blue = static_cast<A_u_short>(transformedPixel.blue * alphaF);
            outP->alpha = static_cast<A_u_short>(transformedPixel.alpha * alphaF);
        }
        else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
            float alphaF = tiP->alpha;
            outP->red = transformedPixel.red * alphaF;
            outP->green = transformedPixel.green * alphaF;
            outP->blue = transformedPixel.blue * alphaF;
            outP->alpha = transformedPixel.alpha * alphaF;
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
    AEGP_SuiteHandler suites(in_dataP->pica_basicP);

    TransformInfo tiP;
    AEFX_CLR_STRUCT(tiP);
    A_long linesL = 0;

    tiP.scale = params[TRANSFORM_SCALE]->u.fs_d.value;
    tiP.angle = ((float)params[TRANSFORM_ANGLE]->u.ad.value) / 65536.0f;
    tiP.offset_x = ((float)params[TRANSFORM_OFFSET]->u.td.x_value) / 65536.0f;
    tiP.offset_y = ((float)params[TRANSFORM_OFFSET]->u.td.y_value) / 65536.0f;
    tiP.maskToLayer = params[TRANSFORM_MASK_TO_LAYER]->u.bd.value;
    tiP.alpha = params[TRANSFORM_ALPHA]->u.fs_d.value;
    tiP.fill = params[TRANSFORM_FILL]->u.fs_d.value;
    tiP.sample = params[TRANSFORM_SAMPLE]->u.pd.value;
    tiP.x_tiles = params[TRANSFORM_X_TILES]->u.bd.value;
    tiP.y_tiles = params[TRANSFORM_Y_TILES]->u.bd.value;
    tiP.mirror = params[TRANSFORM_MIRROR]->u.bd.value;

    tiP.input_worldP = &params[TRANSFORM_INPUT]->u.ld;
    tiP.input_width = params[TRANSFORM_INPUT]->u.ld.width;
    tiP.input_height = params[TRANSFORM_INPUT]->u.ld.height;

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
            &params[TRANSFORM_INPUT]->u.ld,  
            NULL,                             
            (void*)&tiP,                      
            (PF_IteratePixelFloatFunc)RasterTransformFunc<PF_PixelFloat>,     
            output));
        break;

    case PF_PixelFormat_ARGB64:
        ERR(suites.Iterate16Suite1()->iterate(
            in_dataP,
            0,                            
            linesL,                       
            &params[TRANSFORM_INPUT]->u.ld,  
            NULL,                             
            (void*)&tiP,                      
            RasterTransformFunc<PF_Pixel16>,        
            output));
        break;

    case PF_PixelFormat_ARGB32:
    default:
        ERR(suites.Iterate8Suite1()->iterate(
            in_dataP,
            0,                            
            linesL,                       
            &params[TRANSFORM_INPUT]->u.ld,  
            NULL,                             
            (void*)&tiP,                      
            RasterTransformFunc<PF_Pixel8>,         
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
        TransformInfo* infoP = reinterpret_cast<TransformInfo*>(pre_render_dataPV);
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

    TransformInfo* infoP = reinterpret_cast<TransformInfo*>(malloc(sizeof(TransformInfo)));

    if (infoP) {
        PF_ParamDef cur_param;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_SCALE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->scale = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_ANGLE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->angle = ((float)cur_param.u.ad.value) / 65536.0f;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_OFFSET, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->offset_x = ((float)cur_param.u.td.x_value) / 65536.0f;
        infoP->offset_y = ((float)cur_param.u.td.y_value) / 65536.0f;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_MASK_TO_LAYER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->maskToLayer = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_ALPHA, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->alpha = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_FILL, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->fill = cur_param.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_SAMPLE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->sample = cur_param.u.pd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->x_tiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->y_tiles = cur_param.u.bd.value;

        ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORM_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mirror = cur_param.u.bd.value;

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            TRANSFORM_INPUT,
            TRANSFORM_INPUT,
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
    TransformInfo* infoP)
{
    PF_Err err = PF_Err_NONE;

    infoP->input_worldP = input_worldP;
    infoP->input_width = input_worldP->width;
    infoP->input_height = input_worldP->height;

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
                RasterTransformFunc<PF_PixelFloat>,
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
                RasterTransformFunc<PF_Pixel16>,
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
                RasterTransformFunc<PF_Pixel8>,
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
    TransformInfo* infoP)
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

    TransformParams transform_params;

    transform_params.mWidth = input_worldP->width;
    transform_params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    transform_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    transform_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    transform_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    transform_params.mScale = infoP->scale;
    transform_params.mAngle = infoP->angle;
    transform_params.mOffsetX = infoP->offset_x;
    transform_params.mOffsetY = infoP->offset_y;
    transform_params.mMaskToLayer = infoP->maskToLayer;
    transform_params.mAlpha = infoP->alpha;
    transform_params.mFill = infoP->fill;
    transform_params.mSample = infoP->sample;
    transform_params.mXTiles = infoP->x_tiles;
    transform_params.mYTiles = infoP->y_tiles;
    transform_params.mMirror = infoP->mirror;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mScale));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mAngle));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mOffsetX));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mOffsetY));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mMaskToLayer));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mAlpha));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mFill));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mSample));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mXTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mYTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mMirror));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(transform_params.mWidth, threadBlock[0]), RoundUp(transform_params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->transform_kernel,
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
        RasterTransform_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            transform_params.mSrcPitch,
            transform_params.mDstPitch,
            transform_params.m16f,
            transform_params.mWidth,
            transform_params.mHeight,
            transform_params.mScale,
            transform_params.mAngle,
            transform_params.mOffsetX,
            transform_params.mOffsetY,
            transform_params.mMaskToLayer,
            transform_params.mAlpha,
            transform_params.mFill,
            transform_params.mSample,
            transform_params.mXTiles,
            transform_params.mYTiles,
            transform_params.mMirror);

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
            dx_gpu_data->mTransformShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&transform_params, sizeof(TransformParams)));
        DX_ERR(shaderExecution.SetUnorderedAccessView(
            (ID3D12Resource*)dst_mem,
            transform_params.mHeight * dst_row_bytes));
        DX_ERR(shaderExecution.SetShaderResourceView(
            (ID3D12Resource*)src_mem,
            transform_params.mHeight * src_row_bytes));
        DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(transform_params.mWidth, 16), (UINT)DivideRoundUp(transform_params.mHeight, 16)));
    }
#endif
#if HAS_METAL
    else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        PF_Handle metal_handle = (PF_Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLBuffer> transform_param_buffer = [[device newBufferWithBytes : &transform_params
            length : sizeof(TransformParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->transform_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(transform_params.mWidth, threadsPerGroup.width),
                                   DivideRoundUp(transform_params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->transform_pipeline] ;
        [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
        [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
        [computeEncoder setBuffer : transform_param_buffer offset : 0 atIndex : 2] ;
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

    TransformInfo* infoP = reinterpret_cast<TransformInfo*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, TRANSFORM_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, TRANSFORM_INPUT));
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
        "Raster Transform",           
        "DKT Raster Transform",        
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