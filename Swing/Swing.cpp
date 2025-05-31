#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "Swing.h"
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

#if HAS_HLSL
inline PF_Err DXErr(bool inSuccess) {
    if (inSuccess) { return PF_Err_NONE; }
    else { return PF_Err_INTERNAL_STRUCT_DAMAGED; }
}
#define DX_ERR(FUNC) ERR(DXErr(FUNC))
#endif


extern void Swing_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float frequency,
    float angle1,
    float angle2,
    float phase,
    float time,
    int waveType,
    int xTiles,
    int yTiles,
    int mirror,
    float accumulatedPhase,
    int hasFrequencyKeyframes,
    int normalEnabled,
    int compatibilityEnabled,
    float compatFrequency,
    float compatAngle1,
    float compatAngle2,
    float compatPhase,
    int compatWaveType);

static double TriangleWave(double t)
{
    t = fmod(t + 0.75, 1.0);

    if (t < 0)
        t += 1.0;

    return (fabs(t - 0.5) - 0.25) * 4.0;
}

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
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
        PF_OutFlag2_I_MIX_GUID_DEPENDENCIES;

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
    PF_ADD_CHECKBOX("Normal",
        "",
        TRUE,     
        0,
        NORMAL_CHECKBOX_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Frequency",
        0.1f,     
        16.0f,    
        0.1f,     
        16.0f,    
        2.0f,     
        PF_Precision_HUNDREDTHS,   
        0,         
        0,        
        FREQ_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Angle 1",
        -3600.0f,   
        3600.0f,    
        -360.0f,    
        360.0f,     
        -30.0f,     
        PF_Precision_TENTHS,   
        0,           
        0,          
        ANGLE1_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Angle 2",
        -3600.0f,   
        3600.0f,    
        -360.0f,    
        360.0f,     
        30.0f,      
        PF_Precision_TENTHS,   
        0,           
        0,          
        ANGLE2_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Phase",
        0.0f,     
        2.0f,     
        0.0f,     
        2.0f,     
        0.0f,     
        PF_Precision_HUNDREDTHS,   
        0,         
        0,        
        PHASE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Wave",
        2,                     
        1,                   
        "Sine|Triangle",     
        0,                   
        WAVE_TYPE_DISK_ID);

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
        X_TILES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Y Tiles",
        "",
        FALSE,
        0,
        Y_TILES_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Mirror",
        "",
        FALSE,
        0,
        MIRROR_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    PF_ADD_PARAM(in_data, -1, &def);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_START;
    PF_STRCPY(def.name, "Compatibility");
    def.flags = PF_ParamFlag_START_COLLAPSED;
    PF_ADD_PARAM(in_data, -1, &def);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Compatibility",
        "",
        FALSE,
        0,
        COMPATIBILITY_CHECKBOX_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Frequency",
        0.1f,
        16.0f,
        0.1f,
        16.0f,
        2.0f,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        COMPATIBILITY_FREQUENCY_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Angle 1",
        -180.0f,
        180.0f,
        -180.0f,
        180.0f,
        -30.0f,
        PF_Precision_TENTHS,
        0,
        0,
        COMPATIBILITY_ANGLE1_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Angle 2",
        -180.0f,
        180.0f,
        -180.0f,
        180.0f,
        30.0f,
        PF_Precision_TENTHS,
        0,
        0,
        COMPATIBILITY_ANGLE2_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Phase",
        0.0f,
        2.0f,
        0.0f,
        2.0f,
        0.0f,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        COMPATIBILITY_PHASE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Wave",
        2,
        1,
        "Sine|Triangle",
        0,
        COMPATIBILITY_WAVE_TYPE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    def.param_type = PF_Param_GROUP_END;
    PF_ADD_PARAM(in_data, -1, &def);

    out_data->num_params = SWING_NUM_PARAMS;

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

        size_t sizes[] = { strlen(k16fString), strlen(kSwingKernel_OpenCLString) };
        char const* strings[] = { k16fString, kSwingKernel_OpenCLString };

        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->swing_kernel = clCreateKernel(program, "SwingKernel", &result);
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
        dx_gpu_data->mSwingShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"SwingKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mSwingShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kSwingKernel_MetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> swing_function = nil;
            NSString* swing_name = [NSString stringWithCString : "SwingKernel" encoding : NSUTF8StringEncoding];

            swing_function = [[library newFunctionWithName : swing_name]autorelease];

            if (!swing_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->swing_pipeline = [device newComputePipelineStateWithFunction : swing_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->swing_kernel);

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
        dx_gpu_dataP->mSwingShader.reset();

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

        [metal_dataP->swing_pipeline release] ;

        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
            kPFHandleSuite,
            kPFHandleSuiteVersion1,
            out_dataP);

        handle_suite->host_dispose_handle(gpu_dataH);
    }
#endif
    return err;
}

static bool HasAnyFrequencyKeyframes(PF_InData* in_data)
{
    PF_Err err = PF_Err_NONE;
    bool has_keyframes = false;

    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_EffectRefH effect_ref = NULL;
    AEGP_StreamRefH stream_ref = NULL;
    A_long num_keyframes = 0;

    if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
        AEGP_EffectRefH aegp_effect_ref = NULL;
        err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(NULL, in_data->effect_ref, &aegp_effect_ref);

        if (!err && aegp_effect_ref) {
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL,
                aegp_effect_ref,
                SWING_FREQ,
                &stream_ref);

            if (!err && stream_ref) {
                err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(stream_ref, &num_keyframes);

                if (!err && num_keyframes > 0) {
                    has_keyframes = true;
                }

                suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
            }

            suites.EffectSuite4()->AEGP_DisposeEffect(aegp_effect_ref);
        }
    }

    return has_keyframes;
}


PF_Err valueAtTime(
    PF_InData* in_data,
    int stream_index,
    float time_secs,
    PF_FpLong* value_out)
{
    PF_Err err = PF_Err_NONE;

    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_EffectRefH aegp_effect_ref = NULL;
    AEGP_StreamRefH stream_ref = NULL;

    A_Time time;
    time.value = (A_long)(time_secs * in_data->time_scale);
    time.scale = in_data->time_scale;

    if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
        err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(
            NULL,
            in_data->effect_ref,
            &aegp_effect_ref);

        if (!err && aegp_effect_ref) {
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(
                NULL,
                aegp_effect_ref,
                stream_index,
                &stream_ref);

            if (!err && stream_ref) {
                AEGP_StreamValue2 stream_value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(
                    NULL,
                    stream_ref,
                    AEGP_LTimeMode_LayerTime,
                    &time,
                    FALSE,
                    &stream_value);

                if (!err) {
                    AEGP_StreamType stream_type;
                    err = suites.StreamSuite5()->AEGP_GetStreamType(stream_ref, &stream_type);

                    if (!err) {
                        switch (stream_type) {
                        case AEGP_StreamType_OneD:
                            *value_out = stream_value.val.one_d;
                            break;

                        case AEGP_StreamType_TwoD:
                        case AEGP_StreamType_TwoD_SPATIAL:
                            *value_out = stream_value.val.two_d.x;
                            break;

                        case AEGP_StreamType_ThreeD:
                        case AEGP_StreamType_ThreeD_SPATIAL:
                            *value_out = stream_value.val.three_d.x;
                            break;

                        case AEGP_StreamType_COLOR:
                            *value_out = stream_value.val.color.redF;
                            break;

                        default:
                            err = PF_Err_BAD_CALLBACK_PARAM;
                            break;
                        }
                    }

                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
                }

                suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
            }

            suites.EffectSuite4()->AEGP_DisposeEffect(aegp_effect_ref);
        }
    }

    return err;
}

static PF_Err
valueAtTimeHz(
    PF_InData* in_data,
    int stream_index,
    float time_secs,
    float duration,
    SwingParams* infoP,
    PF_FpLong* value_out)
{
    PF_Err err = PF_Err_NONE;

    err = valueAtTime(in_data, stream_index, time_secs, value_out);
    if (err) return err;

    if (stream_index == SWING_FREQ) {
        bool isKeyed = HasAnyFrequencyKeyframes(in_data);
        bool isHz = true;

        if (isHz && isKeyed) {
            float fps = 120.0f;
            int totalSteps = (int)roundf(duration * fps);
            int curSteps = (int)roundf(fps * time_secs);

            if (!infoP->accumulated_phase_initialized) {
                infoP->accumulated_phase = 0.0f;
                infoP->accumulated_phase_initialized = true;
            }

            if (curSteps >= 0) {
                infoP->accumulated_phase = 0.0f;

                for (int i = 0; i <= curSteps; i++) {
                    PF_FpLong stepValue;
                    float adjusted_time = i / fps + infoP->layer_start_seconds;
                    err = valueAtTime(in_data, stream_index, adjusted_time, &stepValue);
                    if (err) return err;

                    infoP->accumulated_phase += stepValue / fps;
                }

                *value_out = infoP->accumulated_phase;
            }
        }
    }

    return err;
}


template <typename PixelType>
static void ProcessSwingEffect(
    PF_InData* in_data,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    double frequency,
    double angle1,
    double angle2,
    double phase,
    A_long waveType,
    bool x_tiles,
    bool y_tiles,
    bool mirror,
    double current_time,
    double accumulated_phase = 0.0,
    bool normal_enabled = true,
    bool compatibility_enabled = false,
    double compat_frequency = 0.0,
    double compat_angle1 = 0.0,
    double compat_angle2 = 0.0,
    double compat_phase = 0.0,
    A_long compat_wave_type = 0)
{
    bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

    if ((normal_enabled && compatibility_enabled) || (!normal_enabled && !compatibility_enabled)) {
        for (A_long y = 0; y < output_worldP->height; y++) {
            PixelType* srcRow = nullptr;
            PixelType* dstRow = nullptr;

            if constexpr (std::is_same_v<PixelType, PF_Pixel8>) {
                srcRow = (PixelType*)((char*)input_worldP->data + y * input_worldP->rowbytes);
                dstRow = (PixelType*)((char*)output_worldP->data + y * output_worldP->rowbytes);
            }
            else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
                srcRow = (PixelType*)((char*)input_worldP->data + y * input_worldP->rowbytes);
                dstRow = (PixelType*)((char*)output_worldP->data + y * output_worldP->rowbytes);
            }
            else {  
                srcRow = (PixelType*)((char*)input_worldP->data + y * input_worldP->rowbytes);
                dstRow = (PixelType*)((char*)output_worldP->data + y * output_worldP->rowbytes);
            }

            memcpy(dstRow, srcRow, output_worldP->width * sizeof(PixelType));
        }
        return;
    }

    double effectivePhase;
    double m;
    double angleRad;

    if (compatibility_enabled) {
        if (compat_wave_type == 0) {  
            m = sin(((current_time * compat_frequency) + compat_phase) * M_PI);
        }
        else {  
            m = TriangleWave(((current_time * compat_frequency) + compat_phase) / 2.0);
        }

        double finalAngle = ((compat_angle2 - compat_angle1) * ((m + 1.0) / 2.0)) + compat_angle1;
        angleRad = -finalAngle * M_PI / 180.0;
    }
    else {
        if (has_frequency_keyframes && accumulated_phase > 0.0) {
            effectivePhase = phase + accumulated_phase;

            if (waveType == 0) {
                m = sin(effectivePhase * M_PI);
            }
            else {
                m = TriangleWave(effectivePhase / 2.0);
            }
        }
        else {
            effectivePhase = phase + (current_time * frequency);

            if (waveType == 0) {
                m = sin(effectivePhase * M_PI);
            }
            else {
                m = TriangleWave(effectivePhase / 2.0);
            }
        }

        double t = (m + 1.0) / 2.0;
        double finalAngle = -(angle1 + t * (angle2 - angle1));
        angleRad = finalAngle * M_PI / 180.0;
    }

    float centerX = input_worldP->width / 2.0f;
    float centerY = input_worldP->height / 2.0f;

    float cos_rot = (float)cos(angleRad);
    float sin_rot = (float)sin(angleRad);

    for (A_long y = 0; y < output_worldP->height; y++) {
        for (A_long x = 0; x < output_worldP->width; x++) {
            float dx = (x - centerX);
            float dy = (y - centerY);

            float rotated_x = (dx * cos_rot - dy * sin_rot) + centerX;
            float rotated_y = (dx * sin_rot + dy * cos_rot) + centerY;

            float u = rotated_x / (float)input_worldP->width;
            float v = rotated_y / (float)input_worldP->height;

            bool outsideBounds = false;

            if (x_tiles) {
                if (mirror) {
                    float fracPart = (float)fmod(fabs(u), 1.0f);
                    int isOdd = (int)floor(fabs(u)) & 1;
                    u = isOdd ? 1.0f - fracPart : fracPart;
                }
                else {
                    u = u - floor(u);
                }
            }
            else if (u < 0.0f || u >= 1.0f) {
                outsideBounds = true;
            }

            if (y_tiles) {
                if (mirror) {
                    float fracPart = (float)fmod(fabs(v), 1.0f);
                    int isOdd = (int)floor(fabs(v)) & 1;
                    v = isOdd ? 1.0f - fracPart : fracPart;
                }
                else {
                    v = v - floor(v);
                }
            }
            else if (v < 0.0f || v >= 1.0f) {
                outsideBounds = true;
            }

            PixelType* outPixel = nullptr;
            if (sizeof(PixelType) == sizeof(PF_Pixel8)) {
                PF_Pixel8* outRow = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                outPixel = reinterpret_cast<PixelType*>(&outRow[x]);
            }
            else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
                PF_Pixel16* outRow = (PF_Pixel16*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                outPixel = reinterpret_cast<PixelType*>(&outRow[x]);
            }
            else {   
                PF_PixelFloat* outRow = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                outPixel = reinterpret_cast<PixelType*>(&outRow[x]);
            }

            if (outsideBounds) {
                if (sizeof(PixelType) == sizeof(PF_Pixel8)) {
                    PF_Pixel8* out8 = reinterpret_cast<PF_Pixel8*>(outPixel);
                    out8->alpha = 0;
                    out8->red = out8->green = out8->blue = 0;
                }
                else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
                    PF_Pixel16* out16 = reinterpret_cast<PF_Pixel16*>(outPixel);
                    out16->alpha = 0;
                    out16->red = out16->green = out16->blue = 0;
                }
                else {   
                    PF_PixelFloat* outF = reinterpret_cast<PF_PixelFloat*>(outPixel);
                    outF->alpha = 0.0f;
                    outF->red = outF->green = outF->blue = 0.0f;
                }
            }
            else {
                u = fmaxf(0.0f, fminf(0.9999f, u));
                v = fmaxf(0.0f, fminf(0.9999f, v));

                float x_sample = u * input_worldP->width;
                float y_sample = v * input_worldP->height;

                int x0 = (int)floor(x_sample);
                int y0 = (int)floor(y_sample);
                int x1 = x0 + 1;
                int y1 = y0 + 1;

                x0 = MAX(0, MIN(input_worldP->width - 1, x0));
                y0 = MAX(0, MIN(input_worldP->height - 1, y0));
                x1 = MAX(0, MIN(input_worldP->width - 1, x1));
                y1 = MAX(0, MIN(input_worldP->height - 1, y1));

                float fx = x_sample - x0;
                float fy = y_sample - y0;

                PixelType p00, p10, p01, p11;
                if (sizeof(PixelType) == sizeof(PF_Pixel8)) {
                    PF_Pixel8* base = (PF_Pixel8*)input_worldP->data;
                    PF_Pixel8* row0 = (PF_Pixel8*)((char*)base + y0 * input_worldP->rowbytes);
                    PF_Pixel8* row1 = (PF_Pixel8*)((char*)base + y1 * input_worldP->rowbytes);

                    p00 = *reinterpret_cast<PixelType*>(&row0[x0]);
                    p10 = *reinterpret_cast<PixelType*>(&row0[x1]);
                    p01 = *reinterpret_cast<PixelType*>(&row1[x0]);
                    p11 = *reinterpret_cast<PixelType*>(&row1[x1]);
                }
                else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
                    PF_Pixel16* base = (PF_Pixel16*)input_worldP->data;
                    PF_Pixel16* row0 = (PF_Pixel16*)((char*)base + y0 * input_worldP->rowbytes);
                    PF_Pixel16* row1 = (PF_Pixel16*)((char*)base + y1 * input_worldP->rowbytes);

                    p00 = *reinterpret_cast<PixelType*>(&row0[x0]);
                    p10 = *reinterpret_cast<PixelType*>(&row0[x1]);
                    p01 = *reinterpret_cast<PixelType*>(&row1[x0]);
                    p11 = *reinterpret_cast<PixelType*>(&row1[x1]);
                }
                else {   
                    PF_PixelFloat* base = (PF_PixelFloat*)input_worldP->data;
                    PF_PixelFloat* row0 = (PF_PixelFloat*)((char*)base + y0 * input_worldP->rowbytes);
                    PF_PixelFloat* row1 = (PF_PixelFloat*)((char*)base + y1 * input_worldP->rowbytes);

                    p00 = *reinterpret_cast<PixelType*>(&row0[x0]);
                    p10 = *reinterpret_cast<PixelType*>(&row0[x1]);
                    p01 = *reinterpret_cast<PixelType*>(&row1[x0]);
                    p11 = *reinterpret_cast<PixelType*>(&row1[x1]);
                }

                if (sizeof(PixelType) == sizeof(PF_Pixel8)) {
                    PF_Pixel8* p00_8 = reinterpret_cast<PF_Pixel8*>(&p00);
                    PF_Pixel8* p10_8 = reinterpret_cast<PF_Pixel8*>(&p10);
                    PF_Pixel8* p01_8 = reinterpret_cast<PF_Pixel8*>(&p01);
                    PF_Pixel8* p11_8 = reinterpret_cast<PF_Pixel8*>(&p11);
                    PF_Pixel8* out_8 = reinterpret_cast<PF_Pixel8*>(outPixel);

                    float oneMinusFx = 1.0f - fx;
                    float oneMinusFy = 1.0f - fy;

                    float w00 = oneMinusFx * oneMinusFy;
                    float w10 = fx * oneMinusFy;
                    float w01 = oneMinusFx * fy;
                    float w11 = fx * fy;

                    out_8->alpha = (A_u_char)(p00_8->alpha * w00 + p10_8->alpha * w10 + p01_8->alpha * w01 + p11_8->alpha * w11);
                    out_8->red = (A_u_char)(p00_8->red * w00 + p10_8->red * w10 + p01_8->red * w01 + p11_8->red * w11);
                    out_8->green = (A_u_char)(p00_8->green * w00 + p10_8->green * w10 + p01_8->green * w01 + p11_8->green * w11);
                    out_8->blue = (A_u_char)(p00_8->blue * w00 + p10_8->blue * w10 + p01_8->blue * w01 + p11_8->blue * w11);
                }
                else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
                    PF_Pixel16* p00_16 = reinterpret_cast<PF_Pixel16*>(&p00);
                    PF_Pixel16* p10_16 = reinterpret_cast<PF_Pixel16*>(&p10);
                    PF_Pixel16* p01_16 = reinterpret_cast<PF_Pixel16*>(&p01);
                    PF_Pixel16* p11_16 = reinterpret_cast<PF_Pixel16*>(&p11);
                    PF_Pixel16* out_16 = reinterpret_cast<PF_Pixel16*>(outPixel);

                    float oneMinusFx = 1.0f - fx;
                    float oneMinusFy = 1.0f - fy;

                    float w00 = oneMinusFx * oneMinusFy;
                    float w10 = fx * oneMinusFy;
                    float w01 = oneMinusFx * fy;
                    float w11 = fx * fy;

                    out_16->alpha = (A_u_short)(p00_16->alpha * w00 + p10_16->alpha * w10 + p01_16->alpha * w01 + p11_16->alpha * w11);
                    out_16->red = (A_u_short)(p00_16->red * w00 + p10_16->red * w10 + p01_16->red * w01 + p11_16->red * w11);
                    out_16->green = (A_u_short)(p00_16->green * w00 + p10_16->green * w10 + p01_16->green * w01 + p11_16->green * w11);
                    out_16->blue = (A_u_short)(p00_16->blue * w00 + p10_16->blue * w10 + p01_16->blue * w01 + p11_16->blue * w11);
                }
                else {   
                    PF_PixelFloat* p00_f = reinterpret_cast<PF_PixelFloat*>(&p00);
                    PF_PixelFloat* p10_f = reinterpret_cast<PF_PixelFloat*>(&p10);
                    PF_PixelFloat* p01_f = reinterpret_cast<PF_PixelFloat*>(&p01);
                    PF_PixelFloat* p11_f = reinterpret_cast<PF_PixelFloat*>(&p11);
                    PF_PixelFloat* out_f = reinterpret_cast<PF_PixelFloat*>(outPixel);

                    float oneMinusFx = 1.0f - fx;
                    float oneMinusFy = 1.0f - fy;

                    float w00 = oneMinusFx * oneMinusFy;
                    float w10 = fx * oneMinusFy;
                    float w01 = oneMinusFx * fy;
                    float w11 = fx * fy;

                    out_f->alpha = p00_f->alpha * w00 + p10_f->alpha * w10 + p01_f->alpha * w01 + p11_f->alpha * w11;
                    out_f->red = p00_f->red * w00 + p10_f->red * w10 + p01_f->red * w01 + p11_f->red * w11;
                    out_f->green = p00_f->green * w00 + p10_f->green * w10 + p01_f->green * w01 + p11_f->green * w11;
                    out_f->blue = p00_f->blue * w00 + p10_f->blue * w10 + p01_f->blue * w01 + p11_f->blue * w11;
                }
            }
        }
    }
}

static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        SwingParams* infoP = reinterpret_cast<SwingParams*>(pre_render_dataPV);
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

    SwingParams* infoP = reinterpret_cast<SwingParams*>(malloc(sizeof(SwingParams)));

    if (infoP) {
        PF_ParamDef cur_param;

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_NORMAL_CHECKBOX, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        if (!err) infoP->normal_enabled = cur_param.u.bd.value;
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_FREQ, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->frequency = cur_param.u.fs_d.value;
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_ANGLE1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->angle1 = cur_param.u.fs_d.value;
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_ANGLE2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->angle2 = cur_param.u.fs_d.value;
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_PHASE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->phase = cur_param.u.fs_d.value;
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_WAVE_TYPE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->waveType = cur_param.u.pd.value - 1;
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->x_tiles = (cur_param.u.bd.value != 0);
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->y_tiles = (cur_param.u.bd.value != 0);
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        AEFX_CLR_STRUCT(cur_param);
        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        infoP->mirror = (cur_param.u.bd.value != 0);
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_COMPATIBILITY_CHECKBOX, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
        if (!err) infoP->compatibility_enabled = cur_param.u.bd.value;
        ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

        if (infoP->compatibility_enabled) {
            ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_COMPATIBILITY_FREQUENCY, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
            if (!err) infoP->compat_frequency = cur_param.u.fs_d.value;
            ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

            ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_COMPATIBILITY_ANGLE1, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
            if (!err) infoP->compat_angle1 = cur_param.u.fs_d.value;
            ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

            ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_COMPATIBILITY_ANGLE2, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
            if (!err) infoP->compat_angle2 = cur_param.u.fs_d.value;
            ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

            ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_COMPATIBILITY_PHASE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
            if (!err) infoP->compat_phase = cur_param.u.fs_d.value;
            ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));

            ERR(PF_CHECKOUT_PARAM(in_dataP, SWING_COMPATIBILITY_WAVE_TYPE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
            if (!err) infoP->compat_wave_type = cur_param.u.pd.value - 1;
            ERR(PF_CHECKIN_PARAM(in_dataP, &cur_param));
        }

        bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_dataP);

        AEGP_LayerIDVal layer_id = 0;
        AEGP_SuiteHandler suites(in_dataP->pica_basicP);

        PF_FpLong layer_time_offset = 0;
        A_Ratio stretch_factor = { 1, 1 };

        if (suites.PFInterfaceSuite1() && in_dataP->effect_ref) {
            AEGP_LayerH layer = NULL;

            err = suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_dataP->effect_ref, &layer);

            if (!err && layer) {
                err = suites.LayerSuite7()->AEGP_GetLayerID(layer, &layer_id);

                A_Time in_point;
                err = suites.LayerSuite7()->AEGP_GetLayerInPoint(layer, AEGP_LTimeMode_LayerTime, &in_point);

                if (!err) {
                    layer_time_offset = (PF_FpLong)in_point.value / (PF_FpLong)in_point.scale;
                }

                err = suites.LayerSuite7()->AEGP_GetLayerStretch(layer, &stretch_factor);
            }
        }

        PF_FpLong current_time = (PF_FpLong)in_dataP->current_time / (PF_FpLong)in_dataP->time_scale;
        PF_FpLong duration = current_time;

        PF_FpLong stretch_ratio = (PF_FpLong)stretch_factor.num / (PF_FpLong)stretch_factor.den;

        current_time -= layer_time_offset;
        duration -= layer_time_offset;

        current_time *= stretch_ratio;
        duration *= stretch_ratio;

        infoP->current_time = current_time;
        infoP->layer_start_seconds = layer_time_offset;

        infoP->accumulated_phase = 0.0f;
        infoP->accumulated_phase_initialized = false;

        if (has_frequency_keyframes && infoP->frequency > 0) {
            PF_FpLong accumulated_phase = 0.0;
            PF_FpLong value_out;
            err = valueAtTimeHz(in_dataP, SWING_FREQ, current_time, duration, infoP, &value_out);
            if (!err) {
                infoP->accumulated_phase = value_out;
                infoP->accumulated_phase_initialized = true;
            }
        }

        extraP->output->pre_render_data = infoP;
        extraP->output->delete_pre_render_data_func = DisposePreRenderData;

        ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
            SWING_INPUT,
            SWING_INPUT,
            &req,
            in_dataP->current_time,
            in_dataP->time_step,
            in_dataP->time_scale,
            &in_result));

        if (!err) {
            struct {
                A_u_char has_frequency_keyframes;
                A_long time_offset;
                AEGP_LayerIDVal layer_id;
                A_Ratio stretch_factor;
                SwingParams params;
            } detection_data;

            detection_data.has_frequency_keyframes = has_frequency_keyframes ? 1 : 0;
            detection_data.time_offset = 0;      
            detection_data.layer_id = layer_id;
            detection_data.stretch_factor = stretch_factor;
            detection_data.params = *infoP;

            ERR(extraP->cb->GuidMixInPtr(in_dataP->effect_ref, sizeof(detection_data), &detection_data));

            extraP->output->max_result_rect = in_result.max_result_rect;
            extraP->output->result_rect = in_result.result_rect;
            extraP->output->solid = FALSE;
        }
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
    SwingParams* infoP)
{
    PF_Err err = PF_Err_NONE;

    if (!err) {
        switch (pixel_format) {
        case PF_PixelFormat_ARGB128:
            ProcessSwingEffect<PF_PixelFloat>(
                in_data,
                input_worldP,
                output_worldP,
                infoP->frequency,
                infoP->angle1,
                infoP->angle2,
                infoP->phase,
                infoP->waveType,
                infoP->x_tiles,
                infoP->y_tiles,
                infoP->mirror,
                infoP->current_time,
                infoP->accumulated_phase,
                infoP->normal_enabled,
                infoP->compatibility_enabled,
                infoP->compat_frequency,
                infoP->compat_angle1,
                infoP->compat_angle2,
                infoP->compat_phase,
                infoP->compat_wave_type);
            break;

        case PF_PixelFormat_ARGB64:
            ProcessSwingEffect<PF_Pixel16>(
                in_data,
                input_worldP,
                output_worldP,
                infoP->frequency,
                infoP->angle1,
                infoP->angle2,
                infoP->phase,
                infoP->waveType,
                infoP->x_tiles,
                infoP->y_tiles,
                infoP->mirror,
                infoP->current_time,
                infoP->accumulated_phase,
                infoP->normal_enabled,
                infoP->compatibility_enabled,
                infoP->compat_frequency,
                infoP->compat_angle1,
                infoP->compat_angle2,
                infoP->compat_phase,
                infoP->compat_wave_type);
            break;

        case PF_PixelFormat_ARGB32:
            ProcessSwingEffect<PF_Pixel8>(
                in_data,
                input_worldP,
                output_worldP,
                infoP->frequency,
                infoP->angle1,
                infoP->angle2,
                infoP->phase,
                infoP->waveType,
                infoP->x_tiles,
                infoP->y_tiles,
                infoP->mirror,
                infoP->current_time,
                infoP->accumulated_phase,
                infoP->normal_enabled,
                infoP->compatibility_enabled,
                infoP->compat_frequency,
                infoP->compat_angle1,
                infoP->compat_angle2,
                infoP->compat_phase,
                infoP->compat_wave_type);
            break;

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
    float mFrequency;
    float mAngle1;
    float mAngle2;
    float mPhase;
    float mTime;
    int mWaveType;
    int mXTiles;
    int mYTiles;
    int mMirror;
    float mAccumulatedPhase;
    int mHasFrequencyKeyframes;
    int mNormalEnabled;
    int mCompatibilityEnabled;
    float mCompatFrequency;
    float mCompatAngle1;
    float mCompatAngle2;
    float mCompatPhase;
    int mCompatWaveType;
} SwingKernelParams;

static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    SwingParams* infoP)
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

    SwingKernelParams params;

    params.mWidth = input_worldP->width;
    params.mHeight = input_worldP->height;

    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;

    params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

    params.mFrequency = (float)infoP->frequency;
    params.mAngle1 = (float)infoP->angle1;
    params.mAngle2 = (float)infoP->angle2;
    params.mPhase = (float)infoP->phase;
    params.mTime = (float)infoP->current_time;
    params.mWaveType = infoP->waveType;
    params.mXTiles = infoP->x_tiles ? 1 : 0;
    params.mYTiles = infoP->y_tiles ? 1 : 0;
    params.mMirror = infoP->mirror ? 1 : 0;
    params.mAccumulatedPhase = (float)infoP->accumulated_phase;
    params.mHasFrequencyKeyframes = HasAnyFrequencyKeyframes(in_dataP) ? 1 : 0;

    params.mNormalEnabled = infoP->normal_enabled ? 1 : 0;
    params.mCompatibilityEnabled = infoP->compatibility_enabled ? 1 : 0;
    params.mCompatFrequency = (float)infoP->compat_frequency;
    params.mCompatAngle1 = (float)infoP->compat_angle1;
    params.mCompatAngle2 = (float)infoP->compat_angle2;
    params.mCompatPhase = (float)infoP->compat_phase;
    params.mCompatWaveType = infoP->compat_wave_type;

    if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        cl_uint param_index = 0;

        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mSrcPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mDstPitch));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.m16f));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mWidth));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mHeight));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mFrequency));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mAngle1));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mAngle2));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mPhase));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mTime));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mWaveType));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mXTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mYTiles));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mMirror));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mAccumulatedPhase));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mHasFrequencyKeyframes));

        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mNormalEnabled));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mCompatibilityEnabled));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mCompatFrequency));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mCompatAngle1));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mCompatAngle2));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(float), &params.mCompatPhase));
        CL_ERR(clSetKernelArg(cl_gpu_dataP->swing_kernel, param_index++, sizeof(int), &params.mCompatWaveType));

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };

        CL_ERR(clEnqueueNDRangeKernel(
            (cl_command_queue)device_info.command_queuePV,
            cl_gpu_dataP->swing_kernel,
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
        Swing_CUDA(
            (const float*)src_mem,
            (float*)dst_mem,
            params.mSrcPitch,
            params.mDstPitch,
            params.m16f,
            params.mWidth,
            params.mHeight,
            params.mFrequency,
            params.mAngle1,
            params.mAngle2,
            params.mPhase,
            params.mTime,
            params.mWaveType,
            params.mXTiles,
            params.mYTiles,
            params.mMirror,
            params.mAccumulatedPhase,
            params.mHasFrequencyKeyframes,
            params.mNormalEnabled,
            params.mCompatibilityEnabled,
            params.mCompatFrequency,
            params.mCompatAngle1,
            params.mCompatAngle2,
            params.mCompatPhase,
            params.mCompatWaveType);

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
            dx_gpu_data->mSwingShader,
            3);

        DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(SwingKernelParams)));
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
            length : sizeof(SwingKernelParams)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        MTLSize threadsPerGroup = { [metal_dataP->swing_pipeline threadExecutionWidth] , 16, 1 };
        MTLSize numThreadgroups = { DivideRoundUp(params.mWidth, threadsPerGroup.width),
                                  DivideRoundUp(params.mHeight, threadsPerGroup.height), 1 };

        [computeEncoder setComputePipelineState : metal_dataP->swing_pipeline] ;
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

    SwingParams* infoP = reinterpret_cast<SwingParams*>(extraP->input->pre_render_data);

    if (infoP) {
        ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, SWING_INPUT, &input_worldP)));
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
        ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, SWING_INPUT));
    }
    else {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
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

    try {
        double frequency = params[SWING_FREQ]->u.fs_d.value;
        double angle1 = params[SWING_ANGLE1]->u.fs_d.value;
        double angle2 = params[SWING_ANGLE2]->u.fs_d.value;
        double phase = params[SWING_PHASE]->u.fs_d.value;
        A_long waveType = params[SWING_WAVE_TYPE]->u.pd.value - 1;
        bool x_tiles = params[SWING_X_TILES]->u.bd.value != 0;
        bool y_tiles = params[SWING_Y_TILES]->u.bd.value != 0;
        bool mirror = params[SWING_MIRROR]->u.bd.value != 0;

        PF_EffectWorld* input_worldP = &params[SWING_INPUT]->u.ld;
        PF_EffectWorld* output_worldP = output;

        PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;

        bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

        PF_FpLong layer_time_offset = 0;
        A_Ratio stretch_factor = { 1, 1 };      

        AEGP_SuiteHandler suites(in_data->pica_basicP);
        if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
            AEGP_LayerH layer = NULL;

            err = suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_data->effect_ref, &layer);

            if (!err && layer) {
                A_Time in_point;
                err = suites.LayerSuite7()->AEGP_GetLayerInPoint(layer, AEGP_LTimeMode_LayerTime, &in_point);

                if (!err) {
                    layer_time_offset = (PF_FpLong)in_point.value / (PF_FpLong)in_point.scale;
                }

                err = suites.LayerSuite7()->AEGP_GetLayerStretch(layer, &stretch_factor);

                PF_FpLong stretch_ratio = (PF_FpLong)stretch_factor.num / (PF_FpLong)stretch_factor.den;

                A_Time comp_time;
                comp_time.value = in_data->current_time;
                comp_time.scale = in_data->time_scale;

                A_Time source_time;
                err = suites.LayerSuite7()->AEGP_ConvertCompToLayerTime(layer, &comp_time, &source_time);

                if (!err) {
                    current_time = (PF_FpLong)source_time.value / (PF_FpLong)source_time.scale;
                }
            }
        }

        if (has_frequency_keyframes) {
            PF_FpLong half_frame_seconds = (PF_FpLong)in_data->time_step / (PF_FpLong)(in_data->time_scale * 2);
            current_time += half_frame_seconds;
        }

        double bytesPerPixel = (double)input_worldP->rowbytes / (double)input_worldP->width;
        bool is32bit = (bytesPerPixel >= 16.0);        
        bool is16bit = (!is32bit && bytesPerPixel >= 8.0);       

        if (is32bit) {
            ProcessSwingEffect<PF_PixelFloat>(
                in_data,
                input_worldP,
                output_worldP,
                frequency,
                angle1,
                angle2,
                phase,
                waveType,
                x_tiles,
                y_tiles,
                mirror,
                current_time);      
        }
        else if (is16bit) {
            ProcessSwingEffect<PF_Pixel16>(
                in_data,
                input_worldP,
                output_worldP,
                frequency,
                angle1,
                angle2,
                phase,
                waveType,
                x_tiles,
                y_tiles,
                mirror,
                current_time);      
        }
        else {
            ProcessSwingEffect<PF_Pixel8>(
                in_data,
                input_worldP,
                output_worldP,
                frequency,
                angle1,
                angle2,
                phase,
                waveType,
                x_tiles,
                y_tiles,
                mirror,
                current_time);      
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
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
        "Swing",                   
        "DKT Swing",            
        "DKT Effects",         
        AE_RESERVED_INFO);      

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
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}

