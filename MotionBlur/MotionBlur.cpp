#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "MotionBlur.h"
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

extern void Motion_Blur_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float motionX,
    float motionY,
    float tuneValue,
    float downsampleX,
    float downsampleY);

extern void Scale_Blur_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float scaleVelocity,
    float anchorX,
    float anchorY,
    float tuneValue,
    float downsampleX,
    float downsampleY);

extern void Angle_Blur_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float rotationAngle,
    float anchorX,
    float anchorY,
    float tuneValue,
    float downsampleX,
    float downsampleY);

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
    PF_Err			err = PF_Err_NONE;
    PF_ParamDef		def;

    AEFX_CLR_STRUCT(def);

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(STR_TUNE_NAME,
        TUNE_MIN,
        TUNE_MAX,
        TUNE_MIN,
        TUNE_MAX,
        TUNE_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        TUNE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Position",
        "Position",
        TRUE,
        0,
        POSITION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Scale",
        "Scale",
        TRUE,
        0,
        SCALE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Angle",
        "Angle",
        TRUE,
        0,
        ANGLE_DISK_ID);

    out_data->num_params = MOTIONBLUR_NUM_PARAMS;

    return err;
}

static PF_Err GetLayerPosition(PF_InData* in_data, AEGP_LayerH layerH, const A_Time* time, AEGP_TwoDVal* position) {
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamSuite5* streamSuite = suites.StreamSuite5();
    if (!streamSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_StreamRefH posStreamH = NULL;
    err = streamSuite->AEGP_GetNewLayerStream(
        NULL,
        layerH,
        AEGP_LayerStream_POSITION,
        &posStreamH);

    if (err || !posStreamH) {
        return err;
    }

    AEGP_StreamValue2 streamValue;
    err = streamSuite->AEGP_GetNewStreamValue(
        NULL,
        posStreamH,
        AEGP_LTimeMode_LayerTime,
        time,
        false,
        &streamValue);

    if (!err) {
        AEGP_StreamType type;
        err = streamSuite->AEGP_GetStreamType(posStreamH, &type);

        if (!err) {
            if (type == AEGP_StreamType_ThreeD || type == AEGP_StreamType_ThreeD_SPATIAL) {
                position->x = streamValue.val.three_d.x;
                position->y = streamValue.val.three_d.y;
            }
            else {
                position->x = streamValue.val.two_d.x;
                position->y = streamValue.val.two_d.y;
            }
        }

        streamSuite->AEGP_DisposeStreamValue(&streamValue);
    }

    streamSuite->AEGP_DisposeStream(posStreamH);

    return err;
}

static PF_Err GetLayerScale(PF_InData* in_data, AEGP_LayerH layerH, const A_Time* time, AEGP_TwoDVal* scale) {
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamSuite5* streamSuite = suites.StreamSuite5();
    if (!streamSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_StreamRefH scaleStreamH = NULL;
    err = streamSuite->AEGP_GetNewLayerStream(
        NULL,
        layerH,
        AEGP_LayerStream_SCALE,
        &scaleStreamH);

    if (err || !scaleStreamH) {
        return err;
    }

    AEGP_StreamValue2 streamValue;
    err = streamSuite->AEGP_GetNewStreamValue(
        NULL,
        scaleStreamH,
        AEGP_LTimeMode_LayerTime,
        time,
        false,
        &streamValue);

    if (!err) {
        scale->x = streamValue.val.two_d.x;
        scale->y = streamValue.val.two_d.y;

        streamSuite->AEGP_DisposeStreamValue(&streamValue);
    }

    streamSuite->AEGP_DisposeStream(scaleStreamH);

    return err;
}

static PF_Err GetLayerRotation(PF_InData* in_data, AEGP_LayerH layerH, const A_Time* time, double* rotation) {
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamSuite5* streamSuite = suites.StreamSuite5();
    if (!streamSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_StreamRefH rotationStreamH = NULL;
    err = streamSuite->AEGP_GetNewLayerStream(
        NULL,
        layerH,
        AEGP_LayerStream_ROTATION,
        &rotationStreamH);

    if (err || !rotationStreamH) {
        return err;
    }

    AEGP_StreamValue2 streamValue;
    err = streamSuite->AEGP_GetNewStreamValue(
        NULL,
        rotationStreamH,
        AEGP_LTimeMode_LayerTime,
        time,
        false,
        &streamValue);

    if (!err) {
        *rotation = streamValue.val.one_d;
        streamSuite->AEGP_DisposeStreamValue(&streamValue);
    }

    streamSuite->AEGP_DisposeStream(rotationStreamH);

    return err;
}

static PF_Err GetLayerAnchorPoint(PF_InData* in_data, AEGP_LayerH layerH, const A_Time* time, AEGP_TwoDVal* anchor) {
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamSuite5* streamSuite = suites.StreamSuite5();
    if (!streamSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_StreamRefH anchorStreamH = NULL;
    err = streamSuite->AEGP_GetNewLayerStream(
        NULL,
        layerH,
        AEGP_LayerStream_ANCHORPOINT,
        &anchorStreamH);

    if (err || !anchorStreamH) {
        return err;
    }

    AEGP_StreamValue2 streamValue;
    err = streamSuite->AEGP_GetNewStreamValue(
        NULL,
        anchorStreamH,
        AEGP_LTimeMode_LayerTime,
        time,
        false,
        &streamValue);

    if (!err) {
        AEGP_StreamType type;
        err = streamSuite->AEGP_GetStreamType(anchorStreamH, &type);

        if (!err) {
            if (type == AEGP_StreamType_ThreeD || type == AEGP_StreamType_ThreeD_SPATIAL) {
                anchor->x = streamValue.val.three_d.x;
                anchor->y = streamValue.val.three_d.y;
            }
            else {
                anchor->x = streamValue.val.two_d.x;
                anchor->y = streamValue.val.two_d.y;
            }
        }

        streamSuite->AEGP_DisposeStreamValue(&streamValue);
    }

    streamSuite->AEGP_DisposeStream(anchorStreamH);

    return err;
}


static PF_FpLong TriangleWave(PF_FpLong t)
{
    t = fmod(t + 0.75, 1.0);

    if (t < 0)
        t += 1.0;

    return (fabs(t - 0.5) - 0.25) * 4.0;
}




static bool DetectMotionFromOtherEffects(PF_InData* in_data, double* motion_x, double* motion_y, double* rotation_angle, double* scale_x, double* scale_y, float* scale_velocity_out) {
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    *motion_x = 0;
    *motion_y = 0;
    *rotation_angle = 0;
    *scale_x = 0;
    *scale_y = 0;
    *scale_velocity_out = 0;

    AEGP_PFInterfaceSuite1* pfInterfaceSuite = suites.PFInterfaceSuite1();
    if (!pfInterfaceSuite) {
        return false;
    }

    AEGP_LayerH layerH = NULL;
    PF_Err err = pfInterfaceSuite->AEGP_GetEffectLayer(in_data->effect_ref, &layerH);
    if (err || !layerH) {
        return false;
    }

    AEGP_EffectRefH our_effectH = NULL;
    err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(NULL, in_data->effect_ref, &our_effectH);
    if (err || !our_effectH) {
        return false;
    }

    AEGP_InstalledEffectKey our_installed_key;
    err = suites.EffectSuite4()->AEGP_GetInstalledKeyFromLayerEffect(our_effectH, &our_installed_key);

    A_char our_match_name[AEGP_MAX_EFFECT_MATCH_NAME_SIZE + 1];
    if (!err) {
        err = suites.EffectSuite4()->AEGP_GetEffectMatchName(our_installed_key, our_match_name);
    }

    if (our_effectH) {
        suites.EffectSuite4()->AEGP_DisposeEffect(our_effectH);
        our_effectH = NULL;
    }

    if (err) {
        return false;      
    }

    A_long num_effects = 0;
    err = suites.EffectSuite4()->AEGP_GetLayerNumEffects(layerH, &num_effects);
    if (err || num_effects <= 0) {
        return false;
    }

    A_Time prev_time, current_time, next_time;

    current_time.value = in_data->current_time;
    current_time.scale = in_data->time_scale;

    prev_time.scale = current_time.scale;
    prev_time.value = current_time.value - in_data->time_step;

    next_time.scale = current_time.scale;
    next_time.value = current_time.value + in_data->time_step;

    bool found_motion = false;

    for (A_long i = 0; i < num_effects; i++) {
        AEGP_EffectRefH effectH = NULL;
        err = suites.EffectSuite4()->AEGP_GetLayerEffectByIndex(NULL, layerH, i, &effectH);
        if (err || !effectH) {
            continue;
        }

        AEGP_EffectFlags effect_flags;
        err = suites.EffectSuite4()->AEGP_GetEffectFlags(effectH, &effect_flags);
        if (err || !(effect_flags & AEGP_EffectFlags_ACTIVE)) {
            suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
            continue;
        }

        AEGP_InstalledEffectKey installed_key;
        err = suites.EffectSuite4()->AEGP_GetInstalledKeyFromLayerEffect(effectH, &installed_key);

        A_char match_name[AEGP_MAX_EFFECT_MATCH_NAME_SIZE + 1];
        if (!err) {
            err = suites.EffectSuite4()->AEGP_GetEffectMatchName(installed_key, match_name);
        }

        if (!err && strcmp(match_name, our_match_name) == 0) {
            suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
            continue;
        }


        if (strstr(match_name, "DKT Oscillate")) {
            A_long direction = 0;
            double angle = 45.0;
            double frequency = 2.00;
            double magnitude = 25;
            A_long wave_type = 0;
            double phase = 0.00;
            bool has_frequency_keyframes = false;
            bool has_phase_keyframes = false;

            bool normal_enabled = true;
            bool compatibility_enabled = false;
            double compat_angle = 0.0;
            double compat_frequency = 0.1;
            double compat_magnitude = 1.0;
            A_long compat_wave_type = 0;

            A_long num_params = 0;
            err = suites.StreamSuite5()->AEGP_GetEffectNumParamStreams(effectH, &num_params);
            if (err) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            const int NORMAL_CHECKBOX_PARAM = 1;
            const int DIRECTION_PARAM = 2;
            const int ANGLE_PARAM = 3;
            const int FREQUENCY_PARAM = 4;
            const int MAGNITUDE_PARAM = 5;
            const int WAVE_TYPE_PARAM = 6;
            const int PHASE_PARAM = 7;
            const int COMPATIBILITY_CHECKBOX_PARAM = 14;
            const int COMPATIBILITY_ANGLE_PARAM = 15;
            const int COMPATIBILITY_FREQUENCY_PARAM = 16;
            const int COMPATIBILITY_MAGNITUDE_PARAM = 17;
            const int COMPATIBILITY_WAVE_TYPE_PARAM = 18;

            if (num_params < 19) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            AEGP_StreamRefH streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, NORMAL_CHECKBOX_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    normal_enabled = (value.val.one_d != 0);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_CHECKBOX_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    compatibility_enabled = (value.val.one_d != 0);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            if ((normal_enabled && compatibility_enabled) || (!normal_enabled && !compatibility_enabled)) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, DIRECTION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    direction = (A_long)value.val.one_d - 1;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, ANGLE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    angle = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, FREQUENCY_PARAM, &streamH);
            if (!err && streamH) {
                A_long num_keyframes = 0;
                err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
                if (!err && num_keyframes > 0) {
                    has_frequency_keyframes = true;
                }

                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    frequency = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MAGNITUDE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    magnitude = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, WAVE_TYPE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    wave_type = (A_long)value.val.one_d - 1;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, PHASE_PARAM, &streamH);
            if (!err && streamH) {
                A_long num_keyframes = 0;
                err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
                if (!err && num_keyframes > 0) {
                    has_phase_keyframes = true;
                }

                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    phase = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            if (compatibility_enabled) {
                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_ANGLE_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_angle = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }
                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }

                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_FREQUENCY_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_frequency = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }
                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }

                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_MAGNITUDE_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_magnitude = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }
                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }

                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_WAVE_TYPE_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_wave_type = (A_long)value.val.one_d - 1;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }
                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }
            }

            AEGP_LayerIDVal layer_id = 0;
            err = suites.LayerSuite7()->AEGP_GetLayerID(layerH, &layer_id);

            double layer_time_offset = 0;
            A_Ratio stretch_factor = { 1, 1 };

            if (!err && layer_id != 0) {
                A_Time in_point;
                err = suites.LayerSuite7()->AEGP_GetLayerInPoint(layerH, AEGP_LTimeMode_LayerTime, &in_point);

                if (!err) {
                    layer_time_offset = (double)in_point.value / (double)in_point.scale;
                }

                err = suites.LayerSuite7()->AEGP_GetLayerStretch(layerH, &stretch_factor);
            }

            double prev_time_secs = (double)prev_time.value / (double)prev_time.scale;
            double current_time_secs = (double)current_time.value / (double)current_time.scale;
            double next_time_secs = (double)next_time.value / (double)next_time.scale;

            prev_time_secs -= layer_time_offset;
            current_time_secs -= layer_time_offset;
            next_time_secs -= layer_time_offset;

            double stretch_ratio = (double)stretch_factor.num / (double)stretch_factor.den;
            prev_time_secs *= stretch_ratio;
            current_time_secs *= stretch_ratio;
            next_time_secs *= stretch_ratio;

            auto TriangleWave = [](double t) -> double {
                t = fmod(t + 0.75, 1.0);
                if (t < 0) t += 1.0;
                return (fabs(t - 0.5) - 0.25) * 4.0;
                };

            auto valueAtTime = [&suites, effectH, FREQUENCY_PARAM, PHASE_PARAM](int stream_index, double time_secs, double time_scale) -> double {
                PF_Err local_err = PF_Err_NONE;
                double value_out = 0.0;

                AEGP_StreamRefH stream_ref = NULL;
                A_Time time;
                time.value = (A_long)(time_secs * time_scale);
                time.scale = (A_long)time_scale;

                local_err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(
                    NULL,
                    effectH,
                    stream_index,
                    &stream_ref);

                if (!local_err && stream_ref) {
                    AEGP_StreamValue2 stream_value;
                    local_err = suites.StreamSuite5()->AEGP_GetNewStreamValue(
                        NULL,
                        stream_ref,
                        AEGP_LTimeMode_LayerTime,
                        &time,
                        FALSE,
                        &stream_value);

                    if (!local_err) {
                        AEGP_StreamType stream_type;
                        local_err = suites.StreamSuite5()->AEGP_GetStreamType(stream_ref, &stream_type);

                        if (!local_err) {
                            switch (stream_type) {
                            case AEGP_StreamType_OneD:
                                value_out = stream_value.val.one_d;
                                break;

                            case AEGP_StreamType_TwoD:
                            case AEGP_StreamType_TwoD_SPATIAL:
                                value_out = stream_value.val.two_d.x;
                                break;

                            case AEGP_StreamType_ThreeD:
                            case AEGP_StreamType_ThreeD_SPATIAL:
                                value_out = stream_value.val.three_d.x;
                                break;

                            case AEGP_StreamType_COLOR:
                                value_out = stream_value.val.color.redF;
                                break;
                            }
                        }

                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
                    }

                    suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
                }

                return value_out;
                };

            auto valueAtTimeHz = [&valueAtTime, &has_frequency_keyframes, FREQUENCY_PARAM](
                int stream_index, double time_secs, double duration, double time_scale, double layer_start_time) -> double {

                    double value_out = valueAtTime(stream_index, time_secs, time_scale);

                    if (stream_index == FREQUENCY_PARAM && has_frequency_keyframes) {
                        bool isHz = true;

                        if (isHz) {
                            double accumulated_phase = 0.0;
                            double fps = 120.0;
                            int totalSteps = (int)round(duration * fps);
                            int curSteps = (int)round(fps * time_secs);

                            if (curSteps >= 0) {
                                for (int i = 0; i <= curSteps; i++) {
                                    double stepValue;
                                    double adjusted_time = i / fps + layer_start_time;
                                    stepValue = valueAtTime(stream_index, adjusted_time, time_scale);
                                    accumulated_phase += stepValue / fps;
                                }
                                value_out = accumulated_phase;
                            }
                        }
                    }

                    return value_out;
                };

            double prev_offsetX = 0, prev_offsetY = 0, prev_scale = 100.0;
            double current_offsetX = 0, current_offsetY = 0, current_scale = 100.0;
            double next_offsetX = 0, next_offsetY = 0, next_scale = 100.0;

            double prev_accumulated_phase = 0.0;
            double current_accumulated_phase = 0.0;
            double next_accumulated_phase = 0.0;
            double time_scale = current_time.scale;

            if (has_frequency_keyframes && normal_enabled) {
                prev_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, prev_time_secs, prev_time_secs, time_scale, layer_time_offset);
                current_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, current_time_secs, current_time_secs, time_scale, layer_time_offset);
                next_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, next_time_secs, next_time_secs, time_scale, layer_time_offset);
            }

            if (compatibility_enabled) {
                double compatAngleRad = compat_angle * M_PI / 180.0;
                double compat_dx = sin(compatAngleRad);
                double compat_dy = cos(compatAngleRad);

                auto calculateCompatWaveValue = [&TriangleWave](int wave_type, double frequency, double time) -> double {
                    double m;
                    if (wave_type == 0) {
                        m = sin(time * frequency * 3.14159);
                    }
                    else {
                        double wavePhase = time * frequency / 2.0;
                        m = TriangleWave(wavePhase);
                    }
                    return m;
                    };

                double prev_compat_wave = calculateCompatWaveValue(compat_wave_type, compat_frequency, prev_time_secs);
                double current_compat_wave = calculateCompatWaveValue(compat_wave_type, compat_frequency, current_time_secs);
                double next_compat_wave = calculateCompatWaveValue(compat_wave_type, compat_frequency, next_time_secs);

                prev_offsetX = compat_dx * compat_magnitude * prev_compat_wave;
                prev_offsetY = compat_dy * compat_magnitude * prev_compat_wave;

                current_offsetX = compat_dx * compat_magnitude * current_compat_wave;
                current_offsetY = compat_dy * compat_magnitude * current_compat_wave;

                next_offsetX = compat_dx * compat_magnitude * next_compat_wave;
                next_offsetY = compat_dy * compat_magnitude * next_compat_wave;
            }
            else if (normal_enabled) {
                double angleRad = angle * M_PI / 180.0;
                double dx = cos(angleRad);
                double dy = sin(angleRad);

                auto calculateWaveValue = [&TriangleWave](int wave_type, double frequency, double time, double phase, double accumulated_phase = 0.0) -> double {
                    double X, m;

                    if (accumulated_phase > 0.0) {
                        if (wave_type == 0) {
                            X = ((accumulated_phase * 2.0 + phase * 2.0) * 3.14159);
                            m = sin(X);
                        }
                        else {
                            X = ((accumulated_phase * 2.0) + (phase * 2.0)) / 2.0 + phase;
                            m = TriangleWave(X);
                        }
                    }
                    else {
                        if (wave_type == 0) {
                            X = (frequency * 2.0 * time) + (phase * 2.0);
                            m = sin(X * M_PI);
                        }
                        else {
                            X = ((frequency * 2.0 * time) + (phase * 2.0)) / 2.0 + phase;
                            m = TriangleWave(X);
                        }
                    }

                    return m;
                    };

                double prev_phase = phase;
                double current_phase = phase;
                double next_phase = phase;

                if (has_phase_keyframes) {
                    prev_phase = valueAtTime(PHASE_PARAM, prev_time_secs + layer_time_offset, time_scale);
                    current_phase = valueAtTime(PHASE_PARAM, current_time_secs + layer_time_offset, time_scale);
                    next_phase = valueAtTime(PHASE_PARAM, next_time_secs + layer_time_offset, time_scale);
                }

                double prev_wave;
                double current_wave;
                double next_wave;

                if (has_frequency_keyframes) {
                    prev_wave = calculateWaveValue(wave_type, 0, 0, prev_phase, prev_accumulated_phase);
                    current_wave = calculateWaveValue(wave_type, 0, 0, current_phase, current_accumulated_phase);
                    next_wave = calculateWaveValue(wave_type, 0, 0, next_phase, next_accumulated_phase);
                }
                else {
                    prev_wave = calculateWaveValue(wave_type, frequency, prev_time_secs, prev_phase);
                    current_wave = calculateWaveValue(wave_type, frequency, current_time_secs, current_phase);
                    next_wave = calculateWaveValue(wave_type, frequency, next_time_secs, next_phase);
                }

                switch (direction) {
                case 0:
                    prev_offsetX = dx * magnitude * prev_wave;
                    prev_offsetY = dy * magnitude * prev_wave;
                    break;

                case 1:
                    prev_scale = 100.0 + (magnitude * prev_wave * 0.1);
                    break;

                case 2: {
                    prev_offsetX = dx * magnitude * prev_wave;
                    prev_offsetY = dy * magnitude * prev_wave;

                    double phaseShift = wave_type == 0 ? 0.25 : 0.125;
                    double m_scale;

                    if (has_frequency_keyframes) {
                        m_scale = calculateWaveValue(wave_type, 0, 0, prev_phase + phaseShift, prev_accumulated_phase);
                    }
                    else {
                        m_scale = calculateWaveValue(wave_type, frequency, prev_time_secs, prev_phase + phaseShift);
                    }
                    prev_scale = 100.0 + (magnitude * m_scale * 0.1);
                    break;
                }
                }

                switch (direction) {
                case 0:
                    current_offsetX = dx * magnitude * current_wave;
                    current_offsetY = dy * magnitude * current_wave;
                    break;

                case 1:
                    current_scale = 100.0 + (magnitude * current_wave * 0.1);
                    break;

                case 2: {
                    current_offsetX = dx * magnitude * current_wave;
                    current_offsetY = dy * magnitude * current_wave;

                    double phaseShift = wave_type == 0 ? 0.25 : 0.125;
                    double m_scale;

                    if (has_frequency_keyframes) {
                        m_scale = calculateWaveValue(wave_type, 0, 0, phase + phaseShift, current_accumulated_phase);
                    }
                    else {
                        m_scale = calculateWaveValue(wave_type, frequency, current_time_secs, phase + phaseShift);
                    }
                    current_scale = 100.0 + (magnitude * m_scale * 0.1);
                    break;
                }
                }

                switch (direction) {
                case 0:
                    next_offsetX = dx * magnitude * next_wave;
                    next_offsetY = dy * magnitude * next_wave;
                    break;

                case 1:
                    next_scale = 100.0 + (magnitude * next_wave * 0.1);
                    break;

                case 2: {
                    next_offsetX = dx * magnitude * next_wave;
                    next_offsetY = dy * magnitude * next_wave;

                    double phaseShift = wave_type == 0 ? 0.25 : 0.125;
                    double m_scale;

                    if (has_frequency_keyframes) {
                        m_scale = calculateWaveValue(wave_type, 0, 0, next_phase + phaseShift, next_accumulated_phase);
                    }
                    else {
                        m_scale = calculateWaveValue(wave_type, frequency, next_time_secs, next_phase + phaseShift);
                    }
                    next_scale = 100.0 + (magnitude * m_scale * 0.1);
                    break;
                }
                }
            }

            double prev_to_current_x = current_offsetX - prev_offsetX;
            double prev_to_current_y = current_offsetY - prev_offsetY;
            double current_to_next_x = next_offsetX - current_offsetX;
            double current_to_next_y = next_offsetY - current_offsetY;

            double prev_to_current_scale = current_scale - prev_scale;
            double current_to_next_scale = next_scale - current_scale;

            double prev_to_current_magnitude = sqrt(prev_to_current_x * prev_to_current_x + prev_to_current_y * prev_to_current_y);
            double current_to_next_magnitude = sqrt(current_to_next_x * current_to_next_x + current_to_next_y * current_to_next_y);

            if (compatibility_enabled || (normal_enabled && direction == 0)) {
                if (current_to_next_magnitude > prev_to_current_magnitude) {
                    *motion_x += current_to_next_x;
                    *motion_y += current_to_next_y;
                }
                else {
                    *motion_x += prev_to_current_x;
                    *motion_y += prev_to_current_y;
                }

                if (fabs(*motion_x) > 0.01 || fabs(*motion_y) > 0.01) {
                    found_motion = true;
                }
            }
            else if (normal_enabled && (direction == 1 || direction == 2)) {
                if (direction == 2) {
                    if (current_to_next_magnitude > prev_to_current_magnitude) {
                        *motion_x += current_to_next_x;
                        *motion_y += current_to_next_y;
                    }
                    else {
                        *motion_x += prev_to_current_x;
                        *motion_y += prev_to_current_y;
                    }

                    if (fabs(*motion_x) > 0.01 || fabs(*motion_y) > 0.01) {
                        found_motion = true;
                    }
                }

                AEGP_CompH compH = NULL;
                err = suites.LayerSuite7()->AEGP_GetLayerParentComp(layerH, &compH);
                if (!err && compH) {
                    AEGP_ItemH itemH = NULL;
                    err = suites.CompSuite11()->AEGP_GetItemFromComp(compH, &itemH);

                    A_long width = 0, height = 0;
                    if (!err && itemH) {
                        err = suites.ItemSuite9()->AEGP_GetItemDimensions(itemH, &width, &height);

                        if (!err) {
                            double scaleChange = fabs(current_to_next_scale) > fabs(prev_to_current_scale) ?
                                current_to_next_scale : prev_to_current_scale;

                            float layer_width = (float)width;
                            float layer_height = (float)height;

                            float current_size = sqrt(pow(layer_width * (current_scale / 100.0f), 2) +
                                pow(layer_height * (current_scale / 100.0f), 2));

                            float prev_size = sqrt(pow(layer_width * (prev_scale / 100.0f), 2) +
                                pow(layer_height * (prev_scale / 100.0f), 2));

                            float next_size = sqrt(pow(layer_width * (next_scale / 100.0f), 2) +
                                pow(layer_height * (next_scale / 100.0f), 2));

                            if (fabs(next_size - current_size) > fabs(current_size - prev_size)) {
                                *scale_velocity_out = next_size - current_size;
                            }
                            else {
                                *scale_velocity_out = current_size - prev_size;
                            }

                            *scale_velocity_out *= 1.6f;

                            if (fabs(*scale_velocity_out) > 0.01f) {
                                found_motion = true;
                            }
                        }
                    }
                }
            }
        }
       

        if (strstr(match_name, "DKT Auto-Shake")) {
            double magnitude = 50;
            double frequency = 2.00;
            double evolution = 0.00;
            double seed = 0.00;
            double angle = 45.0;
            double slack = 0.25;
            double zshake = 0.0;
            bool x_tiles = false;
            bool y_tiles = false;
            bool mirror = false;
            bool has_frequency_keyframes = false;

            bool normal_enabled = true;
            bool compatibility_enabled = false;
            double compat_magnitude = 50.0;
            double compat_speed = 1.0;
            double compat_evolution = 0.0;
            double compat_seed = 0.0;
            double compat_angle = 45.0;
            double compat_slack = 0.25;

            double prev_magnitude = magnitude;
            double prev_frequency = frequency;
            double prev_evolution = evolution;
            double prev_seed = seed;
            double prev_angle = angle;
            double prev_slack = slack;
            double prev_zshake = zshake;

            double next_magnitude = magnitude;
            double next_frequency = frequency;
            double next_evolution = evolution;
            double next_seed = seed;
            double next_angle = angle;
            double next_slack = slack;
            double next_zshake = zshake;

            double prev_compat_magnitude = compat_magnitude;
            double prev_compat_speed = compat_speed;
            double prev_compat_evolution = compat_evolution;
            double prev_compat_seed = compat_seed;
            double prev_compat_angle = compat_angle;
            double prev_compat_slack = compat_slack;

            double next_compat_magnitude = compat_magnitude;
            double next_compat_speed = compat_speed;
            double next_compat_evolution = compat_evolution;
            double next_compat_seed = compat_seed;
            double next_compat_angle = compat_angle;
            double next_compat_slack = compat_slack;

            A_long num_params = 0;
            err = suites.StreamSuite5()->AEGP_GetEffectNumParamStreams(effectH, &num_params);
            if (err) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            const int NORMAL_CHECKBOX_PARAM = 1;
            const int MAGNITUDE_PARAM = 2;
            const int FREQUENCY_PARAM = 3;
            const int EVOLUTION_PARAM = 4;
            const int SEED_PARAM = 5;
            const int ANGLE_PARAM = 6;
            const int SLACK_PARAM = 7;
            const int ZSHAKE_PARAM = 8;
            const int X_TILES_PARAM = 10;
            const int Y_TILES_PARAM = 11;
            const int MIRROR_PARAM = 12;
            const int COMPATIBILITY_CHECKBOX_PARAM = 15;
            const int COMPATIBILITY_MAGNITUDE_PARAM = 16;
            const int COMPATIBILITY_SPEED_PARAM = 17;
            const int COMPATIBILITY_EVOLUTION_PARAM = 18;
            const int COMPATIBILITY_SEED_PARAM = 19;
            const int COMPATIBILITY_ANGLE_PARAM = 20;
            const int COMPATIBILITY_SLACK_PARAM = 21;
            if (num_params < 23) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            AEGP_StreamRefH streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, NORMAL_CHECKBOX_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    normal_enabled = (value.val.one_d != 0);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_CHECKBOX_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    compatibility_enabled = (value.val.one_d != 0);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            if ((normal_enabled && compatibility_enabled) || (!normal_enabled && !compatibility_enabled)) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MAGNITUDE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    magnitude = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_magnitude = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_magnitude = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, FREQUENCY_PARAM, &streamH);
            if (!err && streamH) {
                A_long num_keyframes = 0;
                err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
                if (!err && num_keyframes > 0) {
                    has_frequency_keyframes = true;
                }

                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    frequency = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_frequency = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_frequency = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, EVOLUTION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    evolution = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_evolution = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_evolution = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SEED_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    seed = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_seed = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_seed = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, ANGLE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    angle = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_angle = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_angle = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SLACK_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    slack = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_slack = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_slack = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, ZSHAKE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    zshake = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_zshake = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_zshake = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, X_TILES_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    x_tiles = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, Y_TILES_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    y_tiles = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MIRROR_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    mirror = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            if (compatibility_enabled) {
                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_MAGNITUDE_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_magnitude = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &prev_time, FALSE, &value);
                    if (!err) {
                        prev_compat_magnitude = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &next_time, FALSE, &value);
                    if (!err) {
                        next_compat_magnitude = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }

                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_SPEED_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_speed = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &prev_time, FALSE, &value);
                    if (!err) {
                        prev_compat_speed = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &next_time, FALSE, &value);
                    if (!err) {
                        next_compat_speed = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }

                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_EVOLUTION_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_evolution = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &prev_time, FALSE, &value);
                    if (!err) {
                        prev_compat_evolution = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &next_time, FALSE, &value);
                    if (!err) {
                        next_compat_evolution = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }

                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_SEED_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_seed = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &prev_time, FALSE, &value);
                    if (!err) {
                        prev_compat_seed = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &next_time, FALSE, &value);
                    if (!err) {
                        next_compat_seed = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }

                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_ANGLE_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_angle = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &prev_time, FALSE, &value);
                    if (!err) {
                        prev_compat_angle = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &next_time, FALSE, &value);
                    if (!err) {
                        next_compat_angle = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }

                streamH = NULL;
                err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_SLACK_PARAM, &streamH);
                if (!err && streamH) {
                    AEGP_StreamValue2 value;
                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &current_time, FALSE, &value);
                    if (!err) {
                        compat_slack = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &prev_time, FALSE, &value);
                    if (!err) {
                        prev_compat_slack = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                        &next_time, FALSE, &value);
                    if (!err) {
                        next_compat_slack = value.val.one_d;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                    }

                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }
            }

            AEGP_LayerIDVal layer_id = 0;
            err = suites.LayerSuite7()->AEGP_GetLayerID(layerH, &layer_id);

            double layer_time_offset = 0;
            A_Ratio stretch_factor = { 1, 1 };

            if (!err && layer_id != 0) {
                A_Time in_point;
                err = suites.LayerSuite7()->AEGP_GetLayerInPoint(layerH, AEGP_LTimeMode_LayerTime, &in_point);

                if (!err) {
                    layer_time_offset = (double)in_point.value / (double)in_point.scale;
                }

                err = suites.LayerSuite7()->AEGP_GetLayerStretch(layerH, &stretch_factor);
            }

            double prev_time_secs = (double)prev_time.value / (double)prev_time.scale;
            double current_time_secs = (double)current_time.value / (double)current_time.scale;
            double next_time_secs = (double)next_time.value / (double)next_time.scale;

            prev_time_secs -= layer_time_offset;
            current_time_secs -= layer_time_offset;
            next_time_secs -= layer_time_offset;

            double stretch_ratio = (double)stretch_factor.num / (double)stretch_factor.den;
            prev_time_secs *= stretch_ratio;
            current_time_secs *= stretch_ratio;
            next_time_secs *= stretch_ratio;

            auto valueAtTime = [&suites, effectH](int stream_index, double time_secs, double time_scale) -> double {
                PF_Err local_err = PF_Err_NONE;
                double value_out = 0.0;

                AEGP_StreamRefH stream_ref = NULL;
                A_Time time;
                time.value = (A_long)(time_secs * time_scale);
                time.scale = (A_long)time_scale;

                local_err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(
                    NULL,
                    effectH,
                    stream_index,
                    &stream_ref);

                if (!local_err && stream_ref) {
                    AEGP_StreamValue2 stream_value;
                    local_err = suites.StreamSuite5()->AEGP_GetNewStreamValue(
                        NULL,
                        stream_ref,
                        AEGP_LTimeMode_LayerTime,
                        &time,
                        FALSE,
                        &stream_value);

                    if (!local_err) {
                        AEGP_StreamType stream_type;
                        local_err = suites.StreamSuite5()->AEGP_GetStreamType(stream_ref, &stream_type);

                        if (!local_err) {
                            switch (stream_type) {
                            case AEGP_StreamType_OneD:
                                value_out = stream_value.val.one_d;
                                break;

                            case AEGP_StreamType_TwoD:
                            case AEGP_StreamType_TwoD_SPATIAL:
                                value_out = stream_value.val.two_d.x;
                                break;

                            case AEGP_StreamType_ThreeD:
                            case AEGP_StreamType_ThreeD_SPATIAL:
                                value_out = stream_value.val.three_d.x;
                                break;

                            case AEGP_StreamType_COLOR:
                                value_out = stream_value.val.color.redF;
                                break;
                            }
                        }

                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
                    }

                    suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
                }

                return value_out;
                };

            auto valueAtTimeHz = [&valueAtTime, &has_frequency_keyframes, FREQUENCY_PARAM](
                int stream_index, double time_secs, double duration, double time_scale, double layer_start_time) -> double {

                    double value_out = valueAtTime(stream_index, time_secs, time_scale);

                    if (stream_index == FREQUENCY_PARAM && has_frequency_keyframes) {
                        bool isHz = true;

                        if (isHz) {
                            double accumulated_phase = 0.0;
                            double fps = 120.0;
                            int totalSteps = (int)round(duration * fps);
                            int curSteps = (int)round(fps * time_secs);

                            if (curSteps >= 0) {
                                for (int i = 0; i <= curSteps; i++) {
                                    double stepValue;
                                    double adjusted_time = i / fps + layer_start_time;
                                    stepValue = valueAtTime(stream_index, adjusted_time, time_scale);
                                    accumulated_phase += stepValue / fps;
                                }
                                value_out = accumulated_phase;
                            }
                        }
                    }

                    return value_out;
                };

            double time_scale = current_time.scale;
            double duration = current_time_secs;

            double prev_accumulated_phase = 0.0;
            double current_accumulated_phase = 0.0;
            double next_accumulated_phase = 0.0;

            if (has_frequency_keyframes && normal_enabled) {
                prev_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, prev_time_secs, prev_time_secs, time_scale, layer_time_offset);
                current_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, current_time_secs, current_time_secs, time_scale, layer_time_offset);
                next_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, next_time_secs, next_time_secs, time_scale, layer_time_offset);
            }

            double prev_dx, prev_dy, prev_dz;
            double curr_dx, curr_dy, curr_dz;
            double next_dx, next_dy, next_dz;

            if (normal_enabled) {
                double prev_evolution_val, current_evolution_val, next_evolution_val;

                if (has_frequency_keyframes) {
                    prev_evolution_val = prev_evolution + prev_accumulated_phase;
                    current_evolution_val = evolution + current_accumulated_phase;
                    next_evolution_val = next_evolution + next_accumulated_phase;
                }
                else {
                    prev_evolution_val = prev_evolution + prev_frequency * prev_time_secs;
                    current_evolution_val = evolution + frequency * current_time_secs;
                    next_evolution_val = next_evolution + next_frequency * next_time_secs;
                }

                prev_dx = SimplexNoise::simplex_noise(prev_evolution_val, prev_seed * 49235.319798, 2);
                prev_dy = SimplexNoise::simplex_noise(prev_evolution_val + 7468.329, prev_seed * 19337.940385, 2);
                prev_dz = SimplexNoise::simplex_noise(prev_evolution_val + 14192.277, prev_seed * 71401.168533, 2);

                curr_dx = SimplexNoise::simplex_noise(current_evolution_val, seed * 49235.319798, 2);
                curr_dy = SimplexNoise::simplex_noise(current_evolution_val + 7468.329, seed * 19337.940385, 2);
                curr_dz = SimplexNoise::simplex_noise(current_evolution_val + 14192.277, seed * 71401.168533, 2);

                next_dx = SimplexNoise::simplex_noise(next_evolution_val, next_seed * 49235.319798, 2);
                next_dy = SimplexNoise::simplex_noise(next_evolution_val + 7468.329, next_seed * 19337.940385, 2);
                next_dz = SimplexNoise::simplex_noise(next_evolution_val + 14192.277, next_seed * 71401.168533, 2);

                prev_dx *= prev_magnitude;
                prev_dy *= prev_magnitude * prev_slack;
                prev_dz *= prev_zshake;

                curr_dx *= magnitude;
                curr_dy *= magnitude * slack;
                curr_dz *= zshake;

                next_dx *= next_magnitude;
                next_dy *= next_magnitude * next_slack;
                next_dz *= next_zshake;
            }
            else {
                double prev_evolution_compat = prev_compat_evolution + (prev_time_secs * prev_compat_speed) - prev_compat_speed;
                double curr_evolution_compat = compat_evolution + (current_time_secs * compat_speed) - compat_speed;
                double next_evolution_compat = next_compat_evolution + (next_time_secs * next_compat_speed) - next_compat_speed;

                prev_dx = SimplexNoise::simplex_noise(prev_compat_seed * 54623.245, 0, prev_evolution_compat + prev_compat_seed * 49235.319798, 3);
                prev_dy = SimplexNoise::simplex_noise(0, prev_compat_seed * 8723.5647, prev_evolution_compat + 7468.329 + prev_compat_seed * 19337.940385, 3);
                prev_dz = 0;

                curr_dx = SimplexNoise::simplex_noise(compat_seed * 54623.245, 0, curr_evolution_compat + compat_seed * 49235.319798, 3);
                curr_dy = SimplexNoise::simplex_noise(0, compat_seed * 8723.5647, curr_evolution_compat + 7468.329 + compat_seed * 19337.940385, 3);
                curr_dz = 0;

                next_dx = SimplexNoise::simplex_noise(next_compat_seed * 54623.245, 0, next_evolution_compat + next_compat_seed * 49235.319798, 3);
                next_dy = SimplexNoise::simplex_noise(0, next_compat_seed * 8723.5647, next_evolution_compat + 7468.329 + next_compat_seed * 19337.940385, 3);
                next_dz = 0;

                prev_dx *= prev_compat_magnitude;
                prev_dy *= prev_compat_magnitude * prev_compat_slack;

                curr_dx *= compat_magnitude;
                curr_dy *= compat_magnitude * compat_slack;

                next_dx *= next_compat_magnitude;
                next_dy *= next_compat_magnitude * next_compat_slack;
            }

            double prev_angleRad = (normal_enabled ? prev_angle : prev_compat_angle) * M_PI / 180.0;
            double prev_c = cos(prev_angleRad);
            double prev_s = sin(prev_angleRad);
            double prev_rx = prev_dx * prev_c - prev_dy * prev_s;
            double prev_ry = prev_dx * prev_s + prev_dy * prev_c;

            double curr_angleRad = (normal_enabled ? angle : compat_angle) * M_PI / 180.0;
            double curr_c = cos(curr_angleRad);
            double curr_s = sin(curr_angleRad);
            double curr_rx = curr_dx * curr_c - curr_dy * curr_s;
            double curr_ry = curr_dx * curr_s + curr_dy * curr_c;

            double next_angleRad = (normal_enabled ? next_angle : next_compat_angle) * M_PI / 180.0;
            double next_c = cos(next_angleRad);
            double next_s = sin(next_angleRad);
            double next_rx = next_dx * next_c - next_dy * next_s;
            double next_ry = next_dx * next_s + next_dy * next_c;

            double prev_to_current_x = curr_rx - prev_rx;
            double prev_to_current_y = curr_ry - prev_ry;
            double current_to_next_x = next_rx - curr_rx;
            double current_to_next_y = next_ry - curr_ry;

            double prev_to_current_magnitude = sqrt(prev_to_current_x * prev_to_current_x +
                prev_to_current_y * prev_to_current_y);
            double current_to_next_magnitude = sqrt(current_to_next_x * current_to_next_x +
                current_to_next_y * current_to_next_y);

            if (current_to_next_magnitude > prev_to_current_magnitude) {
                *motion_x += current_to_next_x;
                *motion_y += current_to_next_y;
            }
            else {
                *motion_x += prev_to_current_x;
                *motion_y += prev_to_current_y;
            }

            if (normal_enabled && (prev_zshake != 0 || zshake != 0 || next_zshake != 0)) {
                prev_dz = -prev_dz;
                curr_dz = -curr_dz;
                next_dz = -next_dz;

                double prev_scale_factor = 1000.0 / (1000.0 - prev_dz);
                double curr_scale_factor = 1000.0 / (1000.0 - curr_dz);
                double next_scale_factor = 1000.0 / (1000.0 - next_dz);

                prev_scale_factor = fmin(fmax(prev_scale_factor, 0.1), 10.0);
                curr_scale_factor = fmin(fmax(curr_scale_factor, 0.1), 10.0);
                next_scale_factor = fmin(fmax(next_scale_factor, 0.1), 10.0);

                double prev_to_curr_scale_velocity = curr_scale_factor - prev_scale_factor;
                double curr_to_next_scale_velocity = next_scale_factor - curr_scale_factor;

                if (fabs(curr_to_next_scale_velocity) > fabs(prev_to_curr_scale_velocity)) {
                    *scale_velocity_out = curr_to_next_scale_velocity * 100.0f;
                }
                else {
                    *scale_velocity_out = prev_to_curr_scale_velocity * 100.0f;
                }
            }

            if (fabs(*motion_x) > 0.01 || fabs(*motion_y) > 0.01 || fabs(*scale_velocity_out) > 0.01f) {
                found_motion = true;
            }
        }

else if (strstr(match_name, "DKT Swing")) {
    double frequency = 2.0;
    double angle1 = -30.0;
    double angle2 = 30.0;
    double phase = 0.0;
    A_long wave_type = 0;
    bool x_tiles = false;
    bool y_tiles = false;
    bool mirror = false;
    bool has_frequency_keyframes = false;
    bool has_phase_keyframes = false;
    bool has_angle1_keyframes = false;
    bool has_angle2_keyframes = false;
    bool normal_enabled = true;
    bool compatibility_enabled = false;
    double compat_frequency = 2.0;
    double compat_angle1 = -30.0;
    double compat_angle2 = 30.0;
    double compat_phase = 0.0;
    A_long compat_wave_type = 0;

    A_long num_params = 0;
    err = suites.StreamSuite5()->AEGP_GetEffectNumParamStreams(effectH, &num_params);
    if (err) {
        suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
        continue;
    }

    const int NORMAL_CHECKBOX_PARAM = 1;
    const int FREQUENCY_PARAM = 2;
    const int ANGLE1_PARAM = 3;
    const int ANGLE2_PARAM = 4;
    const int PHASE_PARAM = 5;
    const int WAVE_TYPE_PARAM = 6;
    const int X_TILES_PARAM = 8;
    const int Y_TILES_PARAM = 9;
    const int MIRROR_PARAM = 10;
    const int COMPATIBILITY_CHECKBOX_PARAM = 13;
    const int COMPATIBILITY_FREQUENCY_PARAM = 14;
    const int COMPATIBILITY_ANGLE1_PARAM = 15;
    const int COMPATIBILITY_ANGLE2_PARAM = 16;
    const int COMPATIBILITY_PHASE_PARAM = 17;
    const int COMPATIBILITY_WAVE_TYPE_PARAM = 18;

    if (num_params < 19) {
        suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
        continue;
    }

    AEGP_StreamRefH streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, NORMAL_CHECKBOX_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            normal_enabled = (value.val.one_d != 0);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_CHECKBOX_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            compatibility_enabled = (value.val.one_d != 0);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    if ((normal_enabled && compatibility_enabled) || (!normal_enabled && !compatibility_enabled)) {
        suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
        continue;
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, FREQUENCY_PARAM, &streamH);
    if (!err && streamH) {
        A_long num_keyframes = 0;
        err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
        if (!err && num_keyframes > 0) {
            has_frequency_keyframes = true;
        }

        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            frequency = value.val.one_d;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, ANGLE1_PARAM, &streamH);
    if (!err && streamH) {
        A_long num_keyframes = 0;
        err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
        if (!err && num_keyframes > 0) {
            has_angle1_keyframes = true;
        }

        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            angle1 = value.val.one_d;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, ANGLE2_PARAM, &streamH);
    if (!err && streamH) {
        A_long num_keyframes = 0;
        err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
        if (!err && num_keyframes > 0) {
            has_angle2_keyframes = true;
        }

        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            angle2 = value.val.one_d;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, PHASE_PARAM, &streamH);
    if (!err && streamH) {
        A_long num_keyframes = 0;
        err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
        if (!err && num_keyframes > 0) {
            has_phase_keyframes = true;
        }

        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            phase = value.val.one_d;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, WAVE_TYPE_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            wave_type = (A_long)value.val.one_d - 1;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, X_TILES_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            x_tiles = (value.val.one_d > 0.5);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, Y_TILES_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            y_tiles = (value.val.one_d > 0.5);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MIRROR_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            mirror = (value.val.one_d > 0.5);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    if (compatibility_enabled) {
        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_FREQUENCY_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_frequency = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_ANGLE1_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_angle1 = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_ANGLE2_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_angle2 = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_PHASE_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_phase = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_WAVE_TYPE_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_wave_type = (A_long)value.val.one_d - 1;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }
    }

    double prev_time_secs = (double)prev_time.value / (double)prev_time.scale;
    double current_time_secs = (double)current_time.value / (double)current_time.scale;
    double next_time_secs = (double)next_time.value / (double)next_time.scale;

    AEGP_LayerIDVal layer_id = 0;
    err = suites.LayerSuite7()->AEGP_GetLayerID(layerH, &layer_id);

    double layer_time_offset = 0;
    A_Ratio stretch_factor = { 1, 1 };

    if (!err && layer_id != 0) {
        A_Time in_point;
        err = suites.LayerSuite7()->AEGP_GetLayerInPoint(layerH, AEGP_LTimeMode_LayerTime, &in_point);

        if (!err) {
            layer_time_offset = (double)in_point.value / (double)in_point.scale;
        }

        err = suites.LayerSuite7()->AEGP_GetLayerStretch(layerH, &stretch_factor);
    }

    double layer_relative_prev_time = prev_time_secs - layer_time_offset;
    double layer_relative_current_time = current_time_secs - layer_time_offset;
    double layer_relative_next_time = next_time_secs - layer_time_offset;

    double stretch_ratio = (double)stretch_factor.num / (double)stretch_factor.den;
    prev_time_secs *= stretch_ratio;
    current_time_secs *= stretch_ratio;
    next_time_secs *= stretch_ratio;

    auto valueAtTime = [&suites, effectH](int stream_index, double time_secs, double time_scale) -> double {
        PF_Err local_err = PF_Err_NONE;
        double value_out = 0.0;

        AEGP_StreamRefH stream_ref = NULL;
        A_Time time;
        time.value = (A_long)(time_secs * time_scale);
        time.scale = (A_long)time_scale;

        local_err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(
            NULL,
            effectH,
            stream_index,
            &stream_ref);

        if (!local_err && stream_ref) {
            AEGP_StreamValue2 stream_value;
            local_err = suites.StreamSuite5()->AEGP_GetNewStreamValue(
                NULL,
                stream_ref,
                AEGP_LTimeMode_LayerTime,
                &time,
                FALSE,
                &stream_value);

            if (!local_err) {
                AEGP_StreamType stream_type;
                local_err = suites.StreamSuite5()->AEGP_GetStreamType(stream_ref, &stream_type);

                if (!local_err) {
                    switch (stream_type) {
                    case AEGP_StreamType_OneD:
                        value_out = stream_value.val.one_d;
                        break;

                    case AEGP_StreamType_TwoD:
                    case AEGP_StreamType_TwoD_SPATIAL:
                        value_out = stream_value.val.two_d.x;
                        break;

                    case AEGP_StreamType_ThreeD:
                    case AEGP_StreamType_ThreeD_SPATIAL:
                        value_out = stream_value.val.three_d.x;
                        break;

                    case AEGP_StreamType_COLOR:
                        value_out = stream_value.val.color.redF;
                        break;
                    }
                }

                suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
            }

            suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
        }

        return value_out;
        };

    auto valueAtTimeHz = [&valueAtTime, &has_frequency_keyframes, FREQUENCY_PARAM](
        int stream_index, double time_secs, double duration, double time_scale, double layer_start_time) -> double {

            double value_out = valueAtTime(stream_index, time_secs, time_scale);

            if (stream_index == FREQUENCY_PARAM && has_frequency_keyframes) {
                bool isHz = true;

                if (isHz) {
                    double accumulated_phase = 0.0;
                    double fps = 120.0;
                    int totalSteps = (int)round(duration * fps);
                    int curSteps = (int)round(fps * time_secs);

                    if (curSteps >= 0) {
                        for (int i = 0; i <= curSteps; i++) {
                            double stepValue;
                            double adjusted_time = i / fps + layer_start_time;
                            stepValue = valueAtTime(stream_index, adjusted_time, time_scale);
                            accumulated_phase += stepValue / fps;
                        }
                        value_out = accumulated_phase;
                    }
                }
            }

            return value_out;
        };

    double time_scale = current_time.scale;
    double duration = current_time_secs;

    double prev_accumulated_phase = 0.0;
    double current_accumulated_phase = 0.0;
    double next_accumulated_phase = 0.0;

    if (has_frequency_keyframes && normal_enabled) {
        prev_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, prev_time_secs, prev_time_secs, time_scale, layer_time_offset);
        current_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, current_time_secs, current_time_secs, time_scale, layer_time_offset);
        next_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, next_time_secs, next_time_secs, time_scale, layer_time_offset);
    }

    auto TriangleWave = [](double t) -> double {
        t = fmod(t + 0.75, 1.0);
        if (t < 0) t += 1.0;
        return (fabs(t - 0.5) - 0.25) * 4.0;
        };

    double prev_m, current_m, next_m;
    double prev_t, current_t, next_t;
    double prev_angle, current_angle, next_angle;

    if (compatibility_enabled) {
        A_Time layer_in_point;
        double layer_in_point_secs = 0.0;
        err = suites.LayerSuite7()->AEGP_GetLayerInPoint(layerH, AEGP_LTimeMode_LayerTime, &layer_in_point);
        if (!err) {
            layer_in_point_secs = (double)layer_in_point.value / (double)layer_in_point.scale;
        }

        double prev_compat_time_secs = prev_time_secs - layer_in_point_secs;
        double current_compat_time_secs = current_time_secs - layer_in_point_secs;
        double next_compat_time_secs = next_time_secs - layer_in_point_secs;

        double prev_compat_frequency = compat_frequency;
        double prev_compat_angle1 = compat_angle1;
        double prev_compat_angle2 = compat_angle2;
        double prev_compat_phase = compat_phase;

        double next_compat_frequency = compat_frequency;
        double next_compat_angle1 = compat_angle1;
        double next_compat_angle2 = compat_angle2;
        double next_compat_phase = compat_phase;

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_FREQUENCY_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &prev_time, FALSE, &value);
            if (!err) {
                prev_compat_frequency = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_ANGLE1_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &prev_time, FALSE, &value);
            if (!err) {
                prev_compat_angle1 = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_ANGLE2_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &prev_time, FALSE, &value);
            if (!err) {
                prev_compat_angle2 = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_PHASE_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &prev_time, FALSE, &value);
            if (!err) {
                prev_compat_phase = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_FREQUENCY_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &next_time, FALSE, &value);
            if (!err) {
                next_compat_frequency = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_ANGLE1_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &next_time, FALSE, &value);
            if (!err) {
                next_compat_angle1 = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_ANGLE2_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &next_time, FALSE, &value);
            if (!err) {
                next_compat_angle2 = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_PHASE_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &next_time, FALSE, &value);
            if (!err) {
                next_compat_phase = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        auto calculateCompatWaveValue = [&TriangleWave](int wave_type, double frequency, double time, double phase) -> double {
            double m;
            if (wave_type == 0) {
                m = sin((time * frequency + phase) * M_PI);
            }
            else {
                m = TriangleWave(((time * frequency) + phase) / 2.0);
            }
            return m;
            };

        prev_m = calculateCompatWaveValue(compat_wave_type, prev_compat_frequency, prev_compat_time_secs, prev_compat_phase);
        current_m = calculateCompatWaveValue(compat_wave_type, compat_frequency, current_compat_time_secs, compat_phase);
        next_m = calculateCompatWaveValue(compat_wave_type, next_compat_frequency, next_compat_time_secs, next_compat_phase);

        prev_t = (prev_m + 1.0) / 2.0;
        current_t = (current_m + 1.0) / 2.0;
        next_t = (next_m + 1.0) / 2.0;

        prev_angle = -(prev_compat_angle1 + prev_t * (prev_compat_angle2 - prev_compat_angle1));
        current_angle = -(compat_angle1 + current_t * (compat_angle2 - compat_angle1));
        next_angle = -(next_compat_angle1 + next_t * (next_compat_angle2 - next_compat_angle1));
    }
    else {
        double prev_phase_value = phase;
        double next_phase_value = phase;
        double prev_angle1_value = angle1;
        double prev_angle2_value = angle2;
        double next_angle1_value = angle1;
        double next_angle2_value = angle2;

        if (has_phase_keyframes) {
            prev_phase_value = valueAtTime(PHASE_PARAM, prev_time_secs, time_scale);
            next_phase_value = valueAtTime(PHASE_PARAM, next_time_secs, time_scale);
        }

        if (has_angle1_keyframes) {
            prev_angle1_value = valueAtTime(ANGLE1_PARAM, prev_time_secs, time_scale);
            next_angle1_value = valueAtTime(ANGLE1_PARAM, next_time_secs, time_scale);
        }

        if (has_angle2_keyframes) {
            prev_angle2_value = valueAtTime(ANGLE2_PARAM, prev_time_secs, time_scale);
            next_angle2_value = valueAtTime(ANGLE2_PARAM, next_time_secs, time_scale);
        }
        auto calculateWaveValue = [&TriangleWave](int wave_type, double frequency, double time, double phase, double accumulated_phase = 0.0) -> double {
            double X, m;

            if (accumulated_phase > 0.0) {
                if (wave_type == 0) {
                    m = sin((accumulated_phase + phase) * M_PI);
                }
                else {
                    m = TriangleWave((accumulated_phase + phase) / 2.0);
                }
            }
            else {
                if (wave_type == 0) {
                    X = (frequency * time) + phase;
                    m = sin(X * M_PI);
                }
                else {
                    X = ((frequency * time) + phase) / 2.0 + phase;
                    m = TriangleWave(X);
                }
            }

            return m;
            };

        if (has_frequency_keyframes) {
            prev_m = calculateWaveValue(wave_type, 0, 0, prev_phase_value, prev_accumulated_phase);
            current_m = calculateWaveValue(wave_type, 0, 0, phase, current_accumulated_phase);
            next_m = calculateWaveValue(wave_type, 0, 0, next_phase_value, next_accumulated_phase);
        }
        else {
            prev_m = calculateWaveValue(wave_type, frequency, prev_time_secs, prev_phase_value);
            current_m = calculateWaveValue(wave_type, frequency, current_time_secs, phase);
            next_m = calculateWaveValue(wave_type, frequency, next_time_secs, next_phase_value);
        }

        prev_t = (prev_m + 1.0) / 2.0;
        current_t = (current_m + 1.0) / 2.0;
        next_t = (next_m + 1.0) / 2.0;

        prev_angle = -(prev_angle1_value + prev_t * (prev_angle2_value - prev_angle1_value));
        current_angle = -(angle1 + current_t * (angle2 - angle1));
        next_angle = -(next_angle1_value + next_t * (next_angle2_value - next_angle1_value));
    }

    double prev_angle_rad = prev_angle * M_PI / 180.0;
    double current_angle_rad = current_angle * M_PI / 180.0;
    double next_angle_rad = next_angle * M_PI / 180.0;

    double prev_to_current_angle = current_angle_rad - prev_angle_rad;
    double current_to_next_angle = next_angle_rad - current_angle_rad;

    double angle_change = fabs(current_to_next_angle) > fabs(prev_to_current_angle) ?
        current_to_next_angle : prev_to_current_angle;

    *rotation_angle = angle_change;

    if (fabs(*rotation_angle) > 0.01) {
        found_motion = true;
    }
}

if (strstr(match_name, "DKT Pulse Size")) {
    double frequency = 2.00;
    double shrink = 0.90;
    double grow = 1.10;
    double phase = 0.00;
    A_long wave_type = 0;
    bool x_tiles = false;
    bool y_tiles = false;
    bool mirror = false;
    bool has_frequency_keyframes = false;
    bool has_phase_keyframes = false;
    bool has_shrink_keyframes = false;
    bool has_grow_keyframes = false;
    bool normal_enabled = true;
    bool compatibility_enabled = false;
    double compat_frequency = 2.00;
    double compat_shrink = 0.90;
    double compat_grow = 1.10;
    double compat_phase = 0.00;
    A_long compat_wave_type = 0;

    A_long num_params = 0;
    err = suites.StreamSuite5()->AEGP_GetEffectNumParamStreams(effectH, &num_params);
    if (err) {
        suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
        continue;
    }

    const int NORMAL_CHECKBOX_PARAM = 1;
    const int FREQUENCY_PARAM = 2;
    const int SHRINK_PARAM = 3;
    const int GROW_PARAM = 4;
    const int PHASE_PARAM = 5;
    const int WAVE_TYPE_PARAM = 6;
    const int X_TILES_PARAM = 8;
    const int Y_TILES_PARAM = 9;
    const int MIRROR_PARAM = 10;
    const int COMPATIBILITY_CHECKBOX_PARAM = 13;
    const int COMPATIBILITY_FREQUENCY_PARAM = 14;
    const int COMPATIBILITY_SHRINK_PARAM = 15;
    const int COMPATIBILITY_GROW_PARAM = 16;
    const int COMPATIBILITY_PHASE_PARAM = 17;
    const int COMPATIBILITY_WAVE_TYPE_PARAM = 18;

    if (num_params < 19) {
        suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
        continue;
    }

    AEGP_StreamRefH streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, NORMAL_CHECKBOX_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            normal_enabled = (value.val.one_d != 0);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_CHECKBOX_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            compatibility_enabled = (value.val.one_d != 0);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    if ((normal_enabled && compatibility_enabled) || (!normal_enabled && !compatibility_enabled)) {
        suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
        continue;
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, FREQUENCY_PARAM, &streamH);
    if (!err && streamH) {
        A_long num_keyframes = 0;
        err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
        if (!err && num_keyframes > 0) {
            has_frequency_keyframes = true;
        }

        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            frequency = value.val.one_d;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SHRINK_PARAM, &streamH);
    if (!err && streamH) {
        A_long num_keyframes = 0;
        err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
        if (!err && num_keyframes > 0) {
            has_shrink_keyframes = true;
        }

        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            shrink = value.val.one_d;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, GROW_PARAM, &streamH);
    if (!err && streamH) {
        A_long num_keyframes = 0;
        err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
        if (!err && num_keyframes > 0) {
            has_grow_keyframes = true;
        }

        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            grow = value.val.one_d;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, PHASE_PARAM, &streamH);
    if (!err && streamH) {
        A_long num_keyframes = 0;
        err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(streamH, &num_keyframes);
        if (!err && num_keyframes > 0) {
            has_phase_keyframes = true;
        }

        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            phase = value.val.one_d;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, WAVE_TYPE_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            wave_type = (A_long)value.val.one_d - 1;
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, X_TILES_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            x_tiles = (value.val.one_d > 0.5);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, Y_TILES_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            y_tiles = (value.val.one_d > 0.5);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    streamH = NULL;
    err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MIRROR_PARAM, &streamH);
    if (!err && streamH) {
        AEGP_StreamValue2 value;
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
            &current_time, FALSE, &value);
        if (!err) {
            mirror = (value.val.one_d > 0.5);
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
        }
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    if (compatibility_enabled) {
        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_FREQUENCY_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_frequency = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_SHRINK_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_shrink = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_GROW_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_grow = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_PHASE_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_phase = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_WAVE_TYPE_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;
            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &current_time, FALSE, &value);
            if (!err) {
                compat_wave_type = (A_long)value.val.one_d - 1;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }
            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }
    }

    double prev_time_secs = (double)prev_time.value / (double)prev_time.scale;
    double current_time_secs = (double)current_time.value / (double)current_time.scale;
    double next_time_secs = (double)next_time.value / (double)next_time.scale;

    AEGP_LayerIDVal layer_id = 0;
    err = suites.LayerSuite7()->AEGP_GetLayerID(layerH, &layer_id);

    double layer_time_offset = 0;
    A_Ratio stretch_factor = { 1, 1 };

    if (!err && layer_id != 0) {
        A_Time in_point;
        err = suites.LayerSuite7()->AEGP_GetLayerInPoint(layerH, AEGP_LTimeMode_LayerTime, &in_point);

        if (!err) {
            layer_time_offset = (double)in_point.value / (double)in_point.scale;
        }

        err = suites.LayerSuite7()->AEGP_GetLayerStretch(layerH, &stretch_factor);
    }

    double layer_relative_prev_time = prev_time_secs - layer_time_offset;
    double layer_relative_current_time = current_time_secs - layer_time_offset;
    double layer_relative_next_time = next_time_secs - layer_time_offset;

    double stretch_ratio = (double)stretch_factor.num / (double)stretch_factor.den;
    prev_time_secs *= stretch_ratio;
    current_time_secs *= stretch_ratio;
    next_time_secs *= stretch_ratio;

    auto valueAtTime = [&suites, effectH](int stream_index, double time_secs, double time_scale) -> double {
        PF_Err local_err = PF_Err_NONE;
        double value_out = 0.0;

        AEGP_StreamRefH stream_ref = NULL;
        A_Time time;
        time.value = (A_long)(time_secs * time_scale);
        time.scale = (A_long)time_scale;

        local_err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(
            NULL,
            effectH,
            stream_index,
            &stream_ref);

        if (!local_err && stream_ref) {
            AEGP_StreamValue2 stream_value;
            local_err = suites.StreamSuite5()->AEGP_GetNewStreamValue(
                NULL,
                stream_ref,
                AEGP_LTimeMode_LayerTime,
                &time,
                FALSE,
                &stream_value);

            if (!local_err) {
                AEGP_StreamType stream_type;
                local_err = suites.StreamSuite5()->AEGP_GetStreamType(stream_ref, &stream_type);

                if (!local_err) {
                    switch (stream_type) {
                    case AEGP_StreamType_OneD:
                        value_out = stream_value.val.one_d;
                        break;

                    case AEGP_StreamType_TwoD:
                    case AEGP_StreamType_TwoD_SPATIAL:
                        value_out = stream_value.val.two_d.x;
                        break;

                    case AEGP_StreamType_ThreeD:
                    case AEGP_StreamType_ThreeD_SPATIAL:
                        value_out = stream_value.val.three_d.x;
                        break;

                    case AEGP_StreamType_COLOR:
                        value_out = stream_value.val.color.redF;
                        break;
                    }
                }

                suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
            }

            suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
        }

        return value_out;
        };

    auto valueAtTimeHz = [&valueAtTime, &has_frequency_keyframes, FREQUENCY_PARAM](
        int stream_index, double time_secs, double duration, double time_scale, double layer_start_time) -> double {

            double value_out = valueAtTime(stream_index, time_secs, time_scale);

            if (stream_index == FREQUENCY_PARAM && has_frequency_keyframes) {
                bool isHz = true;

                if (isHz) {
                    double accumulated_phase = 0.0;
                    double fps = 120.0;
                    int totalSteps = (int)round(duration * fps);
                    int curSteps = (int)round(fps * time_secs);

                        for (int i = 0; i <= curSteps; i++) {
                            double stepValue;
                            double adjusted_time = i / fps + layer_start_time;
                            stepValue = valueAtTime(stream_index, adjusted_time, time_scale);
                            accumulated_phase += stepValue / fps;
                        
                        value_out = accumulated_phase;
                    }
                }
            }

            return value_out;
        };

    double time_scale = current_time.scale;
    double duration = current_time_secs;

    double prev_accumulated_phase = 0.0;
    double current_accumulated_phase = 0.0;
    double next_accumulated_phase = 0.0;

    if (has_frequency_keyframes && normal_enabled) {
        prev_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, prev_time_secs, prev_time_secs, time_scale, layer_time_offset);
        current_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, current_time_secs, current_time_secs, time_scale, layer_time_offset);
        next_accumulated_phase = valueAtTimeHz(FREQUENCY_PARAM, next_time_secs, next_time_secs, time_scale, layer_time_offset);
    }

    double prev_frequency = frequency, next_frequency = frequency;
    double prev_shrink = shrink, next_shrink = shrink;
    double prev_grow = grow, next_grow = grow;
    double prev_phase = phase, next_phase = phase;
    double prev_compat_frequency = compat_frequency, next_compat_frequency = compat_frequency;
    double prev_compat_shrink = compat_shrink, next_compat_shrink = compat_shrink;
    double prev_compat_grow = compat_grow, next_compat_grow = compat_grow;
    double prev_compat_phase = compat_phase, next_compat_phase = compat_phase;

    if (has_frequency_keyframes) {
        prev_frequency = valueAtTime(FREQUENCY_PARAM, prev_time_secs, time_scale);
        next_frequency = valueAtTime(FREQUENCY_PARAM, next_time_secs, time_scale);
    }

    if (has_shrink_keyframes) {
        prev_shrink = valueAtTime(SHRINK_PARAM, prev_time_secs, time_scale);
        next_shrink = valueAtTime(SHRINK_PARAM, next_time_secs, time_scale);
    }

    if (has_grow_keyframes) {
        prev_grow = valueAtTime(GROW_PARAM, prev_time_secs, time_scale);
        next_grow = valueAtTime(GROW_PARAM, next_time_secs, time_scale);
    }

    if (has_phase_keyframes) {
        prev_phase = valueAtTime(PHASE_PARAM, prev_time_secs, time_scale);
        next_phase = valueAtTime(PHASE_PARAM, next_time_secs, time_scale);
    }

    if (compatibility_enabled) {
        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_FREQUENCY_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;

            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &prev_time, FALSE, &value);
            if (!err) {
                prev_compat_frequency = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }

            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &next_time, FALSE, &value);
            if (!err) {
                next_compat_frequency = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }

            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_SHRINK_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;

            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &prev_time, FALSE, &value);
            if (!err) {
                prev_compat_shrink = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }

            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &next_time, FALSE, &value);
            if (!err) {
                next_compat_shrink = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }

            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_GROW_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;

            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &prev_time, FALSE, &value);
            if (!err) {
                prev_compat_grow = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }

            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &next_time, FALSE, &value);
            if (!err) {
                next_compat_grow = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }

            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }

        streamH = NULL;
        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, COMPATIBILITY_PHASE_PARAM, &streamH);
        if (!err && streamH) {
            AEGP_StreamValue2 value;

            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &prev_time, FALSE, &value);
            if (!err) {
                prev_compat_phase = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }

            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                &next_time, FALSE, &value);
            if (!err) {
                next_compat_phase = value.val.one_d;
                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
            }

            suites.StreamSuite5()->AEGP_DisposeStream(streamH);
        }
    }

    auto calculateWaveValue = [layer_time_offset](int wave_type, double frequency, double time, double phase, double accumulated_phase = 0.0) -> double {
        double X, m;

        if (accumulated_phase > 0.0) {
            if (wave_type == 0) {
                m = sin((accumulated_phase + phase) * M_PI);
            }
            else {
                double t = fmod(((accumulated_phase + phase) / 2.0) + 0.75, 1.0);
                if (t < 0) t += 1.0;
                m = (fabs(t - 0.5) - 0.25) * 4.0;
            }
        }
        else {
            if (wave_type == 0) {
                X = (frequency * time) + phase;
                m = sin(X * M_PI);
            }
            else {
                X = ((frequency * time) + phase) / 2.0 + phase;
                double t = fmod(X + 0.75, 1.0);
                if (t < 0) t += 1.0;
                m = (fabs(t - 0.5) - 0.25) * 4.0;
            }
        }

        return m;
        };

    auto TriangleWave = [](double t) -> double {
        t = fmod(t + 0.75, 1.0);
        if (t < 0) t += 1.0;
        double result = (fabs(t - 0.5) - 0.25) * 4.0;
        return result;
        };

    double prev_wave, current_wave, next_wave;
    double prev_range, current_range, next_range;
    double prev_scale, current_scale, next_scale;

    if (normal_enabled) {
        if (has_frequency_keyframes) {
            prev_wave = calculateWaveValue(wave_type, 0, 0, prev_phase, prev_accumulated_phase);
            current_wave = calculateWaveValue(wave_type, 0, 0, phase, current_accumulated_phase);
            next_wave = calculateWaveValue(wave_type, 0, 0, next_phase, next_accumulated_phase);
        }
        else {
            prev_wave = calculateWaveValue(wave_type, prev_frequency, prev_time_secs, prev_phase);
            current_wave = calculateWaveValue(wave_type, frequency, current_time_secs, phase);
            next_wave = calculateWaveValue(wave_type, next_frequency, next_time_secs, next_phase);
        }

        prev_range = prev_grow - prev_shrink;
        current_range = grow - shrink;
        next_range = next_grow - next_shrink;

        prev_scale = (prev_range * ((prev_wave + 1.0) / 2.0)) + prev_shrink;
        current_scale = (current_range * ((current_wave + 1.0) / 2.0)) + shrink;
        next_scale = (next_range * ((next_wave + 1.0) / 2.0)) + next_shrink;
    }
    else {
        auto calculateCompatWaveValue = [&TriangleWave](int wave_type, double frequency, double time, double phase) -> double {
            double m;
            if (wave_type == 0) {
                m = sin((time * frequency + phase) * M_PI);
            }
            else {
                m = TriangleWave(((time * frequency) + phase) / 2.0);
            }
            return m;
            };

        prev_wave = calculateCompatWaveValue(compat_wave_type, prev_compat_frequency, prev_time_secs, prev_compat_phase);
        current_wave = calculateCompatWaveValue(compat_wave_type, compat_frequency, current_time_secs, compat_phase);
        next_wave = calculateCompatWaveValue(compat_wave_type, next_compat_frequency, next_time_secs, next_compat_phase);

        prev_range = prev_compat_grow - prev_compat_shrink;
        current_range = compat_grow - compat_shrink;
        next_range = next_compat_grow - next_compat_shrink;

        prev_scale = (prev_range * ((prev_wave + 1.0) / 2.0)) + prev_compat_shrink;
        current_scale = (current_range * ((current_wave + 1.0) / 2.0)) + compat_shrink;
        next_scale = (next_range * ((next_wave + 1.0) / 2.0)) + next_compat_shrink;
    }

    double prev_to_current_scale = current_scale - prev_scale;
    double current_to_next_scale = next_scale - current_scale;

    AEGP_CompH compH = NULL;
    err = suites.LayerSuite7()->AEGP_GetLayerParentComp(layerH, &compH);
    if (!err && compH) {
        AEGP_ItemH itemH = NULL;
        err = suites.CompSuite11()->AEGP_GetItemFromComp(compH, &itemH);

        A_long width = 0, height = 0;
        if (!err && itemH) {
            err = suites.ItemSuite9()->AEGP_GetItemDimensions(itemH, &width, &height);

            if (!err) {
                double scale_change = fabs(current_to_next_scale) > fabs(prev_to_current_scale) ?
                    current_to_next_scale : prev_to_current_scale;

                float max_dimension = MAX(width, height) * 0.5f;
                float displacement = max_dimension * scale_change;

                *scale_velocity_out = displacement * 3.0;

                if (fabs(*scale_velocity_out) > 0.01f) {
                    found_motion = true;
                }
            }
        }
    }
}

        if (strstr(match_name, "DKT Random Displacement")) {
            double magnitude = 0.0;
            double evolution = 0.0;
            double seed = 0.0;
            double scatter = 0.0;
            bool x_tiles = false;
            bool y_tiles = false;
            bool mirror = false;

            A_long num_params = 0;
            err = suites.StreamSuite5()->AEGP_GetEffectNumParamStreams(effectH, &num_params);
            if (err) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            const int MAGNITUDE_PARAM = 1;
            const int EVOLUTION_PARAM = 2;
            const int SEED_PARAM = 3;
            const int SCATTER_PARAM = 4;
            const int X_TILES_PARAM = 6;
            const int Y_TILES_PARAM = 7;
            const int MIRROR_PARAM = 8;
            if (num_params < 10) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            AEGP_StreamRefH streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MAGNITUDE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    magnitude = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, EVOLUTION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    evolution = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SEED_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    seed = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SCATTER_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    scatter = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, X_TILES_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    x_tiles = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, Y_TILES_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    y_tiles = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MIRROR_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    mirror = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            if (magnitude < 0.1) {
                continue;
            }

            double centerX = 0.0;
            double centerY = 0.0;

            AEGP_CompH compH = NULL;
            err = suites.LayerSuite7()->AEGP_GetLayerParentComp(layerH, &compH);
            if (!err && compH) {
                AEGP_ItemH itemH = NULL;
                err = suites.CompSuite11()->AEGP_GetItemFromComp(compH, &itemH);
                if (!err && itemH) {
                    A_long width = 0, height = 0;
                    err = suites.ItemSuite9()->AEGP_GetItemDimensions(itemH, &width, &height);
                    if (!err) {
                        centerX = width / 2.0;
                        centerY = height / 2.0;
                    }
                }
            }

            double prev_evolution = evolution, next_evolution = evolution;
            double prev_seed = seed, next_seed = seed;
            double prev_magnitude = magnitude, next_magnitude = magnitude;
            double prev_scatter = scatter, next_scatter = scatter;

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, EVOLUTION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_evolution = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, EVOLUTION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_evolution = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SEED_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_seed = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SEED_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_seed = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MAGNITUDE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_magnitude = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MAGNITUDE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_magnitude = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SCATTER_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_scatter = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SCATTER_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_scatter = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            double prev_noise_dx = SimplexNoise::simplex_noise(centerX * prev_scatter / 50.0 + prev_seed * 54623.245, centerY * prev_scatter / 500.0, prev_evolution + prev_seed * 49235.319798, 3);

            double prev_noise_dy = SimplexNoise::simplex_noise(centerX * prev_scatter / 50.0, centerY * prev_scatter / 500.0 + prev_seed * 8723.5647, prev_evolution + 7468.329 + prev_seed * 19337.940385, 3);

            double curr_noise_dx = SimplexNoise::simplex_noise(centerX * scatter / 50.0 + seed * 54623.245, centerY * scatter / 500.0, evolution + seed * 49235.319798, 3);

            double curr_noise_dy = SimplexNoise::simplex_noise(centerX * scatter / 50.0, centerY * scatter / 500.0 + seed * 8723.5647, evolution + 7468.329 + seed * 19337.940385, 3);

            double next_noise_dx = SimplexNoise::simplex_noise(centerX * next_scatter / 50.0 + next_seed * 54623.245, centerY * next_scatter / 500.0, next_evolution + next_seed * 49235.319798, 3);

            double next_noise_dy = SimplexNoise::simplex_noise(centerX * next_scatter / 50.0, centerY * next_scatter / 500.0 + next_seed * 8723.5647, next_evolution + 7468.329 + next_seed * 19337.940385, 3);

            double prev_dx = -prev_magnitude * prev_noise_dx;
            double prev_dy = prev_magnitude * prev_noise_dy;

            double curr_dx = -magnitude * curr_noise_dx;
            double curr_dy = magnitude * curr_noise_dy;

            double next_dx = -next_magnitude * next_noise_dx;
            double next_dy = next_magnitude * next_noise_dy;

            double prev_to_current_x = curr_dx - prev_dx;
            double prev_to_current_y = curr_dy - prev_dy;
            double current_to_next_x = next_dx - curr_dx;
            double current_to_next_y = next_dy - curr_dy;

            double prev_to_current_magnitude = sqrt(prev_to_current_x * prev_to_current_x +
                prev_to_current_y * prev_to_current_y);
            double current_to_next_magnitude = sqrt(current_to_next_x * current_to_next_x +
                current_to_next_y * current_to_next_y);

            if (prev_to_current_magnitude > 0.01 || current_to_next_magnitude > 0.01) {
                if (current_to_next_magnitude > prev_to_current_magnitude) {
                    *motion_x += current_to_next_x;
                    *motion_y += current_to_next_y;
                }
                else {
                    *motion_x += prev_to_current_x;
                    *motion_y += prev_to_current_y;
                }

                if (fabs(*motion_x) > 0.01 || fabs(*motion_y) > 0.01) {
                    found_motion = true;
                }
            }
        }
        if (strstr(match_name, "DKT Transform")) {
            float x_pos = 0.0f;
            float y_pos = 0.0f;
            float rotation = 0.0f;
            float scale = 100.0f;
            bool x_tiles = false;
            bool y_tiles = false;
            bool mirror = false;

            A_long num_params = 0;
            err = suites.StreamSuite5()->AEGP_GetEffectNumParamStreams(effectH, &num_params);
            if (err) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            const int POSITION_PARAM = 1;
            const int ROTATION_PARAM = 2;
            const int SCALE_PARAM = 3;
            const int X_TILES_PARAM = 5;
            const int Y_TILES_PARAM = 6;
            const int MIRROR_PARAM = 7;

            if (num_params < 8) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            AEGP_StreamRefH streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, POSITION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    x_pos = value.val.two_d.x;
                    y_pos = value.val.two_d.y;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, ROTATION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    rotation = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SCALE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    scale = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, X_TILES_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    x_tiles = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, Y_TILES_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    y_tiles = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MIRROR_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    mirror = (value.val.one_d > 0.5);
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            float prev_x_pos = x_pos, prev_y_pos = y_pos, prev_rotation = rotation, prev_scale = scale;
            float next_x_pos = x_pos, next_y_pos = y_pos, next_rotation = rotation, next_scale = scale;

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, POSITION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_x_pos = value.val.two_d.x;
                    prev_y_pos = value.val.two_d.y;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_x_pos = value.val.two_d.x;
                    next_y_pos = value.val.two_d.y;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, ROTATION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_rotation = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_rotation = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, SCALE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &prev_time, FALSE, &value);
                if (!err) {
                    prev_scale = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }

                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_LayerTime,
                    &next_time, FALSE, &value);
                if (!err) {
                    next_scale = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            double prev_to_current_x = x_pos - prev_x_pos;
            double prev_to_current_y = y_pos - prev_y_pos;
            double current_to_next_x = next_x_pos - x_pos;
            double current_to_next_y = next_y_pos - y_pos;

            double prev_to_current_magnitude = sqrt(prev_to_current_x * prev_to_current_x + prev_to_current_y * prev_to_current_y);
            double current_to_next_magnitude = sqrt(current_to_next_x * current_to_next_x + current_to_next_y * current_to_next_y);

            if (current_to_next_magnitude > prev_to_current_magnitude) {
                *motion_x += current_to_next_x;
                *motion_y += current_to_next_y;
            }
            else {
                *motion_x += prev_to_current_x;
                *motion_y += prev_to_current_y;
            }

            double prev_to_current_rotation = rotation - prev_rotation;
            double current_to_next_rotation = next_rotation - rotation;

            if (prev_to_current_rotation > 180.0) prev_to_current_rotation -= 360.0;
            if (prev_to_current_rotation < -180.0) prev_to_current_rotation += 360.0;
            if (current_to_next_rotation > 180.0) current_to_next_rotation -= 360.0;
            if (current_to_next_rotation < -180.0) current_to_next_rotation += 360.0;

            double rotation_change = fabs(current_to_next_rotation) > fabs(prev_to_current_rotation) ?
                current_to_next_rotation : prev_to_current_rotation;

            *rotation_angle += rotation_change * M_PI / 180.0;

            double prev_to_current_scale = scale - prev_scale;
            double current_to_next_scale = next_scale - scale;

            double scale_change = fabs(current_to_next_scale) > fabs(prev_to_current_scale) ?
                current_to_next_scale : prev_to_current_scale;

            *scale_x = scale_change / 100.0;
            *scale_y = scale_change / 100.0;

            AEGP_CompH compH = NULL;
            err = suites.LayerSuite7()->AEGP_GetLayerParentComp(layerH, &compH);
            if (!err && compH) {
                AEGP_ItemH itemH = NULL;
                err = suites.CompSuite11()->AEGP_GetItemFromComp(compH, &itemH);

                A_long width = 0, height = 0;
                if (!err && itemH) {
                    err = suites.ItemSuite9()->AEGP_GetItemDimensions(itemH, &width, &height);

                    if (!err) {
                        float layer_width = (float)width;
                        float layer_height = (float)height;

                        float current_size = sqrt(pow(layer_width * (scale / 100.0f), 2) +
                            pow(layer_height * (scale / 100.0f), 2));

                        float prev_size = sqrt(pow(layer_width * (prev_scale / 100.0f), 2) +
                            pow(layer_height * (prev_scale / 100.0f), 2));

                        float next_size = sqrt(pow(layer_width * (next_scale / 100.0f), 2) +
                            pow(layer_height * (next_scale / 100.0f), 2));

                        if (fabs(next_size - current_size) > fabs(current_size - prev_size)) {
                            *scale_velocity_out = next_size - current_size;
                        }
                        else {
                            *scale_velocity_out = current_size - prev_size;
                        }

                        *scale_velocity_out *= 1.6f;
                    }
                }
            }

            if (fabs(*motion_x) > 0.01 || fabs(*motion_y) > 0.01 ||
                fabs(*rotation_angle) > 0.01 || fabs(*scale_velocity_out) > 0.01f) {
                found_motion = true;
            }
        }

        suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
    }

    return found_motion;
}


static PF_Err DetectChanges(
    PF_InData* in_data,
    double* motion_x_prev_curr, double* motion_y_prev_curr,
    double* motion_x_curr_next, double* motion_y_curr_next,
    double* scale_x_prev_curr, double* scale_y_prev_curr,
    double* scale_x_curr_next, double* scale_y_curr_next,
    double* rotation_prev_curr, double* rotation_curr_next,
    bool* has_motion_prev_curr, bool* has_motion_curr_next,
    bool* has_scale_change_prev_curr, bool* has_scale_change_curr_next,
    bool* has_rotation_prev_curr, bool* has_rotation_curr_next,
    float* scale_velocity,
    AEGP_TwoDVal* anchor_point
)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    *motion_x_prev_curr = *motion_y_prev_curr = 0;
    *motion_x_curr_next = *motion_y_curr_next = 0;
    *scale_x_prev_curr = *scale_y_prev_curr = 0;
    *scale_x_curr_next = *scale_y_curr_next = 0;
    *rotation_prev_curr = *rotation_curr_next = 0;
    *has_motion_prev_curr = *has_motion_curr_next = false;
    *has_scale_change_prev_curr = *has_scale_change_curr_next = false;
    *has_rotation_prev_curr = *has_rotation_curr_next = false;
    *scale_velocity = 0.0f;
    anchor_point->x = anchor_point->y = 0;

    AEGP_PFInterfaceSuite1* pfInterfaceSuite = suites.PFInterfaceSuite1();
    if (!pfInterfaceSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_LayerH layerH = NULL;
    PF_Err err = pfInterfaceSuite->AEGP_GetEffectLayer(in_data->effect_ref, &layerH);
    if (err || !layerH) {
        return err;
    }

    AEGP_CompH compH = NULL;
    AEGP_LayerSuite7* layerSuite = suites.LayerSuite7();
    if (!layerSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    err = layerSuite->AEGP_GetLayerParentComp(layerH, &compH);
    if (err || !compH) {
        return err;
    }

    AEGP_ItemH itemH = NULL;
    err = layerSuite->AEGP_GetLayerSourceItem(layerH, &itemH);
    if (err || !itemH) {
        return err;
    }

    A_long width = 0, height = 0;
    AEGP_ItemSuite9* itemSuite = suites.ItemSuite9();
    if (!itemSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    err = itemSuite->AEGP_GetItemDimensions(itemH, &width, &height);
    if (err) {
        return err;
    }

    A_Time prev_time, current_time, next_time;

    current_time.value = in_data->current_time;
    current_time.scale = in_data->time_scale;

    prev_time.scale = in_data->time_scale;
    prev_time.value = in_data->current_time - in_data->time_step;

    next_time.scale = in_data->time_scale;
    next_time.value = in_data->current_time + in_data->time_step;

    AEGP_TwoDVal prev_pos, curr_pos, next_pos;
    err = GetLayerPosition(in_data, layerH, &prev_time, &prev_pos);
    if (!err) {
        err = GetLayerPosition(in_data, layerH, &current_time, &curr_pos);
    }
    if (!err) {
        err = GetLayerPosition(in_data, layerH, &next_time, &next_pos);
    }

    AEGP_TwoDVal prev_scale, curr_scale, next_scale;
    err = GetLayerScale(in_data, layerH, &prev_time, &prev_scale);
    if (!err) {
        err = GetLayerScale(in_data, layerH, &current_time, &curr_scale);
    }
    if (!err) {
        err = GetLayerScale(in_data, layerH, &next_time, &next_scale);
    }

    double prev_rotation = 0, curr_rotation = 0, next_rotation = 0;
    err = GetLayerRotation(in_data, layerH, &prev_time, &prev_rotation);
    if (!err) {
        err = GetLayerRotation(in_data, layerH, &current_time, &curr_rotation);
    }
    if (!err) {
        err = GetLayerRotation(in_data, layerH, &next_time, &next_rotation);
    }

    err = GetLayerAnchorPoint(in_data, layerH, &current_time, anchor_point);

    if (err) {
        return err;
    }

    *motion_x_prev_curr = curr_pos.x - prev_pos.x;
    *motion_y_prev_curr = curr_pos.y - prev_pos.y;
    *motion_x_curr_next = next_pos.x - curr_pos.x;
    *motion_y_curr_next = next_pos.y - curr_pos.y;

    *scale_x_prev_curr = curr_scale.x - prev_scale.x;
    *scale_y_prev_curr = curr_scale.y - prev_scale.y;
    *scale_x_curr_next = next_scale.x - curr_scale.x;
    *scale_y_curr_next = next_scale.y - curr_scale.y;

    *rotation_prev_curr = curr_rotation - prev_rotation;
    *rotation_curr_next = next_rotation - curr_rotation;

    if (*rotation_prev_curr > 180) {
        *rotation_prev_curr -= 360;
    }
    else if (*rotation_prev_curr < -180) {
        *rotation_prev_curr += 360;
    }

    if (*rotation_curr_next > 180) {
        *rotation_curr_next -= 360;
    }
    else if (*rotation_curr_next < -180) {
        *rotation_curr_next -= 360;
    }

    double scale_change_threshold = 0.1;     
    double motion_threshold = 0.5;
    double rotation_threshold = 0.1;      

    *has_motion_prev_curr = (sqrt((*motion_x_prev_curr) * (*motion_x_prev_curr) +
        (*motion_y_prev_curr) * (*motion_y_prev_curr)) > motion_threshold);
    *has_rotation_prev_curr = (fabs(*rotation_prev_curr) > rotation_threshold);

    *has_motion_curr_next = (sqrt((*motion_x_curr_next) * (*motion_x_curr_next) +
        (*motion_y_curr_next) * (*motion_y_curr_next)) > motion_threshold);
    *has_rotation_curr_next = (fabs(*rotation_curr_next) > rotation_threshold);

    float layer_width = (float)width;
    float layer_height = (float)height;

    float current_size_x = layer_width * (curr_scale.x / 100.0f);
    float current_size_y = layer_height * (curr_scale.y / 100.0f);

    float prev_size_x, prev_size_y;

    bool prev_curr_scale_change = (fabs(*scale_x_prev_curr) > scale_change_threshold ||
        fabs(*scale_y_prev_curr) > scale_change_threshold);
    bool curr_next_scale_change = (fabs(*scale_x_curr_next) > scale_change_threshold ||
        fabs(*scale_y_curr_next) > scale_change_threshold);

    if (prev_curr_scale_change) {
        prev_size_x = layer_width * (prev_scale.x / 100.0f);
        prev_size_y = layer_height * (prev_scale.y / 100.0f);
    }
    else if (curr_next_scale_change) {
        prev_size_x = layer_width * (next_scale.x / 100.0f);
        prev_size_y = layer_height * (next_scale.y / 100.0f);
    }
    else {
        prev_size_x = current_size_x;
        prev_size_y = current_size_y;
    }

    float current_length = sqrt(current_size_x * current_size_x + current_size_y * current_size_y);
    float prev_length = sqrt(prev_size_x * prev_size_x + prev_size_y * prev_size_y);

    *scale_velocity = current_length - prev_length;
    *scale_velocity *= 1.6f;

    float scale_velocity_threshold = 0.01f;
    *has_scale_change_prev_curr = fabs(*scale_velocity) > scale_velocity_threshold;
    *has_scale_change_curr_next = fabs(*scale_velocity) > scale_velocity_threshold;

    return PF_Err_NONE;
}

template <typename PixelType>
static PF_Err ApplyMotionBlurGeneric(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    DetectionData* data = (DetectionData*)refcon;

    if (!data->position_enabled ||
        (!data->has_motion_prev_curr && !data->has_motion_curr_next)) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    double motion_x = 0, motion_y = 0;
    if (data->has_motion_prev_curr) {
        motion_x = data->motion_x_prev_curr;
        motion_y = data->motion_y_prev_curr;
    }
    else {
        motion_x = data->motion_x_curr_next;
        motion_y = data->motion_y_curr_next;
    }

    if (data->in_data) {
        motion_x *= (double)data->in_data->downsample_x.num / (double)data->in_data->downsample_x.den;
        motion_y *= (double)data->in_data->downsample_y.num / (double)data->in_data->downsample_y.den;
    }

    float max_dimension = fmax(data->input_world->width, data->input_world->height);
    float texel_size_x = 1.0f / max_dimension;
    float texel_size_y = 1.0f / max_dimension;

    float velocity_x = motion_x * data->tune_value * 0.7f;
    float velocity_y = motion_y * data->tune_value * 0.7f;

    float normalized_velocity_x = velocity_x / data->input_world->width;
    float normalized_velocity_y = velocity_y / data->input_world->height;

    float speed = sqrt((normalized_velocity_x * normalized_velocity_x) +
        (normalized_velocity_y * normalized_velocity_y)) / texel_size_x;

    int nSamples = (int)fmin(100, fmax(2, speed));

    if (nSamples <= 1) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    float aspect_ratio = (float)data->input_world->width / (float)data->input_world->height;

    float accumR, accumG, accumB, accumA;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else {  
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }

    for (int i = 1; i < nSamples; i++) {
        float offset_factor = ((float)i / (float)(nSamples - 1)) - 0.5f;

        float offset_x = velocity_x * offset_factor;
        float offset_y = velocity_y * offset_factor;

        if (aspect_ratio != 1.0f) {
            offset_y *= aspect_ratio;
        }

        float sample_x_f = xL - offset_x;
        float sample_y_f = yL - offset_y;

        int sample_x = (int)(sample_x_f + 0.5f);
        int sample_y = (int)(sample_y_f + 0.5f);

        sample_x = fmax(0, fmin(data->input_world->width - 1, sample_x));
        sample_y = fmax(0, fmin(data->input_world->height - 1, sample_y));

        if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
            PF_PixelFloat* input_pixels = (PF_PixelFloat*)data->input_world->data;
            PF_PixelFloat sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_PixelFloat) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
            PF_Pixel16* input_pixels = (PF_Pixel16*)data->input_world->data;
            PF_Pixel16 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel16) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else {  
            PF_Pixel8* input_pixels = (PF_Pixel8*)data->input_world->data;
            PF_Pixel8 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel8) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
    }

    float inv_nSamples = 1.0f / nSamples;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        outP->red = accumR * inv_nSamples;
        outP->green = accumG * inv_nSamples;
        outP->blue = accumB * inv_nSamples;
        outP->alpha = accumA * inv_nSamples;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        outP->red = static_cast<A_u_short>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_short>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_short>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_short>(accumA * inv_nSamples + 0.5f);
    }
    else {  
        outP->red = static_cast<A_u_char>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_char>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_char>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_char>(accumA * inv_nSamples + 0.5f);
    }

    return PF_Err_NONE;
}



static PF_Err RenderFunc8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return ApplyMotionBlurGeneric<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err RenderFunc16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return ApplyMotionBlurGeneric<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err RenderFuncFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return ApplyMotionBlurGeneric<PF_PixelFloat>(refcon, xL, yL, inP, outP);
}

template <typename PixelType>
static PF_Err ApplyScaleBlurGeneric(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    DetectionData* data = (DetectionData*)refcon;
    float scale_velocity = data->scale_velocity;

    if (!data->scale_enabled || fabs(scale_velocity) < 0.0001f) {       
        *outP = *inP;
        return PF_Err_NONE;
    }

    float max_dimension = fmax(data->input_world->width, data->input_world->height);
    float texel_size_x = 1.0f / max_dimension;
    float texel_size_y = 1.0f / max_dimension;

    float cx = data->anchor_point.x;
    float cy = data->anchor_point.y;

    if (data->in_data) {
        cx *= (float)data->in_data->downsample_x.num / (float)data->in_data->downsample_x.den;
        cy *= (float)data->in_data->downsample_y.num / (float)data->in_data->downsample_y.den;
    }

    float norm_cx = cx / data->input_world->width;
    float norm_cy = cy / data->input_world->height;

    float norm_x = (float)xL / (float)data->input_world->width;
    float norm_y = (float)yL / (float)data->input_world->height;

    float v_x = norm_x - norm_cx;
    float v_y = norm_y - norm_cy;

    float aspect_ratio = (float)data->input_world->width / (float)data->input_world->height;
    v_y *= aspect_ratio;

    float speed = fabs(scale_velocity * data->tune_value) / 2.0f;

    float length_v = sqrt(v_x * v_x + v_y * v_y);
    speed *= length_v;

    int nSamples = (int)fmin(100.01f, fmax(1.01f, speed));

    if (nSamples <= 1) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    float vnorm_x = 0, vnorm_y = 0;
    if (length_v > 0.0001f) {
        vnorm_x = v_x / length_v;
        vnorm_y = v_y / length_v;
    }

    vnorm_x *= texel_size_x * speed;
    vnorm_y *= texel_size_y * speed;

    float accumR, accumG, accumB, accumA;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else {  
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }

    for (int i = 1; i < nSamples; i++) {
        float offset_factor = ((float)i / (float)(nSamples - 1)) - 0.5f;
        float offset_x = vnorm_x * offset_factor;
        float offset_y = vnorm_y * offset_factor;

        offset_y = offset_y * data->input_world->width / data->input_world->height;

        float sample_x_f = xL - (offset_x * data->input_world->width);
        float sample_y_f = yL - (offset_y * data->input_world->height);

        int sample_x = (int)(sample_x_f + 0.5f);
        int sample_y = (int)(sample_y_f + 0.5f);

        sample_x = fmax(0, fmin(data->input_world->width - 1, sample_x));
        sample_y = fmax(0, fmin(data->input_world->height - 1, sample_y));

        if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
            PF_PixelFloat* input_pixels = (PF_PixelFloat*)data->input_world->data;
            PF_PixelFloat sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_PixelFloat) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
            PF_Pixel16* input_pixels = (PF_Pixel16*)data->input_world->data;
            PF_Pixel16 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel16) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else {  
            PF_Pixel8* input_pixels = (PF_Pixel8*)data->input_world->data;
            PF_Pixel8 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel8) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
    }

    float inv_nSamples = 1.0f / nSamples;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        outP->red = accumR * inv_nSamples;
        outP->green = accumG * inv_nSamples;
        outP->blue = accumB * inv_nSamples;
        outP->alpha = accumA * inv_nSamples;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        outP->red = static_cast<A_u_short>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_short>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_short>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_short>(accumA * inv_nSamples + 0.5f);
    }
    else {  
        outP->red = static_cast<A_u_char>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_char>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_char>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_char>(accumA * inv_nSamples + 0.5f);
    }

    return PF_Err_NONE;
}




static PF_Err ScaleBlurFunc8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return ApplyScaleBlurGeneric<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err ScaleBlurFunc16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return ApplyScaleBlurGeneric<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err ScaleBlurFuncFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return ApplyScaleBlurGeneric<PF_PixelFloat>(refcon, xL, yL, inP, outP);
}


template <typename PixelType>
static PF_Err ApplyAngleBlurGeneric(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    DetectionData* data = (DetectionData*)refcon;

    if (!data->angle_enabled ||
        (!data->has_rotation_prev_curr && !data->has_rotation_curr_next && !data->has_effect_rotation)) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    double rotation = 0;
    bool is_effect_rotation = false;

    if (data->has_effect_rotation) {
        rotation = data->effect_rotation;
        is_effect_rotation = true;
    }
    else if (data->has_rotation_prev_curr) {
        rotation = data->rotation_prev_curr;
    }
    else {
        rotation = data->rotation_curr_next;
    }

    float angle_rad;
    if (is_effect_rotation) {
        angle_rad = fabs(rotation);
    }
    else {
        angle_rad = fabs(rotation) * (float)M_PI / 180.0f;
    }

    angle_rad *= data->tune_value;

    if (angle_rad < 0.001f) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    float width = (float)data->input_world->width;
    float height = (float)data->input_world->height;

    float cx = data->anchor_point.x;
    float cy = data->anchor_point.y;

    if (data->in_data) {
        cx *= (float)data->in_data->downsample_x.num / (float)data->in_data->downsample_x.den;
        cy *= (float)data->in_data->downsample_y.num / (float)data->in_data->downsample_y.den;
    }

    float dx = (float)xL - cx;
    float dy = (float)yL - cy;

    float distance = sqrt(dx * dx + dy * dy);

    if (distance < 0.01f) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    float dx_norm = dx / distance;
    float dy_norm = dy / distance;

    float arc_length = distance * angle_rad;

    int nSamples = (int)fmax(2, fmin(arc_length * 2.0f, 100.0f));

    float accumR = 0, accumG = 0, accumB = 0, accumA = 0;

    for (int i = 0; i < nSamples; i++) {
        float sample_angle = -angle_rad / 2.0f + angle_rad * (float)i / ((float)nSamples - 1.0f);

        float sin_angle = sin(sample_angle);
        float cos_angle = cos(sample_angle);

        float rotated_dx = dx_norm * cos_angle - dy_norm * sin_angle;
        float rotated_dy = dx_norm * sin_angle + dy_norm * cos_angle;

        float sample_x = cx + (rotated_dx * distance);
        float sample_y = cy + (rotated_dy * distance);

        int ix = (int)(sample_x + 0.5f);
        int iy = (int)(sample_y + 0.5f);
        ix = fmax(0, fmin(data->input_world->width - 1, ix));
        iy = fmax(0, fmin(data->input_world->height - 1, iy));

        if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
            PF_PixelFloat* input_pixels = (PF_PixelFloat*)data->input_world->data;
            PF_PixelFloat sample = input_pixels[iy * data->input_world->rowbytes / sizeof(PF_PixelFloat) + ix];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
            PF_Pixel16* input_pixels = (PF_Pixel16*)data->input_world->data;
            PF_Pixel16 sample = input_pixels[iy * data->input_world->rowbytes / sizeof(PF_Pixel16) + ix];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else {  
            PF_Pixel8* input_pixels = (PF_Pixel8*)data->input_world->data;
            PF_Pixel8 sample = input_pixels[iy * data->input_world->rowbytes / sizeof(PF_Pixel8) + ix];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
    }

    float inv_nSamples = 1.0f / (float)nSamples;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        outP->red = accumR * inv_nSamples;
        outP->green = accumG * inv_nSamples;
        outP->blue = accumB * inv_nSamples;
        outP->alpha = accumA * inv_nSamples;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        outP->red = static_cast<A_u_short>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_short>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_short>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_short>(accumA * inv_nSamples + 0.5f);
    }
    else {  
        outP->red = static_cast<A_u_char>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_char>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_char>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_char>(accumA * inv_nSamples + 0.5f);
    }

    return PF_Err_NONE;
}


static PF_Err AngleBlurFunc8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return ApplyAngleBlurGeneric<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err AngleBlurFunc16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return ApplyAngleBlurGeneric<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err AngleBlurFuncFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return ApplyAngleBlurGeneric<PF_PixelFloat>(refcon, xL, yL, inP, outP);
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
    cl_kernel motion_blur_kernel;
    cl_kernel scale_blur_kernel;
    cl_kernel angle_blur_kernel;
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
    ShaderObjectPtr mMotionBlurShader;
    ShaderObjectPtr mScaleBlurShader;
    ShaderObjectPtr mAngleBlurShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
    id<MTLComputePipelineState> motion_blur_pipeline;
    id<MTLComputePipelineState> scale_blur_pipeline;
    id<MTLComputePipelineState> angle_blur_pipeline;
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

        size_t sizes[] = { strlen(k16fString), strlen(kMotionBlurKernel_OpenCLString) };
        char const* strings[] = { k16fString, kMotionBlurKernel_OpenCLString };
        cl_context context = (cl_context)device_info.contextPV;
        cl_device_id device = (cl_device_id)device_info.devicePV;

        cl_program program;
        if (!err) {
            program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
            CL_ERR(result);
        }

        CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

        if (!err) {
            cl_gpu_data->motion_blur_kernel = clCreateKernel(program, "MotionBlurKernel", &result);
            CL_ERR(result);
        }

        if (!err) {
            cl_gpu_data->scale_blur_kernel = clCreateKernel(program, "ScaleBlurKernel", &result);
            CL_ERR(result);
        }

        if (!err) {
            cl_gpu_data->angle_blur_kernel = clCreateKernel(program, "AngleBlurKernel", &result);
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
        dx_gpu_data->mMotionBlurShader = std::make_shared<ShaderObject>();
        dx_gpu_data->mScaleBlurShader = std::make_shared<ShaderObject>();
        dx_gpu_data->mAngleBlurShader = std::make_shared<ShaderObject>();

        DX_ERR(dx_gpu_data->mContext->Initialize(
            (ID3D12Device*)device_info.devicePV,
            (ID3D12CommandQueue*)device_info.command_queuePV));

        std::wstring csoPath, sigPath;
        DX_ERR(GetShaderPath(L"MotionBlurKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mMotionBlurShader));

        DX_ERR(GetShaderPath(L"ScaleBlurKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mScaleBlurShader));

        DX_ERR(GetShaderPath(L"AngleBlurKernel", csoPath, sigPath));

        DX_ERR(dx_gpu_data->mContext->LoadShader(
            csoPath.c_str(),
            sigPath.c_str(),
            dx_gpu_data->mAngleBlurShader));

        extraP->output->gpu_data = gpu_dataH;
        out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        NSString* source = [NSString stringWithCString : kMotionBlur_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
            id<MTLFunction> motion_blur_function = nil;
            id<MTLFunction> scale_blur_function = nil;
            id<MTLFunction> angle_blur_function = nil;
            NSString* motion_blur_name = [NSString stringWithCString : "MotionBlurKernel" encoding : NSUTF8StringEncoding];
            NSString* scale_blur_name = [NSString stringWithCString : "ScaleBlurKernel" encoding : NSUTF8StringEncoding];
            NSString* angle_blur_name = [NSString stringWithCString : "AngleBlurKernel" encoding : NSUTF8StringEncoding];

            motion_blur_function = [[library newFunctionWithName : motion_blur_name]autorelease];
            scale_blur_function = [[library newFunctionWithName : scale_blur_name]autorelease];
            angle_blur_function = [[library newFunctionWithName : angle_blur_name]autorelease];

            if (!motion_blur_function || !scale_blur_function || !angle_blur_function) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            if (!err) {
                metal_data->motion_blur_pipeline = [device newComputePipelineStateWithFunction : motion_blur_function error : &error];
                err = NSError2PFErr(error);
            }

            if (!err) {
                metal_data->scale_blur_pipeline = [device newComputePipelineStateWithFunction : scale_blur_function error : &error];
                err = NSError2PFErr(error);
            }

            if (!err) {
                metal_data->angle_blur_pipeline = [device newComputePipelineStateWithFunction : angle_blur_function error : &error];
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

        (void)clReleaseKernel(cl_gpu_dataP->motion_blur_kernel);
        (void)clReleaseKernel(cl_gpu_dataP->scale_blur_kernel);
        (void)clReleaseKernel(cl_gpu_dataP->angle_blur_kernel);

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
        dx_gpu_dataP->mMotionBlurShader.reset();
        dx_gpu_dataP->mScaleBlurShader.reset();
        dx_gpu_dataP->mAngleBlurShader.reset();

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

        [metal_dataP->motion_blur_pipeline release] ;
        [metal_dataP->scale_blur_pipeline release] ;
        [metal_dataP->angle_blur_pipeline release] ;

        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
            kPFHandleSuite,
            kPFHandleSuiteVersion1,
            out_dataP);

        handle_suite->host_dispose_handle(gpu_dataH);
    }
#endif

    return err;
}

typedef struct {
    bool has_motion_prev_curr;
    bool has_scale_change_prev_curr;
    bool has_rotation_prev_curr;
    bool has_motion_curr_next;
    bool has_scale_change_curr_next;
    bool has_rotation_curr_next;
    double motion_x_prev_curr;
    double motion_y_prev_curr;
    double motion_x_curr_next;
    double motion_y_curr_next;
    double scale_x_prev_curr;
    double scale_y_prev_curr;
    double scale_x_curr_next;
    double scale_y_curr_next;
    double rotation_prev_curr;
    double rotation_curr_next;
    bool position_enabled;
    bool scale_enabled;
    bool angle_enabled;
    double tune_value;
    float scale_velocity;
    AEGP_TwoDVal anchor_point;
    double effect_rotation;
    bool has_effect_rotation;
} MotionDetectionData;

static void
DisposePreRenderData(
    void* pre_render_dataPV)
{
    if (pre_render_dataPV) {
        MotionDetectionData* infoP = reinterpret_cast<MotionDetectionData*>(pre_render_dataPV);
        free(infoP);
    }
}

static PF_Err
SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    PF_ParamDef position_param, scale_param, angle_param, tune_param;
    AEFX_CLR_STRUCT(position_param);
    AEFX_CLR_STRUCT(scale_param);
    AEFX_CLR_STRUCT(angle_param);
    AEFX_CLR_STRUCT(tune_param);

    ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_POSITION, in_data->current_time, in_data->time_step, in_data->time_scale, &position_param));
    ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_SCALE, in_data->current_time, in_data->time_step, in_data->time_scale, &scale_param));
    ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &angle_param));
    ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_TUNE, in_data->current_time, in_data->time_step, in_data->time_scale, &tune_param));

    PF_RenderRequest req = extra->input->output_request;

    PF_CheckoutResult checkout_result;
    ERR(extra->cb->checkout_layer(in_data->effect_ref,
        MOTIONBLUR_INPUT,
        MOTIONBLUR_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &checkout_result));

    if (!err) {
        extra->output->result_rect = checkout_result.result_rect;
        extra->output->max_result_rect = checkout_result.result_rect;

        extra->output->solid = FALSE;
        extra->output->pre_render_data = NULL;

        double motion_x_prev_curr = 0, motion_y_prev_curr = 0;
        double motion_x_curr_next = 0, motion_y_curr_next = 0;
        double scale_x_prev_curr = 0, scale_y_prev_curr = 0;
        double scale_x_curr_next = 0, scale_y_curr_next = 0;
        double rotation_prev_curr = 0, rotation_curr_next = 0;
        bool has_motion_prev_curr = false, has_motion_curr_next = false;
        bool has_scale_change_prev_curr = false, has_scale_change_curr_next = false;
        bool has_rotation_prev_curr = false, has_rotation_curr_next = false;
        float scale_velocity = 0.0f;
        AEGP_TwoDVal anchor_point;

        AEGP_LayerIDVal layer_id = 0;
        AEGP_SuiteHandler suites(in_data->pica_basicP);

        PF_FpLong layer_time_offset = 0;
        A_Ratio stretch_factor = { 1, 1 };      

        if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
            AEGP_LayerH layer = NULL;

            err = suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_data->effect_ref, &layer);

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

        PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;

        PF_FpLong stretch_ratio = (PF_FpLong)stretch_factor.num / (PF_FpLong)stretch_factor.den;

        current_time -= layer_time_offset;

        current_time *= stretch_ratio;

        DetectChanges(in_data,
            &motion_x_prev_curr, &motion_y_prev_curr,
            &motion_x_curr_next, &motion_y_curr_next,
            &scale_x_prev_curr, &scale_y_prev_curr,
            &scale_x_curr_next, &scale_y_curr_next,
            &rotation_prev_curr, &rotation_curr_next,
            &has_motion_prev_curr, &has_motion_curr_next,
            &has_scale_change_prev_curr, &has_scale_change_curr_next,
            &has_rotation_prev_curr, &has_rotation_curr_next,
            &scale_velocity, &anchor_point);

        double effect_motion_x = 0, effect_motion_y = 0;
        double effect_rotation = 0;
        double effect_scale_x = 0, effect_scale_y = 0;
        float effect_scale_velocity = 0.0f;

        bool has_effect_motion = DetectMotionFromOtherEffects(
            in_data,
            &effect_motion_x, &effect_motion_y,
            &effect_rotation,
            &effect_scale_x, &effect_scale_y,
            &effect_scale_velocity);

        if (has_effect_motion) {
            if (position_param.u.bd.value) {
                motion_x_curr_next += effect_motion_x;
                motion_y_curr_next += effect_motion_y;

                if (!has_motion_curr_next) {
                    double motion_magnitude = sqrt(motion_x_curr_next * motion_x_curr_next +
                        motion_y_curr_next * motion_y_curr_next);
                    has_motion_curr_next = (motion_magnitude > 0.5);
                }
            }

            if (angle_param.u.bd.value) {
                rotation_curr_next += effect_rotation;

                if (!has_rotation_curr_next) {
                    has_rotation_curr_next = (fabs(rotation_curr_next) > 0.1);
                }
            }

            if (scale_param.u.bd.value) {
                scale_x_curr_next += effect_scale_x;
                scale_y_curr_next += effect_scale_y;

                if (fabs(effect_scale_velocity) > 0.01f) {
                    scale_velocity = effect_scale_velocity;
                }

                if (!has_scale_change_curr_next) {
                    has_scale_change_curr_next = (fabs(scale_x_curr_next) > 0.1 ||
                        fabs(scale_y_curr_next) > 0.1 ||
                        fabs(scale_velocity) > 0.01f);
                }
            }
        }

        MotionDetectionData* infoP = reinterpret_cast<MotionDetectionData*>(malloc(sizeof(MotionDetectionData)));
        if (infoP) {
            infoP->has_motion_prev_curr = has_motion_prev_curr;
            infoP->has_scale_change_prev_curr = has_scale_change_prev_curr;
            infoP->has_rotation_prev_curr = has_rotation_prev_curr;
            infoP->has_motion_curr_next = has_motion_curr_next;
            infoP->has_scale_change_curr_next = has_scale_change_curr_next;
            infoP->has_rotation_curr_next = has_rotation_curr_next;
            infoP->motion_x_prev_curr = motion_x_prev_curr;
            infoP->motion_y_prev_curr = motion_y_prev_curr;
            infoP->motion_x_curr_next = motion_x_curr_next;
            infoP->motion_y_curr_next = motion_y_curr_next;
            infoP->scale_x_prev_curr = scale_x_prev_curr;
            infoP->scale_y_prev_curr = scale_y_prev_curr;
            infoP->scale_x_curr_next = scale_x_curr_next;
            infoP->scale_y_curr_next = scale_y_curr_next;
            infoP->rotation_prev_curr = rotation_prev_curr;
            infoP->rotation_curr_next = rotation_curr_next;
            infoP->position_enabled = position_param.u.bd.value;
            infoP->scale_enabled = scale_param.u.bd.value;
            infoP->angle_enabled = angle_param.u.bd.value;
            infoP->tune_value = tune_param.u.fs_d.value;
            infoP->scale_velocity = scale_velocity;
            infoP->anchor_point = anchor_point;
            infoP->effect_rotation = effect_rotation;
            infoP->has_effect_rotation = (fabs(effect_rotation) > 0.01);

            extra->output->pre_render_data = infoP;
            extra->output->delete_pre_render_data_func = DisposePreRenderData;

            struct {
                A_u_char has_motion_prev_curr;
                A_u_char has_scale_change_prev_curr;
                A_u_char has_rotation_prev_curr;
                A_u_char has_motion_curr_next;
                A_u_char has_scale_change_curr_next;
                A_u_char has_rotation_curr_next;
                A_u_char position_enabled;
                A_u_char scale_enabled;
                A_u_char angle_enabled;
                float motion_x_prev_curr;
                float motion_y_prev_curr;
                float motion_x_curr_next;
                float motion_y_curr_next;
                float scale_velocity;
                float anchor_x;
                float anchor_y;
                float tune_value;
                float effect_motion_x;
                float effect_motion_y;
                float effect_rotation;
                float effect_scale_velocity;
                A_u_char has_effect_motion;
                AEGP_LayerIDVal layer_id;
                PF_FpLong current_time;
                PF_FpLong layer_time_offset;
                A_Ratio stretch_factor;       
            } detection_data;

            detection_data.has_motion_prev_curr = has_motion_prev_curr ? 1 : 0;
            detection_data.has_scale_change_prev_curr = has_scale_change_prev_curr ? 1 : 0;
            detection_data.has_rotation_prev_curr = has_rotation_prev_curr ? 1 : 0;
            detection_data.has_motion_curr_next = has_motion_curr_next ? 1 : 0;
            detection_data.has_scale_change_curr_next = has_scale_change_curr_next ? 1 : 0;
            detection_data.has_rotation_curr_next = has_rotation_curr_next ? 1 : 0;
            detection_data.position_enabled = position_param.u.bd.value ? 1 : 0;
            detection_data.scale_enabled = scale_param.u.bd.value ? 1 : 0;
            detection_data.angle_enabled = angle_param.u.bd.value ? 1 : 0;
            detection_data.motion_x_prev_curr = motion_x_prev_curr;
            detection_data.motion_y_prev_curr = motion_y_prev_curr;
            detection_data.motion_x_curr_next = motion_x_curr_next;
            detection_data.motion_y_curr_next = motion_y_curr_next;
            detection_data.scale_velocity = scale_velocity;
            detection_data.anchor_x = anchor_point.x;
            detection_data.anchor_y = anchor_point.y;
            detection_data.tune_value = tune_param.u.fs_d.value;
            detection_data.effect_motion_x = effect_motion_x;
            detection_data.effect_motion_y = effect_motion_y;
            detection_data.effect_rotation = effect_rotation;
            detection_data.effect_scale_velocity = effect_scale_velocity;
            detection_data.has_effect_motion = has_effect_motion ? 1 : 0;
            detection_data.layer_id = layer_id;
            detection_data.current_time = current_time;
            detection_data.layer_time_offset = layer_time_offset;
            detection_data.stretch_factor = stretch_factor;

            ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(detection_data), &detection_data));

            extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
        }
        else {
            err = PF_Err_OUT_OF_MEMORY;
        }
    }

    ERR(PF_CHECKIN_PARAM(in_data, &position_param));
    ERR(PF_CHECKIN_PARAM(in_data, &scale_param));
    ERR(PF_CHECKIN_PARAM(in_data, &angle_param));
    ERR(PF_CHECKIN_PARAM(in_data, &tune_param));

    return err;
}

size_t RoundUp(
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

typedef struct {
    int mSrcPitch;
    int mDstPitch;
    int m16f;
    int mWidth;
    int mHeight;
    float mMotionX;
    float mMotionY;
    float mTuneValue;
    float mDownsampleX;
    float mDownsampleY;
} MotionBlurParams;

typedef struct {
    int mSrcPitch;
    int mDstPitch;
    int m16f;
    int mWidth;
    int mHeight;
    float mScaleVelocity;
    float mAnchorX;
    float mAnchorY;
    float mTuneValue;
    float mDownsampleX;
    float mDownsampleY;
} ScaleBlurParams;

typedef struct {
    int mSrcPitch;
    int mDstPitch;
    int m16f;
    int mWidth;
    int mHeight;
    float mRotationAngle;
    float mAnchorX;
    float mAnchorY;
    float mTuneValue;
    float mDownsampleX;
    float mDownsampleY;
} AngleBlurParams;

static PF_Err
SmartRenderGPU(
    PF_InData* in_dataP,
    PF_OutData* out_dataP,
    PF_PixelFormat pixel_format,
    PF_EffectWorld* input_worldP,
    PF_EffectWorld* output_worldP,
    PF_SmartRenderExtra* extraP,
    MotionDetectionData* infoP)
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

    PF_EffectWorld* intermediate_buffer1;
    PF_EffectWorld* intermediate_buffer2;

    ERR(gpu_suite->CreateGPUWorld(in_dataP->effect_ref,
        extraP->input->device_index,
        input_worldP->width,
        input_worldP->height,
        input_worldP->pix_aspect_ratio,
        in_dataP->field,
        pixel_format,
        false,
        &intermediate_buffer1));

    ERR(gpu_suite->CreateGPUWorld(in_dataP->effect_ref,
        extraP->input->device_index,
        input_worldP->width,
        input_worldP->height,
        input_worldP->pix_aspect_ratio,
        in_dataP->field,
        pixel_format,
        false,
        &intermediate_buffer2));

    void* src_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

    void* dst_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

    void* im1_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, intermediate_buffer1, &im1_mem));

    void* im2_mem = 0;
    ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, intermediate_buffer2, &im2_mem));

    float downsampleX = 1.0f;
    float downsampleY = 1.0f;

    if (in_dataP) {
        downsampleX = (float)in_dataP->downsample_x.num / (float)in_dataP->downsample_x.den;
        downsampleY = (float)in_dataP->downsample_y.num / (float)in_dataP->downsample_y.den;
    }

    MotionBlurParams motion_params;
    ScaleBlurParams scale_params;
    AngleBlurParams angle_params;

    int width = input_worldP->width;
    int height = input_worldP->height;
    int is16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
    A_long src_row_bytes = input_worldP->rowbytes;
    A_long dst_row_bytes = output_worldP->rowbytes;
    A_long im1_row_bytes = intermediate_buffer1->rowbytes;
    A_long im2_row_bytes = intermediate_buffer2->rowbytes;

    motion_params.mWidth = width;
    motion_params.mHeight = height;
    motion_params.m16f = is16f;
    motion_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
    motion_params.mDstPitch = im1_row_bytes / bytes_per_pixel;
    motion_params.mMotionX = 0.0f;
    motion_params.mMotionY = 0.0f;
    motion_params.mTuneValue = infoP->tune_value;
    motion_params.mDownsampleX = downsampleX;
    motion_params.mDownsampleY = downsampleY;

    if (infoP->has_motion_curr_next) {
        motion_params.mMotionX = infoP->motion_x_curr_next;
        motion_params.mMotionY = infoP->motion_y_curr_next;
    }
    else if (infoP->has_motion_prev_curr) {
        motion_params.mMotionX = infoP->motion_x_prev_curr;
        motion_params.mMotionY = infoP->motion_y_prev_curr;
    }

    scale_params.mWidth = width;
    scale_params.mHeight = height;
    scale_params.m16f = is16f;
    scale_params.mSrcPitch = im1_row_bytes / bytes_per_pixel;
    scale_params.mDstPitch = im2_row_bytes / bytes_per_pixel;
    scale_params.mScaleVelocity = infoP->scale_velocity;
    scale_params.mAnchorX = infoP->anchor_point.x;
    scale_params.mAnchorY = infoP->anchor_point.y;
    scale_params.mTuneValue = infoP->tune_value;
    scale_params.mDownsampleX = downsampleX;
    scale_params.mDownsampleY = downsampleY;

    angle_params.mWidth = width;
    angle_params.mHeight = height;
    angle_params.m16f = is16f;
    angle_params.mSrcPitch = im2_row_bytes / bytes_per_pixel;
    angle_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
    angle_params.mRotationAngle = 0.0f;
    angle_params.mAnchorX = infoP->anchor_point.x;
    angle_params.mAnchorY = infoP->anchor_point.y;
    angle_params.mTuneValue = infoP->tune_value;
    angle_params.mDownsampleX = downsampleX;
    angle_params.mDownsampleY = downsampleY;

    if (infoP->has_effect_rotation) {
        angle_params.mRotationAngle = infoP->effect_rotation;
    }
    else if (infoP->has_rotation_curr_next) {
        angle_params.mRotationAngle = infoP->rotation_curr_next * (float)M_PI / 180.0f;
    }
    else if (infoP->has_rotation_prev_curr) {
        angle_params.mRotationAngle = infoP->rotation_prev_curr * (float)M_PI / 180.0f;
    }

    void* current_src = src_mem;
    void* current_dst = im1_mem;
    void* temp_mem = 0;

    if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
    {
        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

        cl_mem cl_src_mem = (cl_mem)src_mem;
        cl_mem cl_im1_mem = (cl_mem)im1_mem;
        cl_mem cl_im2_mem = (cl_mem)im2_mem;
        cl_mem cl_dst_mem = (cl_mem)dst_mem;

        size_t threadBlock[2] = { 16, 16 };
        size_t grid[2] = { RoundUp(width, threadBlock[0]), RoundUp(height, threadBlock[1]) };

        if (infoP->position_enabled && (infoP->has_motion_prev_curr || infoP->has_motion_curr_next)) {
            cl_uint param_index = 0;
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(cl_mem), &cl_im1_mem));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(int), &motion_params.mSrcPitch));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(int), &motion_params.mDstPitch));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(int), &motion_params.m16f));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(unsigned int), &motion_params.mWidth));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(unsigned int), &motion_params.mHeight));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(float), &motion_params.mMotionX));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(float), &motion_params.mMotionY));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(float), &motion_params.mTuneValue));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(float), &motion_params.mDownsampleX));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->motion_blur_kernel, param_index++, sizeof(float), &motion_params.mDownsampleY));

            CL_ERR(clEnqueueNDRangeKernel(
                (cl_command_queue)device_info.command_queuePV,
                cl_gpu_dataP->motion_blur_kernel,
                2,
                0,
                grid,
                threadBlock,
                0,
                0,
                0));

            current_src = im1_mem;
            current_dst = im2_mem;
        }
        else {
            CL_ERR(clEnqueueCopyBuffer(
                (cl_command_queue)device_info.command_queuePV,
                cl_src_mem,
                cl_im1_mem,
                0,
                0,
                height * src_row_bytes,
                0,
                NULL,
                NULL));

            current_src = im1_mem;
            current_dst = im2_mem;
        }

        if (infoP->scale_enabled && fabs(infoP->scale_velocity) > 0.01f) {
            cl_uint param_index = 0;
            cl_mem cl_curr_src = (cl_mem)current_src;
            cl_mem cl_curr_dst = (cl_mem)current_dst;

            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(cl_mem), &cl_curr_src));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(cl_mem), &cl_curr_dst));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(int), &scale_params.mSrcPitch));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(int), &scale_params.mDstPitch));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(int), &scale_params.m16f));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(unsigned int), &scale_params.mWidth));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(unsigned int), &scale_params.mHeight));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(float), &scale_params.mScaleVelocity));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(float), &scale_params.mAnchorX));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(float), &scale_params.mAnchorY));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(float), &scale_params.mTuneValue));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(float), &scale_params.mDownsampleX));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->scale_blur_kernel, param_index++, sizeof(float), &scale_params.mDownsampleY));

            CL_ERR(clEnqueueNDRangeKernel(
                (cl_command_queue)device_info.command_queuePV,
                cl_gpu_dataP->scale_blur_kernel,
                2,
                0,
                grid,
                threadBlock,
                0,
                0,
                0));

            temp_mem = current_src;
            current_src = current_dst;
            current_dst = temp_mem;
        }

        if (infoP->angle_enabled && (infoP->has_rotation_prev_curr || infoP->has_rotation_curr_next || infoP->has_effect_rotation)) {
            cl_uint param_index = 0;
            cl_mem cl_curr_src = (cl_mem)current_src;

            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(cl_mem), &cl_curr_src));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(int), &angle_params.mSrcPitch));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(int), &angle_params.mDstPitch));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(int), &angle_params.m16f));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(unsigned int), &angle_params.mWidth));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(unsigned int), &angle_params.mHeight));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(float), &angle_params.mRotationAngle));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(float), &angle_params.mAnchorX));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(float), &angle_params.mAnchorY));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(float), &angle_params.mTuneValue));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(float), &angle_params.mDownsampleX));
            CL_ERR(clSetKernelArg(cl_gpu_dataP->angle_blur_kernel, param_index++, sizeof(float), &angle_params.mDownsampleY));

            CL_ERR(clEnqueueNDRangeKernel(
                (cl_command_queue)device_info.command_queuePV,
                cl_gpu_dataP->angle_blur_kernel,
                2,
                0,
                grid,
                threadBlock,
                0,
                0,
                0));
        }
        else {
            cl_mem cl_curr_src = (cl_mem)current_src;
            CL_ERR(clEnqueueCopyBuffer(
                (cl_command_queue)device_info.command_queuePV,
                cl_curr_src,
                cl_dst_mem,
                0,
                0,
                height * dst_row_bytes,
                0,
                NULL,
                NULL));
        }
    }
#if HAS_CUDA
    else if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
        if (infoP->position_enabled && (infoP->has_motion_prev_curr || infoP->has_motion_curr_next)) {
            Motion_Blur_CUDA(
                (const float*)src_mem,
                (float*)im1_mem,
                motion_params.mSrcPitch,
                motion_params.mDstPitch,
                motion_params.m16f,
                motion_params.mWidth,
                motion_params.mHeight,
                motion_params.mMotionX,
                motion_params.mMotionY,
                motion_params.mTuneValue,
                motion_params.mDownsampleX,
                motion_params.mDownsampleY);

            if (cudaPeekAtLastError() != cudaSuccess) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            current_src = im1_mem;
            current_dst = im2_mem;
        }
        else {
            cudaMemcpy(im1_mem, src_mem, height * src_row_bytes, cudaMemcpyDeviceToDevice);

            current_src = im1_mem;
            current_dst = im2_mem;
        }

        if (!err && infoP->scale_enabled && fabs(infoP->scale_velocity) > 0.01f) {
            Scale_Blur_CUDA(
                (const float*)current_src,
                (float*)current_dst,
                scale_params.mSrcPitch,
                scale_params.mDstPitch,
                scale_params.m16f,
                scale_params.mWidth,
                scale_params.mHeight,
                scale_params.mScaleVelocity,
                scale_params.mAnchorX,
                scale_params.mAnchorY,
                scale_params.mTuneValue,
                scale_params.mDownsampleX,
                scale_params.mDownsampleY);

            if (cudaPeekAtLastError() != cudaSuccess) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }

            temp_mem = current_src;
            current_src = current_dst;
            current_dst = temp_mem;
        }

        if (!err && infoP->angle_enabled && (infoP->has_rotation_prev_curr || infoP->has_rotation_curr_next || infoP->has_effect_rotation)) {
            Angle_Blur_CUDA(
                (const float*)current_src,
                (float*)dst_mem,
                angle_params.mSrcPitch,
                angle_params.mDstPitch,
                angle_params.m16f,
                angle_params.mWidth,
                angle_params.mHeight,
                angle_params.mRotationAngle,
                angle_params.mAnchorX,
                angle_params.mAnchorY,
                angle_params.mTuneValue,
                angle_params.mDownsampleX,
                angle_params.mDownsampleY);

            if (cudaPeekAtLastError() != cudaSuccess) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
        }
        else if (!err) {
            cudaMemcpy(dst_mem, current_src, height * dst_row_bytes, cudaMemcpyDeviceToDevice);
        }
    }
#endif
#if HAS_HLSL
    else if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
    {
        typedef struct {
            int mSrcPitch;
            int mDstPitch;
            int m16f;
            int mWidth;
            int mHeight;
            float mMotionX;
            float mMotionY;
            float mTuneValue;
            float mDownsampleX;
            float mDownsampleY;
            float _padding[2];    
        } MotionBlurParamsDX;

        typedef struct {
            int mSrcPitch;
            int mDstPitch;
            int m16f;
            int mWidth;
            int mHeight;
            float mScaleVelocity;
            float mAnchorX;
            float mAnchorY;
            float mTuneValue;
            float mDownsampleX;
            float mDownsampleY;
            float _padding[1];    
        } ScaleBlurParamsDX;

        typedef struct {
            int mSrcPitch;
            int mDstPitch;
            int m16f;
            int mWidth;
            int mHeight;
            float mRotationAngle;
            float mAnchorX;
            float mAnchorY;
            float mTuneValue;
            float mDownsampleX;
            float mDownsampleY;
            float _padding[1];    
        } AngleBlurParamsDX;

        MotionBlurParamsDX motion_params_dx;
        motion_params_dx.mSrcPitch = motion_params.mSrcPitch;
        motion_params_dx.mDstPitch = motion_params.mDstPitch;
        motion_params_dx.m16f = motion_params.m16f;
        motion_params_dx.mWidth = motion_params.mWidth;
        motion_params_dx.mHeight = motion_params.mHeight;
        motion_params_dx.mMotionX = motion_params.mMotionX;
        motion_params_dx.mMotionY = motion_params.mMotionY;
        motion_params_dx.mTuneValue = motion_params.mTuneValue;
        motion_params_dx.mDownsampleX = motion_params.mDownsampleX;
        motion_params_dx.mDownsampleY = motion_params.mDownsampleY;

        ScaleBlurParamsDX scale_params_dx;
        scale_params_dx.mSrcPitch = scale_params.mSrcPitch;
        scale_params_dx.mDstPitch = scale_params.mDstPitch;
        scale_params_dx.m16f = scale_params.m16f;
        scale_params_dx.mWidth = scale_params.mWidth;
        scale_params_dx.mHeight = scale_params.mHeight;
        scale_params_dx.mScaleVelocity = scale_params.mScaleVelocity;
        scale_params_dx.mAnchorX = scale_params.mAnchorX;
        scale_params_dx.mAnchorY = scale_params.mAnchorY;
        scale_params_dx.mTuneValue = scale_params.mTuneValue;
        scale_params_dx.mDownsampleX = scale_params.mDownsampleX;
        scale_params_dx.mDownsampleY = scale_params.mDownsampleY;

        AngleBlurParamsDX angle_params_dx;
        angle_params_dx.mSrcPitch = angle_params.mSrcPitch;
        angle_params_dx.mDstPitch = angle_params.mDstPitch;
        angle_params_dx.m16f = angle_params.m16f;
        angle_params_dx.mWidth = angle_params.mWidth;
        angle_params_dx.mHeight = angle_params.mHeight;
        angle_params_dx.mRotationAngle = angle_params.mRotationAngle;
        angle_params_dx.mAnchorX = angle_params.mAnchorX;
        angle_params_dx.mAnchorY = angle_params.mAnchorY;
        angle_params_dx.mTuneValue = angle_params.mTuneValue;
        angle_params_dx.mDownsampleX = angle_params.mDownsampleX;
        angle_params_dx.mDownsampleY = angle_params.mDownsampleY;

        PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
        DirectXGPUData* dx_gpu_data = reinterpret_cast<DirectXGPUData*>(*gpu_dataH);

        if (infoP->position_enabled && (infoP->has_motion_prev_curr || infoP->has_motion_curr_next)) {
            DXShaderExecution shaderExecution(
                dx_gpu_data->mContext,
                dx_gpu_data->mMotionBlurShader,
                3);

            DX_ERR(shaderExecution.SetParamBuffer(&motion_params_dx, sizeof(MotionBlurParamsDX)));
            DX_ERR(shaderExecution.SetUnorderedAccessView(
                (ID3D12Resource*)im1_mem,
                motion_params.mHeight * im1_row_bytes));
            DX_ERR(shaderExecution.SetShaderResourceView(
                (ID3D12Resource*)src_mem,
                motion_params.mHeight * src_row_bytes));
            DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(motion_params.mWidth, 16), (UINT)DivideRoundUp(motion_params.mHeight, 16)));

            current_src = im1_mem;
            current_dst = im2_mem;
        }
        else {
            dx_gpu_data->mContext->mCommandList->CopyResource(
                (ID3D12Resource*)im1_mem,
                (ID3D12Resource*)src_mem);
            dx_gpu_data->mContext->CloseWaitAndReset();

            current_src = im1_mem;
            current_dst = im2_mem;
        }

        if (!err && infoP->scale_enabled && fabs(infoP->scale_velocity) > 0.01f) {
            DXShaderExecution shaderExecution(
                dx_gpu_data->mContext,
                dx_gpu_data->mScaleBlurShader,
                3);

            DX_ERR(shaderExecution.SetParamBuffer(&scale_params_dx, sizeof(ScaleBlurParamsDX)));
            DX_ERR(shaderExecution.SetUnorderedAccessView(
                (ID3D12Resource*)current_dst,
                scale_params.mHeight * im2_row_bytes));
            DX_ERR(shaderExecution.SetShaderResourceView(
                (ID3D12Resource*)current_src,
                scale_params.mHeight * im1_row_bytes));
            DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(scale_params.mWidth, 16), (UINT)DivideRoundUp(scale_params.mHeight, 16)));

            temp_mem = current_src;
            current_src = current_dst;
            current_dst = temp_mem;
        }

        if (!err && infoP->angle_enabled && (infoP->has_rotation_prev_curr || infoP->has_rotation_curr_next || infoP->has_effect_rotation)) {
            DXShaderExecution shaderExecution(
                dx_gpu_data->mContext,
                dx_gpu_data->mAngleBlurShader,
                3);

            DX_ERR(shaderExecution.SetParamBuffer(&angle_params_dx, sizeof(AngleBlurParamsDX)));
            DX_ERR(shaderExecution.SetUnorderedAccessView(
                (ID3D12Resource*)dst_mem,
                angle_params.mHeight * dst_row_bytes));
            DX_ERR(shaderExecution.SetShaderResourceView(
                (ID3D12Resource*)current_src,
                angle_params.mHeight * (current_src == im1_mem ? im1_row_bytes : im2_row_bytes)));
            DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(angle_params.mWidth, 16), (UINT)DivideRoundUp(angle_params.mHeight, 16)));
        }
        else if (!err) {
            dx_gpu_data->mContext->mCommandList->CopyResource(
                (ID3D12Resource*)dst_mem,
                (ID3D12Resource*)current_src);
            dx_gpu_data->mContext->CloseWaitAndReset();
        }
    }
#endif
#if HAS_METAL
    else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
    {
        ScopedAutoreleasePool pool;

        PF_Handle metal_handle = (PF_Handle)extraP->input->gpu_data;
        MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

        typedef struct {
            int mSrcPitch;
            int mDstPitch;
            int m16f;
            int mWidth;
            int mHeight;
            float mMotionX;
            float mMotionY;
            float mTuneValue;
            float mDownsampleX;
            float mDownsampleY;
        } MotionBlurParamsMetal;

        typedef struct {
            int mSrcPitch;
            int mDstPitch;
            int m16f;
            int mWidth;
            int mHeight;
            float mScaleVelocity;
            float mAnchorX;
            float mAnchorY;
            float mTuneValue;
            float mDownsampleX;
            float mDownsampleY;
        } ScaleBlurParamsMetal;

        typedef struct {
            int mSrcPitch;
            int mDstPitch;
            int m16f;
            int mWidth;
            int mHeight;
            float mRotationAngle;
            float mAnchorX;
            float mAnchorY;
            float mTuneValue;
            float mDownsampleX;
            float mDownsampleY;
        } AngleBlurParamsMetal;

        MotionBlurParamsMetal motion_params_metal;
        motion_params_metal.mSrcPitch = motion_params.mSrcPitch;
        motion_params_metal.mDstPitch = motion_params.mDstPitch;
        motion_params_metal.m16f = motion_params.m16f;
        motion_params_metal.mWidth = motion_params.mWidth;
        motion_params_metal.mHeight = motion_params.mHeight;
        motion_params_metal.mMotionX = motion_params.mMotionX;
        motion_params_metal.mMotionY = motion_params.mMotionY;
        motion_params_metal.mTuneValue = motion_params.mTuneValue;
        motion_params_metal.mDownsampleX = motion_params.mDownsampleX;
        motion_params_metal.mDownsampleY = motion_params.mDownsampleY;

        ScaleBlurParamsMetal scale_params_metal;
        scale_params_metal.mSrcPitch = scale_params.mSrcPitch;
        scale_params_metal.mDstPitch = scale_params.mDstPitch;
        scale_params_metal.m16f = scale_params.m16f;
        scale_params_metal.mWidth = scale_params.mWidth;
        scale_params_metal.mHeight = scale_params.mHeight;
        scale_params_metal.mScaleVelocity = scale_params.mScaleVelocity;
        scale_params_metal.mAnchorX = scale_params.mAnchorX;
        scale_params_metal.mAnchorY = scale_params.mAnchorY;
        scale_params_metal.mTuneValue = scale_params.mTuneValue;
        scale_params_metal.mDownsampleX = scale_params.mDownsampleX;
        scale_params_metal.mDownsampleY = scale_params.mDownsampleY;

        AngleBlurParamsMetal angle_params_metal;
        angle_params_metal.mSrcPitch = angle_params.mSrcPitch;
        angle_params_metal.mDstPitch = angle_params.mDstPitch;
        angle_params_metal.m16f = angle_params.m16f;
        angle_params_metal.mWidth = angle_params.mWidth;
        angle_params_metal.mHeight = angle_params.mHeight;
        angle_params_metal.mRotationAngle = angle_params.mRotationAngle;
        angle_params_metal.mAnchorX = angle_params.mAnchorX;
        angle_params_metal.mAnchorY = angle_params.mAnchorY;
        angle_params_metal.mTuneValue = angle_params.mTuneValue;
        angle_params_metal.mDownsampleX = angle_params.mDownsampleX;
        angle_params_metal.mDownsampleY = angle_params.mDownsampleY;

        id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
        id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        id<MTLBuffer> motion_param_buffer = [[device newBufferWithBytes : &motion_params_metal
            length : sizeof(MotionBlurParamsMetal)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLBuffer> scale_param_buffer = [[device newBufferWithBytes : &scale_params_metal
            length : sizeof(ScaleBlurParamsMetal)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLBuffer> angle_param_buffer = [[device newBufferWithBytes : &angle_params_metal
            length : sizeof(AngleBlurParamsMetal)
            options : MTLResourceStorageModeManaged]autorelease];

        id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
        id<MTLBuffer> im1_metal_buffer = (id<MTLBuffer>)im1_mem;
        id<MTLBuffer> im2_metal_buffer = (id<MTLBuffer>)im2_mem;
        id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

        if (infoP->position_enabled && (infoP->has_motion_prev_curr || infoP->has_motion_curr_next)) {
            MTLSize threadsPerGroup = { [metal_dataP->motion_blur_pipeline threadExecutionWidth] , 16, 1 };
            MTLSize numThreadgroups = { DivideRoundUp(motion_params.mWidth, threadsPerGroup.width), DivideRoundUp(motion_params.mHeight, threadsPerGroup.height), 1 };

            [computeEncoder setComputePipelineState : metal_dataP->motion_blur_pipeline] ;
            [computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
            [computeEncoder setBuffer : im1_metal_buffer offset : 0 atIndex : 1] ;
            [computeEncoder setBuffer : motion_param_buffer offset : 0 atIndex : 2] ;
            [computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;

            current_src = im1_mem;
            current_dst = im2_mem;
        }
        else {
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            [blitEncoder copyFromBuffer : src_metal_buffer sourceOffset : 0
                toBuffer : im1_metal_buffer destinationOffset : 0
                size : height * src_row_bytes] ;
            [blitEncoder endEncoding] ;

            current_src = im1_mem;
            current_dst = im2_mem;
        }

        if (infoP->scale_enabled && fabs(infoP->scale_velocity) > 0.01f) {
            id<MTLBuffer> curr_src_buffer = (id<MTLBuffer>)current_src;
            id<MTLBuffer> curr_dst_buffer = (id<MTLBuffer>)current_dst;

            MTLSize threadsPerGroup = { [metal_dataP->scale_blur_pipeline threadExecutionWidth] , 16, 1 };
            MTLSize numThreadgroups = { DivideRoundUp(scale_params.mWidth, threadsPerGroup.width), DivideRoundUp(scale_params.mHeight, threadsPerGroup.height), 1 };

            [computeEncoder setComputePipelineState : metal_dataP->scale_blur_pipeline] ;
            [computeEncoder setBuffer : curr_src_buffer offset : 0 atIndex : 0] ;
            [computeEncoder setBuffer : curr_dst_buffer offset : 0 atIndex : 1] ;
            [computeEncoder setBuffer : scale_param_buffer offset : 0 atIndex : 2] ;
            [computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;

            temp_mem = current_src;
            current_src = current_dst;
            current_dst = temp_mem;
        }

        if (infoP->angle_enabled && (infoP->has_rotation_prev_curr || infoP->has_rotation_curr_next || infoP->has_effect_rotation)) {
            id<MTLBuffer> curr_src_buffer = (id<MTLBuffer>)current_src;

            MTLSize threadsPerGroup = { [metal_dataP->angle_blur_pipeline threadExecutionWidth] , 16, 1 };
            MTLSize numThreadgroups = { DivideRoundUp(angle_params.mWidth, threadsPerGroup.width), DivideRoundUp(angle_params.mHeight, threadsPerGroup.height), 1 };

            [computeEncoder setComputePipelineState : metal_dataP->angle_blur_pipeline] ;
            [computeEncoder setBuffer : curr_src_buffer offset : 0 atIndex : 0] ;
            [computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
            [computeEncoder setBuffer : angle_param_buffer offset : 0 atIndex : 2] ;
            [computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;
        }
        else {
            id<MTLBuffer> curr_src_buffer = (id<MTLBuffer>)current_src;
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
            [blitEncoder copyFromBuffer : curr_src_buffer sourceOffset : 0
                toBuffer : dst_metal_buffer destinationOffset : 0
                size : height * dst_row_bytes] ;
            [blitEncoder endEncoding] ;
        }

        [computeEncoder endEncoding];
        [commandBuffer commit] ;
        [commandBuffer waitUntilCompleted] ;

        err = NSError2PFErr([commandBuffer error]);
        }
#endif  

        ERR(gpu_suite->DisposeGPUWorld(in_dataP->effect_ref, intermediate_buffer1));
        ERR(gpu_suite->DisposeGPUWorld(in_dataP->effect_ref, intermediate_buffer2));
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
    MotionDetectionData* infoP)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    DetectionData data;
    data.has_motion_prev_curr = infoP->has_motion_prev_curr;
    data.has_scale_change_prev_curr = infoP->has_scale_change_prev_curr;
    data.has_rotation_prev_curr = infoP->has_rotation_prev_curr;
    data.has_motion_curr_next = infoP->has_motion_curr_next;
    data.has_scale_change_curr_next = infoP->has_scale_change_curr_next;
    data.has_rotation_curr_next = infoP->has_rotation_curr_next;
    data.motion_x_prev_curr = infoP->motion_x_prev_curr;
    data.motion_y_prev_curr = infoP->motion_y_prev_curr;
    data.motion_x_curr_next = infoP->motion_x_curr_next;
    data.motion_y_curr_next = infoP->motion_y_curr_next;
    data.scale_x_prev_curr = infoP->scale_x_prev_curr;
    data.scale_y_prev_curr = infoP->scale_y_prev_curr;
    data.scale_x_curr_next = infoP->scale_x_curr_next;
    data.scale_y_curr_next = infoP->scale_y_curr_next;
    data.rotation_prev_curr = infoP->rotation_prev_curr;
    data.rotation_curr_next = infoP->rotation_curr_next;
    data.position_enabled = infoP->position_enabled;
    data.scale_enabled = infoP->scale_enabled;
    data.angle_enabled = infoP->angle_enabled;
    data.tune_value = infoP->tune_value;
    data.input_world = input_worldP;
    data.scale_velocity = infoP->scale_velocity;
    data.anchor_point = infoP->anchor_point;
    data.effect_rotation = infoP->effect_rotation;
    data.has_effect_rotation = infoP->has_effect_rotation;
    data.in_data = in_data;

    PF_EffectWorld temp_world1, temp_world2;

    PF_WorldSuite2* world_suite = NULL;
    SPBasicSuite* basic_suite = in_data->pica_basicP;

    if (basic_suite) {
        basic_suite->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&world_suite);
    }

    if (world_suite) {
        ERR(world_suite->PF_NewWorld(
            in_data->effect_ref,
            output_worldP->width,
            output_worldP->height,
            TRUE,
            pixel_format,
            &temp_world1));

        ERR(world_suite->PF_NewWorld(
            in_data->effect_ref,
            output_worldP->width,
            output_worldP->height,
            TRUE,
            pixel_format,
            &temp_world2));

        if (err) {
            return err;
        }

        ERR(suites.WorldTransformSuite1()->copy(
            in_data->effect_ref,
            input_worldP,
            &temp_world1,
            NULL,
            NULL));

        const double bytesPerPixel = static_cast<double>(input_worldP->rowbytes) /
            static_cast<double>(input_worldP->width);

        PF_EffectWorld* current_result = &temp_world1;
        PF_EffectWorld* next_result = &temp_world2;

        if (data.position_enabled &&
            (data.has_motion_prev_curr || data.has_motion_curr_next)) {
            data.input_world = current_result;

            if (bytesPerPixel >= 16.0) {
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    RenderFuncFloat,
                    next_result));
            }
            else if (bytesPerPixel >= 8.0) {
                ERR(suites.Iterate16Suite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    RenderFunc16,
                    next_result));
            }
            else {
                ERR(suites.Iterate8Suite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    RenderFunc8,
                    next_result));
            }

            PF_EffectWorld* temp = current_result;
            current_result = next_result;
            next_result = temp;
        }

        if (data.scale_enabled && (fabs(data.scale_velocity) > 0.01f)) {
            data.input_world = current_result;

            if (bytesPerPixel >= 16.0) {
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    ScaleBlurFuncFloat,
                    next_result));
            }
            else if (bytesPerPixel >= 8.0) {
                ERR(suites.Iterate16Suite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    ScaleBlurFunc16,
                    next_result));
            }
            else {
                ERR(suites.Iterate8Suite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    ScaleBlurFunc8,
                    next_result));
            }

            PF_EffectWorld* temp = current_result;
            current_result = next_result;
            next_result = temp;
        }

        if (data.angle_enabled &&
            (data.has_rotation_prev_curr || data.has_rotation_curr_next || data.has_effect_rotation)) {
            data.input_world = current_result;

            if (bytesPerPixel >= 16.0) {
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    AngleBlurFuncFloat,
                    next_result));
            }
            else if (bytesPerPixel >= 8.0) {
                ERR(suites.Iterate16Suite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    AngleBlurFunc16,
                    next_result));
            }
            else {
                ERR(suites.Iterate8Suite1()->iterate(
                    in_data,
                    0,
                    output_worldP->height,
                    current_result,
                    NULL,              
                    &data,
                    AngleBlurFunc8,
                    next_result));
            }

            PF_EffectWorld* temp = current_result;
            current_result = next_result;
            next_result = temp;
        }

        ERR(suites.WorldTransformSuite1()->copy(
            in_data->effect_ref,
            current_result,
            output_worldP,
            NULL,
            NULL));

        ERR(world_suite->PF_DisposeWorld(in_data->effect_ref, &temp_world1));
        ERR(world_suite->PF_DisposeWorld(in_data->effect_ref, &temp_world2));

        if (basic_suite) {
            basic_suite->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
        }
    }

    return err;
}

static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra,
    bool isGPU)
{
    PF_Err err = PF_Err_NONE,
        err2 = PF_Err_NONE;

    PF_EffectWorld* input_worldP = NULL,
        * output_worldP = NULL;

    MotionDetectionData* infoP = reinterpret_cast<MotionDetectionData*>(extra->input->pre_render_data);

    if (infoP) {
        ERR((extra->cb->checkout_layer_pixels(in_data->effect_ref, MOTIONBLUR_INPUT, &input_worldP)));
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

        AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
            kPFWorldSuite,
            kPFWorldSuiteVersion2,
            out_data);
        PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
        ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

        if (isGPU) {
            ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extra, infoP));
        }
        else {
            ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extra, infoP));
        }
        ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, MOTIONBLUR_INPUT));
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
        "Motion Blur",  
        "DKT Motion Blur",   
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
        case PF_Cmd_SMART_PRE_RENDER:
            err = SmartPreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra);
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

