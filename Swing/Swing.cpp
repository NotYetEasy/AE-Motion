#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>

// Define M_PI if not already defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/**
 * Generates a triangle wave based on input value
 * @param t Input value
 * @return Wave amplitude between -1 and 1
 */
static double
TriangleWave(double t)
{
    // Shift phase by 0.75 and normalize to [0,1]
    t = fmod(t + 0.75, 1.0);

    // Handle negative values
    if (t < 0)
        t += 1.0;

    // Transform to triangle wave [-1,1]
    return (fabs(t - 0.5) - 0.25) * 4.0;
}

static PF_Err
About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    try {
        suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
            "Swing\r"
            "Created by DKT with Unknown's help.\r"
            "Under development!!\r"
            "Discord: dkt0 and unknown1234\r"
            "Contact us if you want to contribute or report bugs!");
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}

static PF_Err
GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    try {
        // Set version information
        out_data->my_version = PF_VERSION(MAJOR_VERSION,
            MINOR_VERSION,
            BUG_VERSION,
            STAGE_VERSION,
            BUILD_VERSION);

        // Set plugin flags for SmartFX
        out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
            PF_OutFlag_PIX_INDEPENDENT |
            PF_OutFlag_NON_PARAM_VARY;


        out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
            PF_OutFlag2_FLOAT_COLOR_AWARE |
            PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
            PF_OutFlag2_REVEALS_ZERO_ALPHA |
            PF_OutFlag2_I_MIX_GUID_DEPENDENCIES;
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
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

    try {
        AEFX_CLR_STRUCT(def);

        // Add frequency parameter
        PF_ADD_FLOAT_SLIDERX("Frequency",
            0.1f,    // min
            16.0f,   // max
            0.1f,    // min_slider
            16.0f,   // max_slider
            2.0f,    // default
            PF_Precision_TENTHS,  // precision
            0,       // display flags
            0,       // flags
            FREQ_DISK_ID);

        AEFX_CLR_STRUCT(def);

        // Add angle 1 parameter - using a regular slider instead of angle
        PF_ADD_FLOAT_SLIDERX("Angle 1",
            -3600.0f,  // min
            3600.0f,   // max
            -360.0f,   // min_slider
            360.0f,    // max_slider
            -30.0f,    // default
            PF_Precision_TENTHS,  // precision
            0,         // display flags
            0,         // flags
            ANGLE1_DISK_ID);

        AEFX_CLR_STRUCT(def);

        // Add angle 2 parameter - using a regular slider instead of angle
        PF_ADD_FLOAT_SLIDERX("Angle 2",
            -3600.0f,  // min
            3600.0f,   // max
            -360.0f,   // min_slider
            360.0f,    // max_slider
            30.0f,     // default
            PF_Precision_TENTHS,  // precision
            0,         // display flags
            0,         // flags
            ANGLE2_DISK_ID);

        AEFX_CLR_STRUCT(def);

        // Add phase parameter
        PF_ADD_FLOAT_SLIDERX("Phase",
            0.0f,    // min
            2.0f,    // max
            0.0f,    // min_slider
            2.0f,    // max_slider
            0.0f,    // default
            PF_Precision_HUNDREDTHS,  // precision
            0,       // display flags
            0,       // flags
            PHASE_DISK_ID);

        AEFX_CLR_STRUCT(def);

        // Add wave type popup
        PF_ADD_POPUP("Wave",
            2,                  // number of choices
            1,                  // default
            "Sine|Triangle",    // choices
            0,                  // flags
            WAVE_TYPE_DISK_ID);

        out_data->num_params = SWING_NUM_PARAMS;
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}

static PF_Err GetLayerAnchorPoint(
    PF_InData* in_data,
    AEGP_LayerH layerH,
    A_Time* current_time,
    PF_Point* anchor_point)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamRefH streamH = NULL;

    // Get the anchor point stream, passing NULL for plugin ID
    err = suites.StreamSuite5()->AEGP_GetNewLayerStream(
        NULL,           // Pass NULL for plugin ID
        layerH,
        AEGP_LayerStream_ANCHORPOINT,
        &streamH);

    if (!err && streamH) {
        AEGP_StreamValue2 stream_value;

        // Get the stream value at current time, passing NULL for plugin ID
        err = suites.StreamSuite5()->AEGP_GetNewStreamValue(
            NULL,       // Pass NULL for plugin ID
            streamH,
            AEGP_LTimeMode_LayerTime,
            current_time,
            true,       // pre-expression
            &stream_value);

        if (!err) {
            anchor_point->x = (A_short)stream_value.val.two_d.x;
            anchor_point->y = (A_short)stream_value.val.two_d.y;

            // Dispose of the stream value
            suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
        }

        // Dispose of the stream
        suites.StreamSuite5()->AEGP_DisposeStream(streamH);
    }

    return err;
}

// Check if there are any keyframes on the frequency parameter
static bool HasAnyFrequencyKeyframes(PF_InData* in_data)
{
    PF_Err err = PF_Err_NONE;
    bool has_keyframes = false;

    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Get the effect reference
    AEGP_EffectRefH effect_ref = NULL;
    AEGP_StreamRefH stream_ref = NULL;
    A_long num_keyframes = 0;

    // Get the effect reference
    if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
        AEGP_EffectRefH aegp_effect_ref = NULL;
        err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(NULL, in_data->effect_ref, &aegp_effect_ref);

        if (!err && aegp_effect_ref) {
            // Get the stream for the frequency parameter
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL,
                aegp_effect_ref,
                SWING_FREQ,
                &stream_ref);

            if (!err && stream_ref) {
                // Check how many keyframes are on this stream
                err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(stream_ref, &num_keyframes);

                // If there are any keyframes, set the flag
                if (!err && num_keyframes > 0) {
                    has_keyframes = true;
                }

                // Dispose of the stream reference
                suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
            }

            // Dispose of the effect reference
            suites.EffectSuite4()->AEGP_DisposeEffect(aegp_effect_ref);
        }
    }

    return has_keyframes;
}



/**
 * Data structure for thread-local rendering information
 */
typedef struct {
    double frequency;
    double angle1;
    double angle2;
    double phase;
    A_long waveType;
    double current_time;
    PF_InData* in_data;
    PF_EffectWorld* input_worldP;
    PF_EffectWorld* output_worldP;
} SwingRenderData;

/**
 * Smart PreRender function - prepares for rendering
 */
static PF_Err SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    try {
        AEGP_SuiteHandler suites(in_data->pica_basicP);

        // Get all parameters for calculating max rotation
        PF_ParamDef freq_param, angle1_param, angle2_param, phase_param, wave_param;
        AEFX_CLR_STRUCT(freq_param);
        AEFX_CLR_STRUCT(angle1_param);
        AEFX_CLR_STRUCT(angle2_param);
        AEFX_CLR_STRUCT(phase_param);
        AEFX_CLR_STRUCT(wave_param);

        ERR(PF_CHECKOUT_PARAM(in_data, SWING_FREQ, in_data->current_time, in_data->time_step, in_data->time_scale, &freq_param));
        ERR(PF_CHECKOUT_PARAM(in_data, SWING_ANGLE1, in_data->current_time, in_data->time_step, in_data->time_scale, &angle1_param));
        ERR(PF_CHECKOUT_PARAM(in_data, SWING_ANGLE2, in_data->current_time, in_data->time_step, in_data->time_scale, &angle2_param));
        ERR(PF_CHECKOUT_PARAM(in_data, SWING_PHASE, in_data->current_time, in_data->time_step, in_data->time_scale, &phase_param));
        ERR(PF_CHECKOUT_PARAM(in_data, SWING_WAVE_TYPE, in_data->current_time, in_data->time_step, in_data->time_scale, &wave_param));

        // Calculate maximum possible rotation angle
        double max_angle = fmax(fabs(angle1_param.u.fs_d.value), fabs(angle2_param.u.fs_d.value));

        // Get downsample factors
        float downsample_x = (float)in_data->downsample_x.den / (float)in_data->downsample_x.num;
        float downsample_y = (float)in_data->downsample_y.den / (float)in_data->downsample_y.num;

        // Calculate buffer expansion based on maximum rotation and scale by downsample factor
        A_long expansion = (A_long)((max_angle / 45.0) * 50 + 100) * (A_long)downsample_x;

        // Get the original request rect
        PF_Rect request_rect = extra->input->output_request.rect;

        // Create our expanded rect accounting for pre-effect source origin
        PF_Rect expanded_rect = request_rect;
        expanded_rect.left = request_rect.left - expansion + in_data->pre_effect_source_origin_x;
        expanded_rect.top = request_rect.top - expansion + in_data->pre_effect_source_origin_y;
        expanded_rect.right = request_rect.right + expansion + in_data->pre_effect_source_origin_x;
        expanded_rect.bottom = request_rect.bottom + expansion + in_data->pre_effect_source_origin_y;

        // Get anchor point and transform
        PF_Point anchor_point = { 0,0 };
        A_Matrix4 transform = { 0 };
        AEGP_LayerH layerH = NULL;

        ERR(suites.LayerSuite9()->AEGP_GetActiveLayer(&layerH));

        if (!err && layerH) {
            A_Time current_time_val;
            current_time_val.value = in_data->current_time;
            current_time_val.scale = in_data->time_scale;

            // Get layer-to-world transform
            ERR(suites.LayerSuite9()->AEGP_GetLayerToWorldXform(layerH, &current_time_val, &transform));

            // Get anchor point
            AEGP_StreamRefH streamH = NULL;
            ERR(suites.StreamSuite5()->AEGP_GetNewLayerStream(
                NULL,
                layerH,
                AEGP_LayerStream_ANCHORPOINT,
                &streamH));

            if (!err && streamH) {
                AEGP_StreamValue2 stream_value;
                ERR(suites.StreamSuite5()->AEGP_GetNewStreamValue(
                    NULL,
                    streamH,
                    AEGP_LTimeMode_LayerTime,
                    &current_time_val,
                    true,
                    &stream_value));

                if (!err) {
                    anchor_point.x = (A_short)stream_value.val.two_d.x;
                    anchor_point.y = (A_short)stream_value.val.two_d.y;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }
        }

        // Create data structure for GuidMixInPtr
        struct {
            double frequency;
            double angle1;
            double angle2;
            double phase;
            A_long waveType;
            PF_Rect expanded_rect;
            PF_Point anchor_point;
            A_Matrix4 transform;
            float pre_effect_source_origin_x;
            float pre_effect_source_origin_y;
            float downsample_x;
            float downsample_y;
        } mix_data;

        // Fill the structure
        mix_data.frequency = freq_param.u.fs_d.value;
        mix_data.angle1 = angle1_param.u.fs_d.value;
        mix_data.angle2 = angle2_param.u.fs_d.value;
        mix_data.phase = phase_param.u.fs_d.value;
        mix_data.waveType = wave_param.u.pd.value - 1;
        mix_data.expanded_rect = expanded_rect;
        mix_data.anchor_point = anchor_point;
        mix_data.transform = transform;
        mix_data.pre_effect_source_origin_x = in_data->pre_effect_source_origin_x;
        mix_data.pre_effect_source_origin_y = in_data->pre_effect_source_origin_y;
        mix_data.downsample_x = downsample_x;
        mix_data.downsample_y = downsample_y;

        // Mix in the data for caching
        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(mix_data), &mix_data));

        // Set up render request with expanded area
        PF_RenderRequest req = extra->input->output_request;
        req.rect = expanded_rect;
        req.preserve_rgb_of_zero_alpha = TRUE;

        // Checkout the input layer with expanded request
        PF_CheckoutResult checkout_result;
        ERR(extra->cb->checkout_layer(in_data->effect_ref,
            SWING_INPUT,
            SWING_INPUT,
            &req,
            in_data->current_time,
            in_data->time_step,
            in_data->time_scale,
            &checkout_result));

        // Set our output rects
        extra->output->max_result_rect = expanded_rect;
        extra->output->result_rect = expanded_rect;
        extra->output->solid = FALSE;
        extra->output->flags |= PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;

        // Check in parameters
        ERR(PF_CHECKIN_PARAM(in_data, &freq_param));
        ERR(PF_CHECKIN_PARAM(in_data, &angle1_param));
        ERR(PF_CHECKIN_PARAM(in_data, &angle2_param));
        ERR(PF_CHECKIN_PARAM(in_data, &phase_param));
        ERR(PF_CHECKIN_PARAM(in_data, &wave_param));
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}


static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    try {
        AEGP_SuiteHandler suites(in_data->pica_basicP);

        // Get all parameters
        PF_ParamDef freq_param, angle1_param, angle2_param, phase_param, wave_param;
        AEFX_CLR_STRUCT(freq_param);
        AEFX_CLR_STRUCT(angle1_param);
        AEFX_CLR_STRUCT(angle2_param);
        AEFX_CLR_STRUCT(phase_param);
        AEFX_CLR_STRUCT(wave_param);

        ERR(PF_CHECKOUT_PARAM(in_data, SWING_FREQ, in_data->current_time, in_data->time_step, in_data->time_scale, &freq_param));
        ERR(PF_CHECKOUT_PARAM(in_data, SWING_ANGLE1, in_data->current_time, in_data->time_step, in_data->time_scale, &angle1_param));
        ERR(PF_CHECKOUT_PARAM(in_data, SWING_ANGLE2, in_data->current_time, in_data->time_step, in_data->time_scale, &angle2_param));
        ERR(PF_CHECKOUT_PARAM(in_data, SWING_PHASE, in_data->current_time, in_data->time_step, in_data->time_scale, &phase_param));
        ERR(PF_CHECKOUT_PARAM(in_data, SWING_WAVE_TYPE, in_data->current_time, in_data->time_step, in_data->time_scale, &wave_param));

        // Checkout input layer pixels
        PF_EffectWorld* input_worldP = NULL;
        err = extra->cb->checkout_layer_pixels(in_data->effect_ref, SWING_INPUT, &input_worldP);
        if (err) return err;

        // Checkout output buffer
        PF_EffectWorld* output_worldP = NULL;
        err = extra->cb->checkout_output(in_data->effect_ref, &output_worldP);
        if (err) return err;

        if (input_worldP && output_worldP) {
            // Get pixel format
            PF_PixelFormat pixelFormat;
            AEGP_WorldType worldType;

            PF_WorldSuite2* wsP = NULL;
            ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
            ERR(wsP->PF_GetPixelFormat(output_worldP, &pixelFormat));
            ERR(suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2));

            switch (pixelFormat) {
            case PF_PixelFormat_ARGB128:
                worldType = AEGP_WorldType_32;
                break;
            case PF_PixelFormat_ARGB64:
                worldType = AEGP_WorldType_16;
                break;
            case PF_PixelFormat_ARGB32:
                worldType = AEGP_WorldType_8;
                break;
            }

            // Calculate animation parameters
            double current_time = (double)in_data->current_time / (double)in_data->time_scale;
            double frequency = freq_param.u.fs_d.value;
            double angle1 = angle1_param.u.fs_d.value;
            double angle2 = angle2_param.u.fs_d.value;
            double phase = phase_param.u.fs_d.value;
            A_long waveType = wave_param.u.pd.value - 1;

            double effective_phase = phase + (current_time * frequency);
            double m = (waveType == 0) ? sin(effective_phase * PF_PI) : TriangleWave(effective_phase / 2.0);
            double t = (m + 1.0) / 2.0;
            double finalAngle = angle1 + t * (angle2 - angle1);
            double angleRad = -finalAngle * PF_PI / 180.0;

            // Get anchor point
            PF_Point anchor_point = { 0,0 };
            AEGP_LayerH layerH = NULL;

            ERR(suites.LayerSuite9()->AEGP_GetActiveLayer(&layerH));

            if (!err && layerH) {
                A_Time current_time_val;
                current_time_val.value = in_data->current_time;
                current_time_val.scale = in_data->time_scale;

                AEGP_StreamRefH streamH = NULL;
                ERR(suites.StreamSuite5()->AEGP_GetNewLayerStream(
                    NULL,
                    layerH,
                    AEGP_LayerStream_ANCHORPOINT,
                    &streamH));

                if (!err && streamH) {
                    AEGP_StreamValue2 stream_value;
                    ERR(suites.StreamSuite5()->AEGP_GetNewStreamValue(
                        NULL,
                        streamH,
                        AEGP_LTimeMode_LayerTime,
                        &current_time_val,
                        true,
                        &stream_value));

                    if (!err) {
                        anchor_point.x = (A_short)stream_value.val.two_d.x;
                        anchor_point.y = (A_short)stream_value.val.two_d.y;
                        suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
                    }
                    suites.StreamSuite5()->AEGP_DisposeStream(streamH);
                }
            }

            // Calculate proper center coordinates
            float center_x = anchor_point.x + in_data->pre_effect_source_origin_x;
            float center_y = anchor_point.y + in_data->pre_effect_source_origin_y;

            // Calculate input to output offset
            float input_offset_x = ((output_worldP->width - input_worldP->width) / 2.0f);
            float input_offset_y = ((output_worldP->height - input_worldP->height) / 2.0f);

            float sin_rot = sin(angleRad);
            float cos_rot = cos(angleRad);

            // Setup sampling parameters
            PF_SampPB samp_pb;
            AEFX_CLR_STRUCT(samp_pb);
            samp_pb.src = input_worldP;

            for (A_long y = 0; y < output_worldP->height; y++) {
                for (A_long x = 0; x < output_worldP->width; x++) {
                    // Adjust coordinates relative to anchor point
                    float dx = (x - input_offset_x) - center_x;
                    float dy = (y - input_offset_y) - center_y;

                    // Apply rotation
                    float rotated_x = center_x + (dx * cos_rot - dy * sin_rot);
                    float rotated_y = center_y + (dx * sin_rot + dy * cos_rot);

                    // Convert back to source space
                    float src_x = rotated_x - in_data->pre_effect_source_origin_x;
                    float src_y = rotated_y - in_data->pre_effect_source_origin_y;

                    PF_Fixed fix_x = (PF_Fixed)(src_x * 65536.0f);
                    PF_Fixed fix_y = (PF_Fixed)(src_y * 65536.0f);

                    if (src_x >= 0 && src_x < input_worldP->width &&
                        src_y >= 0 && src_y < input_worldP->height) {

                        switch (worldType) {
                        case AEGP_WorldType_8: {
                            PF_Pixel8* outRow = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                            PF_Pixel8 sample;
                            ERR(suites.Sampling8Suite1()->subpixel_sample(in_data->effect_ref,
                                fix_x, fix_y, &samp_pb, &sample));
                            outRow[x] = sample;
                            break;
                        }
                        case AEGP_WorldType_16: {
                            PF_Pixel16* outRow = (PF_Pixel16*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                            PF_Pixel16 sample;
                            ERR(suites.Sampling16Suite1()->subpixel_sample16(in_data->effect_ref,
                                fix_x, fix_y, &samp_pb, &sample));
                            outRow[x] = sample;
                            break;
                        }
                        case AEGP_WorldType_32: {
                            PF_PixelFloat* outRow = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                            PF_PixelFloat sample;
                            ERR(suites.SamplingFloatSuite1()->subpixel_sample_float(in_data->effect_ref,
                                fix_x, fix_y, &samp_pb, &sample));
                            outRow[x] = sample;
                            break;
                        }
                        }
                    }
                    else {
                        switch (worldType) {
                        case AEGP_WorldType_8: {
                            PF_Pixel8* outRow = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                            outRow[x].alpha = 0;
                            outRow[x].red = outRow[x].green = outRow[x].blue = 0;
                            break;
                        }
                        case AEGP_WorldType_16: {
                            PF_Pixel16* outRow = (PF_Pixel16*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                            outRow[x].alpha = 0;
                            outRow[x].red = outRow[x].green = outRow[x].blue = 0;
                            break;
                        }
                        case AEGP_WorldType_32: {
                            PF_PixelFloat* outRow = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                            outRow[x].alpha = 0;
                            outRow[x].red = outRow[x].green = outRow[x].blue = 0;
                            break;
                        }
                        }
                    }
                }
            }
        }

        // Check in resources
        if (input_worldP) {
            err = extra->cb->checkin_layer_pixels(in_data->effect_ref, SWING_INPUT);
        }

        ERR(PF_CHECKIN_PARAM(in_data, &freq_param));
        ERR(PF_CHECKIN_PARAM(in_data, &angle1_param));
        ERR(PF_CHECKIN_PARAM(in_data, &angle2_param));
        ERR(PF_CHECKIN_PARAM(in_data, &phase_param));
        ERR(PF_CHECKIN_PARAM(in_data, &wave_param));
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}


/**
 * Legacy rendering function for older AE versions
 */
static PF_Err
LegacyRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;

    try {
        AEGP_SuiteHandler suites(in_data->pica_basicP);

        // Get parameters
        double frequency = params[SWING_FREQ]->u.fs_d.value;
        double angle1 = params[SWING_ANGLE1]->u.fs_d.value;
        double angle2 = params[SWING_ANGLE2]->u.fs_d.value;
        double phase = params[SWING_PHASE]->u.fs_d.value;
        A_long waveType = params[SWING_WAVE_TYPE]->u.pd.value - 1;

        // Calculate current time in seconds
        double current_time = (double)in_data->current_time / (double)in_data->time_scale;

        // Calculate effective phase (phase + time * frequency)
        double effective_phase = phase + (current_time * frequency);

        // Calculate modulation value based on wave type (0 = Sine, 1 = Triangle)
        double m;
        if (waveType == 0) {
            // Sine wave
            m = sin(effective_phase * M_PI);
        }
        else {
            // Triangle wave
            m = TriangleWave(effective_phase / 2.0);
        }

        // Map modulation from -1...1 to 0...1
        double t = (m + 1.0) / 2.0;

        // Calculate final angle by interpolating between angle1 and angle2
        double finalAngle = angle1 + t * (angle2 - angle1);

        // Get input layer
        PF_EffectWorld* input_worldP = &params[SWING_INPUT]->u.ld;
        PF_EffectWorld* output_worldP = output;

        // If the angle is very small, just do a direct copy
        if (fabs(finalAngle) < 0.01) {
            // Simple memcpy approach for near-zero angles
            for (A_long y = 0; y < output_worldP->height; y++) {
                char* inRow = (char*)input_worldP->data + y * input_worldP->rowbytes;
                char* outRow = (char*)output_worldP->data + y * output_worldP->rowbytes;

                // Copy entire row (works regardless of pixel format)
                memcpy(outRow, inRow, output_worldP->rowbytes);
            }
        }
        else {
            // Convert angle to radians
            double angleRad = finalAngle * PF_RAD_PER_DEGREE;

            // Center of rotation is the center of the input world
            PF_FpLong centerX = input_worldP->width / 2.0;
            PF_FpLong centerY = input_worldP->height / 2.0;

            // Calculate trig values once
            double cos_angle = cos(angleRad);
            double sin_angle = sin(angleRad);

            // Check if we're in 16-bit mode by examining world structure
            bool is16bit = false;
            bool is32bit = false;

            // Determine pixel depth by examining the world
            double bytesPerPixel = (double)input_worldP->rowbytes / (double)input_worldP->width;

            if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
                is32bit = true;
            }
            else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
                is16bit = true;
            }

            // Perform rotation based on pixel depth
            if (is32bit) {
                // 32-bit floating point
                for (A_long y = 0; y < output_worldP->height; y++) {
                    PF_PixelFloat* outRow = (PF_PixelFloat*)((char*)output_worldP->data +
                        y * output_worldP->rowbytes);

                    for (A_long x = 0; x < output_worldP->width; x++) {
                        // Calculate the source point by applying inverse rotation
                        double dx = x - centerX;
                        double dy = y - centerY;

                        double srcX = centerX + (dx * cos_angle + dy * sin_angle);
                        double srcY = centerY + (-dx * sin_angle + dy * cos_angle);

                        // Check if source point is within bounds
                        if (srcX >= 0 && srcX < input_worldP->width - 1 &&
                            srcY >= 0 && srcY < input_worldP->height - 1) {

                            // Simple nearest neighbor sampling
                            A_long sx = (A_long)(srcX + 0.5);
                            A_long sy = (A_long)(srcY + 0.5);

                            PF_PixelFloat* inRow = (PF_PixelFloat*)((char*)input_worldP->data +
                                sy * input_worldP->rowbytes);

                            outRow[x] = inRow[sx];
                        }
                        else {
                            // Outside the source image, set to transparent black
                            outRow[x].alpha = 0;
                            outRow[x].red = 0;
                            outRow[x].green = 0;
                            outRow[x].blue = 0;
                        }
                    }
                }
            }
            else if (is16bit) {
                // 16-bit
                for (A_long y = 0; y < output_worldP->height; y++) {
                    PF_Pixel16* outRow = (PF_Pixel16*)((char*)output_worldP->data +
                        y * output_worldP->rowbytes);

                    for (A_long x = 0; x < output_worldP->width; x++) {
                        // Calculate the source point by applying inverse rotation
                        double dx = x - centerX;
                        double dy = y - centerY;

                        double srcX = centerX + (dx * cos_angle + dy * sin_angle);
                        double srcY = centerY + (-dx * sin_angle + dy * cos_angle);

                        // Check if source point is within bounds
                        if (srcX >= 0 && srcX < input_worldP->width - 1 &&
                            srcY >= 0 && srcY < input_worldP->height - 1) {

                            // Simple nearest neighbor sampling
                            A_long sx = (A_long)(srcX + 0.5);
                            A_long sy = (A_long)(srcY + 0.5);

                            PF_Pixel16* inRow = (PF_Pixel16*)((char*)input_worldP->data +
                                sy * input_worldP->rowbytes);

                            outRow[x] = inRow[sx];
                        }
                        else {
                            // Outside the source image, set to transparent black
                            outRow[x].alpha = 0;
                            outRow[x].red = 0;
                            outRow[x].green = 0;
                            outRow[x].blue = 0;
                        }
                    }
                }
            }
            else {
                // 8-bit (default)
                for (A_long y = 0; y < output_worldP->height; y++) {
                    PF_Pixel8* outRow = (PF_Pixel8*)((char*)output_worldP->data +
                        y * output_worldP->rowbytes);

                    for (A_long x = 0; x < output_worldP->width; x++) {
                        // Calculate the source point by applying inverse rotation
                        double dx = x - centerX;
                        double dy = y - centerY;

                        double srcX = centerX + (dx * cos_angle + dy * sin_angle);
                        double srcY = centerY + (-dx * sin_angle + dy * cos_angle);

                        // Check if source point is within bounds
                        if (srcX >= 0 && srcX < input_worldP->width - 1 &&
                            srcY >= 0 && srcY < input_worldP->height - 1) {

                            // Simple nearest neighbor sampling
                            A_long sx = (A_long)(srcX + 0.5);
                            A_long sy = (A_long)(srcY + 0.5);

                            PF_Pixel8* inRow = (PF_Pixel8*)((char*)input_worldP->data + sy * input_worldP->rowbytes);

                            outRow[x] = inRow[sx];
                        }
                        else {
                            // Outside the source image, set to transparent black
                            outRow[x].alpha = 0;
                            outRow[x].red = 0;
                            outRow[x].green = 0;
                            outRow[x].blue = 0;
                        }
                    }
                }
            }
        }
    }
    catch (const std::exception& e) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}

/**
 * Plugin registration function
 */
extern "C" DllExport
PF_Err PluginDataEntryFunction(
    PF_PluginDataPtr inPtr,
    PF_PluginDataCB inPluginDataCallBackPtr,
    SPBasicSuite* inSPBasicSuitePtr,
    const char* inHostName,
    const char* inHostVersion)
{
    PF_Err result = PF_Err_INVALID_CALLBACK;

    try {
        // Register the effect with After Effects
        result = PF_REGISTER_EFFECT(
            inPtr,
            inPluginDataCallBackPtr,
            "Swing",             // Effect name
            "DKT Swing",         // Match name - make sure this is unique
            "DKT Effects",       // Category
            AE_RESERVED_INFO
        );
    }
    catch (...) {
        result = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return result;
}

PF_Err
EffectMain(
    PF_Cmd            cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err err = PF_Err_NONE;

    try {
        switch (cmd) {
        case PF_Cmd_ABOUT:
            err = About(in_data, out_data, params, output);
            break;

        case PF_Cmd_GLOBAL_SETUP:
            err = GlobalSetup(in_data, out_data, params, output);
            break;

        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_data, out_data, params, output);
            break;

        case PF_Cmd_SMART_PRE_RENDER:
            err = SmartPreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
            break;

        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra);
            break;

        case PF_Cmd_RENDER:
            // Fallback for older versions that don't support Smart Render
            err = LegacyRender(in_data, out_data, params, output);
            break;

        default:
            break;
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    catch (const std::exception& e) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}