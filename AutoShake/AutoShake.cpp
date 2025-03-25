/**
 * AutoShake.cpp
 * After Effects plugin that applies realistic camera shake to layers
 * Created by DKT
 */

#include "AutoShake.h"
#include <cmath>
#include <mutex>

 /**
  * About command handler - displays plugin information
  */
static PF_Err About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Create a shorter about message to prevent crashes
    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
        "Auto-Shake v%d.%d\r"
        "Created by DKT with Unknown's help.\r"
        "Under development!!\r"
        "Discord: dkt0 ; unknown1234\r"
        "Contact me if you want to contribute or report bugs!",
        MAJOR_VERSION,
        MINOR_VERSION);
    return PF_Err_NONE;
}

/**
 * Global setup - registers plugin capabilities
 */
static PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    // Set version information - ensure this matches the PiPL version (1.0)
    out_data->my_version = PF_VERSION(1,
        0,
        0,
        0,
        0);

    // Set plugin flags
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
        PF_OutFlag_PIX_INDEPENDENT |
        PF_OutFlag_I_EXPAND_BUFFER |
        PF_OutFlag_SEND_UPDATE_PARAMS_UI |
        PF_OutFlag_WIDE_TIME_INPUT |
        PF_OutFlag_NON_PARAM_VARY |
        PF_OutFlag_FORCE_RERENDER |
        PF_OutFlag_I_HAVE_EXTERNAL_DEPENDENCIES;


    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
        PF_OutFlag2_FLOAT_COLOR_AWARE |
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
        PF_OutFlag2_REVEALS_ZERO_ALPHA |
        PF_OutFlag2_PRESERVES_FULLY_OPAQUE_PIXELS |
        PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT;

    return PF_Err_NONE;
}

/**
 * Sets up the parameters for the effect
 */
static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);

    // Magnitude parameter
    PF_ADD_FLOAT_SLIDERX("Magnitude",
        0,       // Min
        2000,    // Max
        0,       // Min display
        2000,    // Max display
        50,      // Default
        PF_Precision_INTEGER,
        0,
        0,
        MAGNITUDE_DISK_ID);

    // Frequency parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Frequency (Hz)",
        0,       // Min
        16,      // Max
        0,       // Min display
        5,       // Max display
        2,       // Default
        PF_Precision_HUNDREDTHS,
        0,
        0,
        FREQUENCY_DISK_ID);

    // Evolution parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Evolution",
        0,       // Min
        2000,    // Max
        0,       // Min display
        2,       // Max display
        0,       // Default
        PF_Precision_HUNDREDTHS,
        0,
        0,
        EVOLUTION_DISK_ID);

    // Seed parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Seed",
        0,       // Min
        5,       // Max
        0,       // Min display
        5,       // Max display
        0,       // Default
        PF_Precision_HUNDREDTHS,
        0,
        0,
        SEED_DISK_ID);

    // Angle parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_ANGLE("Angle",
        AUTOSHAKE_ANGLE_DFLT, // Default (45 degrees)
        ANGLE_DISK_ID);

    // Slack parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Slack",
        0,         // Min
        100,       // Max
        0,         // Min display
        100,       // Max display (display as percentage)
        25,        // Default (25%)
        PF_Precision_INTEGER,
        PF_ValueDisplayFlag_PERCENT, // Display as percentage
        0,
        SLACK_DISK_ID);

    // Z Shake parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Z Shake",
        AUTOSHAKE_ZSHAKE_MIN,     // Min
        AUTOSHAKE_ZSHAKE_MAX,     // Max
        AUTOSHAKE_ZSHAKE_MIN,     // Min display
        AUTOSHAKE_ZSHAKE_MAX,     // Max display
        AUTOSHAKE_ZSHAKE_DFLT,    // Default
        PF_Precision_INTEGER,
        0,
        0,
        ZSHAKE_DISK_ID);

    out_data->num_params = AUTOSHAKE_NUM_PARAMS;

    return err;
}

/**
 * Calculate shake offsets based on parameters and time
 */
static void CalculateShakeOffsets(
    ShakeInfo* info,
    PF_FpLong current_time,
    PF_FpLong* rx,
    PF_FpLong* ry,
    PF_FpLong* dz)
{
    // Calculate angle in radians for direction vector
    PF_FpLong angleRad = info->angle * (PF_PI / 180.0);
    PF_FpLong s = -sin(angleRad);
    PF_FpLong c = -cos(angleRad);

    // Calculate evolution value
    PF_FpLong evolutionValue = info->evolution + info->frequency * current_time;

    // Generate noise values using SimplexNoise
    PF_FpLong dx = SimplexNoise::noise(evolutionValue, info->seed * 49235.319798);
    PF_FpLong dy = SimplexNoise::noise(evolutionValue + 7468.329, info->seed * 19337.940385);
    *dz = SimplexNoise::noise(evolutionValue + 14192.277, info->seed * 71401.168533);

    // Scale noise by parameters
    dx *= info->magnitude;
    dy *= info->magnitude * info->slack;
    *dz *= info->zshake;

    // Apply rotation to get final offset
    *rx = dx * c - dy * s;
    *ry = dx * s + dy * c;
}

/**
 * Legacy render function for older AE versions
 */
static PF_Err LegacyRender(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Get parameter values
    ShakeInfo info;
    AEFX_CLR_STRUCT(info);

    info.magnitude = params[AUTOSHAKE_MAGNITUDE]->u.fs_d.value;
    info.frequency = params[AUTOSHAKE_FREQUENCY]->u.fs_d.value;
    info.evolution = params[AUTOSHAKE_EVOLUTION]->u.fs_d.value;
    info.seed = params[AUTOSHAKE_SEED]->u.fs_d.value;
    info.angle = params[AUTOSHAKE_ANGLE]->u.ad.value / (PF_FpLong)(PF_RAD_PER_DEGREE); // Convert to degrees
    info.slack = params[AUTOSHAKE_SLACK]->u.fs_d.value / 100.0; // Convert from percentage
    info.zshake = params[AUTOSHAKE_ZSHAKE]->u.fs_d.value;

    // Convert current time to seconds
    PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;

    // Calculate shake offsets
    PF_FpLong rx, ry, dz;
    CalculateShakeOffsets(&info, current_time, &rx, &ry, &dz);

    // For legacy rendering, we need to manually transform and copy pixels
    PF_Rect src_rect;
    PF_Rect dest_rect = output->extent_hint;

    // Calculate source rectangle
    if (dz != 0) {
        // For Z shake, we need to scale the content
        PF_FpLong scale = 100.0 + (dz / 10.0);

        PF_FpLong centerX = output->width / 2.0;
        PF_FpLong centerY = output->height / 2.0;

        PF_FpLong scaledWidth = (params[AUTOSHAKE_INPUT]->u.ld.width * 100.0) / scale;
        PF_FpLong scaledHeight = (params[AUTOSHAKE_INPUT]->u.ld.height * 100.0) / scale;

        src_rect.left = centerX - (scaledWidth / 2.0) - rx;
        src_rect.top = centerY - (scaledHeight / 2.0) - ry;
        src_rect.right = src_rect.left + scaledWidth;
        src_rect.bottom = src_rect.top + scaledHeight;
    }
    else {
        // For non-scaled content
        PF_FpLong centerX = output->width / 2.0;
        PF_FpLong centerY = output->height / 2.0;
        PF_FpLong halfWidth = params[AUTOSHAKE_INPUT]->u.ld.width / 2.0;
        PF_FpLong halfHeight = params[AUTOSHAKE_INPUT]->u.ld.height / 2.0;

        src_rect.left = centerX - halfWidth - rx;
        src_rect.top = centerY - halfHeight - ry;
        src_rect.right = src_rect.left + params[AUTOSHAKE_INPUT]->u.ld.width;
        src_rect.bottom = src_rect.top + params[AUTOSHAKE_INPUT]->u.ld.height;
    }

    // Clear output buffer
    PF_Pixel empty_pixel = { 0, 0, 0, 0 };
    ERR(PF_FILL(&empty_pixel, &dest_rect, output));

    // Use legacy transform to copy pixels
    if (!err) {
        ERR(suites.WorldTransformSuite1()->copy_hq(
            in_data->effect_ref,
            &params[AUTOSHAKE_INPUT]->u.ld,
            output,
            NULL,
            &src_rect));
    }

    return err;
}

/**
 * Pre-render function for Smart Render pipeline
 * Determines buffer requirements and prepares for rendering
 */
static PF_Err SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    // Initialize effect info structure
    ShakeInfo info;
    AEFX_CLR_STRUCT(info);

    PF_ParamDef param_copy;
    AEFX_CLR_STRUCT(param_copy);

    // Initialize max_result_rect to the current output request rect
    extra->output->max_result_rect = extra->input->output_request.rect;

    // Get magnitude parameter
    ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_MAGNITUDE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) info.magnitude = param_copy.u.fs_d.value;

    // Get Z shake parameter
    ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_ZSHAKE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) info.zshake = param_copy.u.fs_d.value;

    // Calculate buffer expansion based on magnitude
    A_long expansion = ceil(info.magnitude * 1.5);

    // Set up render request with expanded area
    PF_RenderRequest req = extra->input->output_request;

    if (expansion > 0) {
        req.rect.left -= expansion;
        req.rect.top -= expansion;
        req.rect.right += expansion;
        req.rect.bottom += expansion;
    }
    req.preserve_rgb_of_zero_alpha = TRUE;

    // Checkout the input layer with our expanded request
    PF_CheckoutResult checkout;
    ERR(extra->cb->checkout_layer(in_data->effect_ref,
        AUTOSHAKE_INPUT,
        AUTOSHAKE_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &checkout));

    if (!err) {
        // Update max_result_rect based on checkout result
        extra->output->max_result_rect = checkout.max_result_rect;

        // Apply expansion to max_result_rect
        if (expansion > 0) {
            extra->output->max_result_rect.left -= expansion;
            extra->output->max_result_rect.top -= expansion;
            extra->output->max_result_rect.right += expansion;
            extra->output->max_result_rect.bottom += expansion;
        }

        // Set result_rect to match max_result_rect
        extra->output->result_rect = extra->output->max_result_rect;

        // Set output flags
        extra->output->solid = FALSE;
        extra->output->pre_render_data = NULL;
        extra->output->flags = PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;
    }

    return err;
}

/**
 * Data structure for thread-local rendering information
 */
typedef struct {
    ShakeInfo info;
    PF_FpLong current_time;
    PF_InData* in_data;
    PF_EffectWorld* input_worldP;
    PF_EffectWorld* output_worldP;
} ThreadRenderData;

/**
 * Smart Render function - performs the actual effect rendering
 */
static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Use thread_local to ensure each thread has its own render data
    // This improves performance in multi-threaded rendering
    thread_local ThreadRenderData render_data;
    AEFX_CLR_STRUCT(render_data);
    render_data.in_data = in_data;

    // Checkout input layer pixels
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, AUTOSHAKE_INPUT, &render_data.input_worldP));
    if (!err) {
        // Checkout output buffer
        ERR(extra->cb->checkout_output(in_data->effect_ref, &render_data.output_worldP));
    }

    if (!err && render_data.input_worldP && render_data.output_worldP) {
        PF_ParamDef param_copy;
        AEFX_CLR_STRUCT(param_copy);

        // Get all effect parameters
        ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_MAGNITUDE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.magnitude = param_copy.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_FREQUENCY, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.frequency = param_copy.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_EVOLUTION, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.evolution = param_copy.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_SEED, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.seed = param_copy.u.fs_d.value;

        ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.angle = param_copy.u.ad.value / (PF_FpLong)(PF_RAD_PER_DEGREE); // Convert to degrees

        ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_SLACK, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.slack = param_copy.u.fs_d.value / 100.0; // Convert from percentage

        ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_ZSHAKE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.zshake = param_copy.u.fs_d.value;

        if (!err) {
            // Convert current time to seconds
            render_data.current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;

            // Calculate shake offsets
            PF_FpLong rx, ry, dz;
            CalculateShakeOffsets(&render_data.info, render_data.current_time, &rx, &ry, &dz);

            // Calculate Z scale (if Z shake is enabled)
            PF_FpLong scale = 100.0;
            if (render_data.info.zshake != 0) {
                scale = 100.0 + (dz / 10.0);
            }

            // Clear output buffer before copying (performance optimization)
            PF_Pixel empty_pixel = { 0, 0, 0, 0 };
            ERR(suites.FillMatteSuite2()->fill(
                in_data->effect_ref,
                &empty_pixel,
                NULL,
                render_data.output_worldP));

            if (!err) {
                // Calculate the source rectangle based on scale and offset
                PF_Rect src_rect;

                // Calculate expansion used in SmartPreRender
                A_long expansion = ceil(render_data.info.magnitude * 1.5);

                // Handle source rectangle differently based on whether scaling is applied
                if (scale != 100.0) {
                    // For scaled content - use center of expanded buffer
                    PF_FpLong centerX = render_data.output_worldP->width / 2.0;
                    PF_FpLong centerY = render_data.output_worldP->height / 2.0;

                    // Calculate scaled dimensions
                    PF_FpLong scaledWidth = (render_data.input_worldP->width * 100.0) / scale;
                    PF_FpLong scaledHeight = (render_data.input_worldP->height * 100.0) / scale;

                    // Position the source rectangle
                    src_rect.left = centerX - (scaledWidth / 2.0) - rx;
                    src_rect.top = centerY - (scaledHeight / 2.0) - ry;
                    src_rect.right = src_rect.left + scaledWidth;
                    src_rect.bottom = src_rect.top + scaledHeight;
                }
                else {
                    // For non-scaled content
                    // Place the layer in the center of the expanded buffer
                    PF_FpLong centerX = render_data.output_worldP->width / 2.0;
                    PF_FpLong centerY = render_data.output_worldP->height / 2.0;
                    PF_FpLong halfWidth = render_data.input_worldP->width / 2.0;
                    PF_FpLong halfHeight = render_data.input_worldP->height / 2.0;

                    // Position the source rectangle
                    src_rect.left = centerX - halfWidth - rx;
                    src_rect.top = centerY - halfHeight - ry;
                    src_rect.right = src_rect.left + render_data.input_worldP->width;
                    src_rect.bottom = src_rect.top + render_data.input_worldP->height;
                }

                // Use high-quality copy for transformation
                ERR(suites.WorldTransformSuite1()->copy_hq(
                    in_data->effect_ref,
                    render_data.input_worldP,
                    render_data.output_worldP,
                    NULL,  // No destination rect - use entire output buffer
                    &src_rect));
            }
        }
    }

    // Check in the input layer pixels
    if (render_data.input_worldP) {
        ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, AUTOSHAKE_INPUT));
    }

    return err;
}

/**
 * Plugin registration function
 */
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
        "Auto-Shake",    // Name
        "DKT Auto-Shake", // Match Name
        "DKT Effects",   // Category
        AE_RESERVED_INFO, // Reserved Info
        "EffectMain",    // Entry point
        "https://www.adobe.com"); // support URL

    return result;
}

/**
 * Main entry point for the effect
 * Handles all command dispatching
 */
PF_Err EffectMain(
    PF_Cmd cmd,
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
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }

    return err;
}
