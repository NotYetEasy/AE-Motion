#include "RandomDisplacement.h"
#include "SimplexNoise.h"
#include "AE_EffectCB.h"

static void ComputeDisplacement(
    double x,
    double y,
    double evolution,
    double seed,
    double scatter,
    double magnitude,
    double* dx,
    double* dy)
{
    // Convert scatter from percentage (0-100) to decimal (0-1)
    double normalizedScatter = scatter / 100.0;

    // Calculate noise value for X displacement
    *dx = SimplexNoise::noise(
        x * normalizedScatter / 50.0 + seed * 54623.245,
        y * normalizedScatter / 500.0,
        evolution + seed * 49235.319798
    );

    // Calculate noise value for Y displacement
    *dy = SimplexNoise::noise(
        x * normalizedScatter / 50.0,
        y * normalizedScatter / 500.0 + seed * 8723.5647,
        evolution + 7468.329 + seed * 19337.940385
    );

    // Apply magnitude - keep the original sign handling from the source code
    *dx *= -magnitude;
    *dy *= magnitude;
}

/**
 * About command handler - displays plugin information
 */
static PF_Err About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Create a shorter about message to prevent crashes
    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
        "Random Displacement v%d.%d\r"
        "Created by DKT.\r"
        "Under development!!\r"
        "Discord: dkt0\r"
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
    // Set version information
    out_data->my_version = PF_VERSION(MAJOR_VERSION,
        MINOR_VERSION,
        BUG_VERSION,
        STAGE_VERSION,
        BUILD_VERSION);

    // Set plugin flags 
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
        PF_OutFlag_PIX_INDEPENDENT |
        PF_OutFlag_USE_OUTPUT_EXTENT |  
        PF_OutFlag_I_EXPAND_BUFFER;

    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
        PF_OutFlag2_FLOAT_COLOR_AWARE |
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

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

    // MAGNITUDE PARAMETER
    PF_ADD_FLOAT_SLIDERX("Magnitude",
        0,
        2000,
        0,
        200,
        50,
        PF_Precision_INTEGER,  // Integer precision (step=1)
        0,
        0,
        MAGNITUDE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // EVOLUTION PARAMETER
    PF_ADD_FLOAT_SLIDERX("Evolution",
        0,
        2000,
        0,
        5,
        0,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        EVOLUTION_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // SEED PARAMETER
    PF_ADD_FLOAT_SLIDERX("Seed",
        0,
        5,
        0,
        5,
        0,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        SEED_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // SCATTER PARAMETER
    PF_ADD_FLOAT_SLIDERX("Scatter",
        RANDOM_DISPLACEMENT_SCATTER_MIN * 100, // 0 -> 0
        RANDOM_DISPLACEMENT_SCATTER_MAX * 100, // 2 -> 200
        RANDOM_DISPLACEMENT_SCATTER_MIN * 100,
        RANDOM_DISPLACEMENT_SCATTER_MAX * 100,
        RANDOM_DISPLACEMENT_SCATTER_DFLT * 100, // 0.5 -> 50
        PF_Precision_INTEGER,
        PF_ValueDisplayFlag_PERCENT,
        0,
        SCATTER_DISK_ID);

    out_data->num_params = RANDOM_DISPLACEMENT_NUM_PARAMS;

    return err;
}

/**
 * Smart PreRender function - prepares for rendering by calculating buffer sizes
 */
static PF_Err SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    DisplacementInfo info;
    AEFX_CLR_STRUCT(info);
    PF_ParamDef param_copy;
    AEFX_CLR_STRUCT(param_copy);

    // Get parameter values
    err = PF_CHECKOUT_PARAM(in_data, RANDOM_DISPLACEMENT_MAGNITUDE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
    if (!err) info.magnitude = param_copy.u.fs_d.value;

    err = PF_CHECKOUT_PARAM(in_data, RANDOM_DISPLACEMENT_EVOLUTION, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
    if (!err) info.evolution = param_copy.u.fs_d.value;

    err = PF_CHECKOUT_PARAM(in_data, RANDOM_DISPLACEMENT_SEED, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
    if (!err) info.seed = param_copy.u.fs_d.value;

    err = PF_CHECKOUT_PARAM(in_data, RANDOM_DISPLACEMENT_SCATTER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
    if (!err) info.scatter = param_copy.u.fs_d.value;

    // Calculate buffer expansion based on magnitude
    A_long expansion = ceil(info.magnitude * 1.1);

    // Set up render request with expanded area
    PF_RenderRequest req = extra->input->output_request;

    if (expansion > 0) {
        req.rect.left -= expansion;
        req.rect.top -= expansion;
        req.rect.right += expansion;
        req.rect.bottom += expansion;
    }
    req.preserve_rgb_of_zero_alpha = TRUE;

    // Checkout the input layer with expanded request
    PF_CheckoutResult checkout;
    err = extra->cb->checkout_layer(in_data->effect_ref,
        RANDOM_DISPLACEMENT_INPUT,
        RANDOM_DISPLACEMENT_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &checkout);

    if (!err) {
        extra->output->max_result_rect = checkout.max_result_rect;
        extra->output->result_rect = checkout.result_rect;

        if (expansion > 0) {
            extra->output->max_result_rect.left -= expansion;
            extra->output->max_result_rect.top -= expansion;
            extra->output->max_result_rect.right += expansion;
            extra->output->max_result_rect.bottom += expansion;
        }

        extra->output->solid = FALSE;
        extra->output->pre_render_data = NULL;
        extra->output->flags = PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;
    }

    return err;
}

/**
 * Smart Render function - performs the actual effect rendering
 */
static PF_Err SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    PF_EffectWorld* input_worldP = NULL;
    PF_EffectWorld* output_worldP = NULL;

    DisplacementInfo info;
    AEFX_CLR_STRUCT(info);

    // Checkout the input & output buffers
    err = extra->cb->checkout_layer_pixels(in_data->effect_ref, RANDOM_DISPLACEMENT_INPUT, &input_worldP);
    if (!err) {
        err = extra->cb->checkout_output(in_data->effect_ref, &output_worldP);
    }

    if (!err && input_worldP && output_worldP) {
        PF_ParamDef param_copy;
        AEFX_CLR_STRUCT(param_copy);

        // Get parameter values
        err = PF_CHECKOUT_PARAM(in_data, RANDOM_DISPLACEMENT_MAGNITUDE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
        if (!err) info.magnitude = param_copy.u.fs_d.value;

        err = PF_CHECKOUT_PARAM(in_data, RANDOM_DISPLACEMENT_EVOLUTION, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
        if (!err) info.evolution = param_copy.u.fs_d.value;

        err = PF_CHECKOUT_PARAM(in_data, RANDOM_DISPLACEMENT_SEED, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
        if (!err) info.seed = param_copy.u.fs_d.value;

        err = PF_CHECKOUT_PARAM(in_data, RANDOM_DISPLACEMENT_SCATTER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
        if (!err) info.scatter = param_copy.u.fs_d.value;

        if (!err) {
            // Clear output buffer before copying
            PF_Pixel empty_pixel = { 0, 0, 0, 0 };
            err = suites.FillMatteSuite2()->fill(
                in_data->effect_ref,
                &empty_pixel,
                NULL,
                output_worldP);

            if (!err) {
                // Calculate the center position of the layer
                double centerX = input_worldP->width / 2.0;
                double centerY = input_worldP->height / 2.0;

                // Calculate displacement values
                double dx, dy;
                ComputeDisplacement(
                    centerX,
                    centerY,
                    info.evolution,
                    info.seed,
                    info.scatter,
                    info.magnitude,
                    &dx,
                    &dy
                );

                // Create a source rect for the entire input
                PF_Rect src_rect = {
                    0, 0,
                    input_worldP->width,
                    input_worldP->height
                };

                // Create a destination rect with the displacement offset
                PF_Rect dest_rect = {
                    static_cast<A_long>(dx),
                    static_cast<A_long>(dy),
                    static_cast<A_long>(dx) + input_worldP->width,
                    static_cast<A_long>(dy) + input_worldP->height
                };

                // Copy the entire layer with the offset position
                err = suites.WorldTransformSuite1()->copy_hq(
                    in_data->effect_ref,
                    input_worldP,
                    output_worldP,
                    &dest_rect,
                    &src_rect);
            }
        }
    }

    // Always check in the input layer pixels
    if (input_worldP) {
        extra->cb->checkin_layer_pixels(in_data->effect_ref, RANDOM_DISPLACEMENT_INPUT);
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

    result = PF_REGISTER_EFFECT(
        inPtr,
        inPluginDataCallBackPtr,
        "Random Displacement",
        "DKT Random Displacement",
        "DKT Effects",
        AE_RESERVED_INFO
    );

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
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    return err;
}
