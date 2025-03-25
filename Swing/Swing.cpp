#define NOMINMAX

#include "Swing.h"
#include <math.h>
#include <fstream>
#include <chrono>
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
static PF_Err
SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    try {
        // Initialize effect info structure
        SwingInfo info;
        AEFX_CLR_STRUCT(info);

        PF_ParamDef param_copy;
        AEFX_CLR_STRUCT(param_copy);

        // Initialize max_result_rect to the current output request rect
        extra->output->max_result_rect = extra->input->output_request.rect;

        // Get angle parameters to calculate maximum rotation expansion
        err = PF_CHECKOUT_PARAM(in_data, SWING_ANGLE1, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
        if (!err) {
            info.angle1 = param_copy.u.fs_d.value;
        }
        else {
            return err;
        }

        err = PF_CHECKOUT_PARAM(in_data, SWING_ANGLE2, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
        if (!err) {
            info.angle2 = param_copy.u.fs_d.value;
        }
        else {
            return err;
        }

        // Calculate maximum rotation angle (absolute value)
        double max_angle = fmax(fabs(info.angle1), fabs(info.angle2));

        // Calculate buffer expansion based on maximum rotation angle - INCREASED for better coverage
        A_long expansion = (A_long)(max_angle / 30.0) * 30 + 50;

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
        err = extra->cb->checkout_layer(in_data->effect_ref,
            SWING_INPUT,
            SWING_INPUT,
            &req,
            in_data->current_time,
            in_data->time_step,
            in_data->time_scale,
            &checkout);

        if (!err) {
            // Update max_result_rect based on checkout result
            extra->output->max_result_rect = checkout.max_result_rect;

            // Set result_rect to match max_result_rect
            extra->output->result_rect = extra->output->max_result_rect;

            // Set output flags
            extra->output->solid = FALSE;
            extra->output->pre_render_data = NULL;
            extra->output->flags = PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;
        }
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}

/**
* Smart Render function - performs the actual effect rendering
*/
static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    try {
        AEGP_SuiteHandler suites(in_data->pica_basicP);

        // Checkout input layer pixels
        PF_EffectWorld* input_worldP = NULL;
        err = extra->cb->checkout_layer_pixels(in_data->effect_ref, SWING_INPUT, &input_worldP);
        if (err) {
            return err;
        }

        // Checkout output buffer
        PF_EffectWorld* output_worldP = NULL;
        err = extra->cb->checkout_output(in_data->effect_ref, &output_worldP);
        if (err) {
            return err;
        }

        if (input_worldP && output_worldP) {
            PF_ParamDef param_copy;
            AEFX_CLR_STRUCT(param_copy);

            // Get all effect parameters
            // Frequency
            double frequency = 2.0;
            err = PF_CHECKOUT_PARAM(in_data, SWING_FREQ, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
            if (!err) {
                frequency = param_copy.u.fs_d.value;
            }
            else {
                return err;
            }

            // Angle 1
            double angle1 = -30.0;
            err = PF_CHECKOUT_PARAM(in_data, SWING_ANGLE1, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
            if (!err) {
                angle1 = param_copy.u.fs_d.value;
            }
            else {
                return err;
            }

            // Angle 2
            double angle2 = 30.0;
            err = PF_CHECKOUT_PARAM(in_data, SWING_ANGLE2, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
            if (!err) {
                angle2 = param_copy.u.fs_d.value;
            }
            else {
                return err;
            }

            // Phase
            double phase = 0.0;
            err = PF_CHECKOUT_PARAM(in_data, SWING_PHASE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
            if (!err) {
                phase = param_copy.u.fs_d.value;
            }
            else {
                return err;
            }

            // Wave type
            A_long waveType = 0;
            err = PF_CHECKOUT_PARAM(in_data, SWING_WAVE_TYPE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy);
            if (!err) {
                waveType = param_copy.u.pd.value - 1;
            }
            else {
                return err;
            }

            // Convert current time to seconds
            double current_time = (double)in_data->current_time / (double)in_data->time_scale;

            // Calculate effective phase including time component
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

                                PF_Pixel8* inRow = (PF_Pixel8*)((char*)input_worldP->data +
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
            }
        }

        // Check in the input layer pixels
        if (input_worldP) {
            err = extra->cb->checkin_layer_pixels(in_data->effect_ref, SWING_INPUT);
            if (err) {
                return err;
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

