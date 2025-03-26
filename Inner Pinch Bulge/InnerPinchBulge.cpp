/*
    InnerPinchBulge.cpp

    This effect creates a pinch or bulge effect from a center point with adjustable radius and strength.
*/

#include "InnerPinchBulge.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>

static PF_Err
About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_SPRINTF(out_data->return_msg,
        "Inner Pinch/Bulge\r"
        "Creates a pinch or bulge effect from a center point.\r"
        "Created for After Effects.");
    return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    out_data->my_version = PF_VERSION(MAJOR_VERSION,
        MINOR_VERSION,
        BUG_VERSION,
        STAGE_VERSION,
        BUILD_VERSION);

    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
        PF_OutFlag_PIX_INDEPENDENT |
        PF_OutFlag_I_EXPAND_BUFFER;    // This flag is crucial

    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
        PF_OutFlag2_FLOAT_COLOR_AWARE |
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

    return PF_Err_NONE;
}

static PF_Err
ParamsSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err        err = PF_Err_NONE;
    PF_ParamDef    def;

    AEFX_CLR_STRUCT(def);

    // Add the Center parameter
    PF_ADD_POINT("Center",
        INNER_PINCH_BULGE_CENTER_X_DFLT,
        INNER_PINCH_BULGE_CENTER_Y_DFLT,
        0,
        CENTER_DISK_ID);

    // Add the "Strength" parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Strength",
        INNER_PINCH_BULGE_STRENGTH_MIN,
        INNER_PINCH_BULGE_STRENGTH_MAX,
        INNER_PINCH_BULGE_STRENGTH_MIN,
        INNER_PINCH_BULGE_STRENGTH_MAX,
        INNER_PINCH_BULGE_STRENGTH_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        STRENGTH_DISK_ID);

    // Add the "Radius" parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Radius",
        INNER_PINCH_BULGE_RADIUS_MIN,
        INNER_PINCH_BULGE_RADIUS_MAX,
        INNER_PINCH_BULGE_RADIUS_MIN,
        INNER_PINCH_BULGE_RADIUS_MAX,
        INNER_PINCH_BULGE_RADIUS_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        RADIUS_DISK_ID);

    // Add the "Feather" parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Feather",
        INNER_PINCH_BULGE_FEATHER_MIN,
        INNER_PINCH_BULGE_FEATHER_MAX,
        INNER_PINCH_BULGE_FEATHER_MIN,
        INNER_PINCH_BULGE_FEATHER_MAX,
        INNER_PINCH_BULGE_FEATHER_DFLT,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        FEATHER_DISK_ID);

    // Add the "Use Gaussian" checkbox
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Use Gaussian",
        "",
        FALSE,
        0,
        GAUSSIAN_DISK_ID);

    out_data->num_params = INNER_PINCH_BULGE_NUM_PARAMS;

    return err;
}

// Helper function to implement mix operation
static float mix(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

// Helper function to implement smoothstep operation
static float smoothstep(float edge0, float edge1, float x) {
    float t = MAX(0.0f, MIN(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

// Helper function for Gaussian smoothing
static float gsmooth(float low, float high, float value) {
    float p = MIN(MAX((value - low) / (high - low), 0.0f), 1.0f);
    p -= 1.0f;
    return exp(-((p * p) / 0.125f));
}

// First, let's fix the SampleBilinear function to handle horizontal and vertical bounds differently
template <typename PixelType>
static void
SampleBilinear(
    PF_EffectWorld* input,
    float x,
    float y,
    PixelType* outP)
{
    // Check if x is out of bounds - if so, make transparent
    // But keep y visible even when out of bounds
    if (x < 0 || x >= input->width) {
        outP->alpha = 0;
        outP->red = 0;
        outP->green = 0;
        outP->blue = 0;
        return;
    }

    // Get integer and fractional parts
    int x1 = (int)x;
    int y1 = (int)y;
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float fx = x - x1;
    float fy = y - y1;

    // Clamp y coordinates to valid range (but not x, which should be transparent)
    y1 = MIN(MAX(y1, 0), input->height - 1);
    y2 = MIN(MAX(y2, 0), input->height - 1);

    // Clamp x only for accessing the buffer (not for transparency check)
    int x1_clamped = MIN(MAX(x1, 0), input->width - 1);
    int x2_clamped = MIN(MAX(x2, 0), input->width - 1);

    // Get pointers to pixels
    PixelType* p11, * p12, * p21, * p22;

    // Determine pixel depth by examining the world
    double bytesPerPixel = (double)input->rowbytes / (double)input->width;

    if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
        PF_PixelFloat* base = reinterpret_cast<PF_PixelFloat*>(input->data);
        p11 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_PixelFloat) + x1_clamped]);
        p12 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_PixelFloat) + x1_clamped]);
        p21 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_PixelFloat) + x2_clamped]);
        p22 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_PixelFloat) + x2_clamped]);
    }
    else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
        PF_Pixel16* base = reinterpret_cast<PF_Pixel16*>(input->data);
        p11 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_Pixel16) + x1_clamped]);
        p12 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_Pixel16) + x1_clamped]);
        p21 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_Pixel16) + x2_clamped]);
        p22 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_Pixel16) + x2_clamped]);
    }
    else { // 8-bit (default)
        PF_Pixel8* base = reinterpret_cast<PF_Pixel8*>(input->data);
        p11 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_Pixel8) + x1_clamped]);
        p12 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_Pixel8) + x1_clamped]);
        p21 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_Pixel8) + x2_clamped]);
        p22 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_Pixel8) + x2_clamped]);
    }

    // Check edge conditions for x
    bool x1_valid = (x1 >= 0 && x1 < input->width);
    bool x2_valid = (x2 >= 0 && x2 < input->width);

    // Interpolate with zero alpha for out-of-bounds x values
    float s1, s2;

    // Alpha
    s1 = x1_valid ? (1 - fx) * p11->alpha : 0;
    s1 += x2_valid ? fx * p21->alpha : 0;

    s2 = x1_valid ? (1 - fx) * p12->alpha : 0;
    s2 += x2_valid ? fx * p22->alpha : 0;

    outP->alpha = (1 - fy) * s1 + fy * s2;

    // Only process color if we have any alpha
    if (outP->alpha > 0) {
        // Red
        s1 = x1_valid ? (1 - fx) * p11->red : 0;
        s1 += x2_valid ? fx * p21->red : 0;

        s2 = x1_valid ? (1 - fx) * p12->red : 0;
        s2 += x2_valid ? fx * p22->red : 0;

        outP->red = (1 - fy) * s1 + fy * s2;

        // Green
        s1 = x1_valid ? (1 - fx) * p11->green : 0;
        s1 += x2_valid ? fx * p21->green : 0;

        s2 = x1_valid ? (1 - fx) * p12->green : 0;
        s2 += x2_valid ? fx * p22->green : 0;

        outP->green = (1 - fy) * s1 + fy * s2;

        // Blue
        s1 = x1_valid ? (1 - fx) * p11->blue : 0;
        s1 += x2_valid ? fx * p21->blue : 0;

        s2 = x1_valid ? (1 - fx) * p12->blue : 0;
        s2 += x2_valid ? fx * p22->blue : 0;

        outP->blue = (1 - fy) * s1 + fy * s2;
    }
    else {
        // Zero out color channels if alpha is zero
        outP->red = 0;
        outP->green = 0;
        outP->blue = 0;
    }
}


static PF_Err
InnerPinchFunc8(
    void* refcon,
    A_long        xL,
    A_long        yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    PF_Err        err = PF_Err_NONE;
    PinchBulgeInfo* piP = reinterpret_cast<PinchBulgeInfo*>(refcon);

    if (piP) {
        // Get input layer dimensions
        float width = (float)piP->input->width;
        float height = (float)piP->input->height;

        // Calculate center position based on user parameter
        // Use a smaller scale factor (10000.0f instead of 1000.0f) to reduce sensitivity
        float centerX = width / 2.0f + (PF_FpLong)piP->center_x / 10000.0f;
        float centerY = height / 2.0f - (PF_FpLong)piP->center_y / 10000.0f;

        // Calculate normalized UV coordinates
        float uvX = (float)xL / width;
        float uvY = (float)yL / height;

        // Calculate center in normalized coordinates
        float centerNormX = centerX / width;
        float centerNormY = centerY / height;

        // Calculate edge feathering
        float featherAmount = piP->feather;
        float damping = 1.0f;

        if (featherAmount > 0.0f) {
            float edgeX1 = smoothstep(0.0f, featherAmount, uvX);
            float edgeX2 = smoothstep(1.0f, 1.0f - featherAmount, uvX);
            float edgeY1 = smoothstep(0.0f, featherAmount, uvY);
            float edgeY2 = smoothstep(1.0f, 1.0f - featherAmount, uvY);
            damping = edgeX1 * edgeX2 * edgeY1 * edgeY2;
        }

        // Offset UV by center
        float offsetX = uvX - centerNormX;
        float offsetY = uvY - centerNormY;

        // Adjust for aspect ratio (multiply X by aspect ratio)
        float aspectRatio = width / height;
        offsetX *= aspectRatio;

        // Calculate distance from center
        float dist = sqrt(offsetX * offsetX + offsetY * offsetY);

        // Store original coordinates
        float prevOffsetX = offsetX;
        float prevOffsetY = offsetY;

        // Apply distortion if within radius
        if (dist < piP->radius) {
            float p = dist / piP->radius;

            if (piP->strength > 0.0f) {
                // Bulge effect
                if (piP->useGaussian) {
                    // Gaussian smoothing
                    float factor = mix(1.0f, gsmooth(0.0f, piP->radius / dist, p), piP->strength * 0.75f);
                    offsetX *= factor;
                    offsetY *= factor;
                }
                else {
                    // Regular smoothstep
                    float factor = mix(1.0f, smoothstep(0.0f, piP->radius / dist, p), piP->strength * 0.75f);
                    offsetX *= factor;
                    offsetY *= factor;
                }
            }
            else {
                // Pinch effect
                float factor = mix(1.0f, pow(p, 1.0f + piP->strength * 0.75f) * piP->radius / dist, 1.0f - p);
                offsetX *= factor;
                offsetY *= factor;
            }

            // Apply edge feathering
            offsetX = prevOffsetX * (1.0f - damping) + offsetX * damping;
            offsetY = prevOffsetY * (1.0f - damping) + offsetY * damping;
        }

        // Undo aspect ratio adjustment
        offsetX /= aspectRatio;

        // Calculate final sampling coordinates
        float srcX = (offsetX + centerNormX) * width;
        float srcY = (offsetY + centerNormY) * height;

        // Sample the input at the calculated position
        SampleBilinear<PF_Pixel8>(piP->input, srcX, srcY, outP);
    }
    else {
        *outP = *inP;
    }

    return err;
}


static PF_Err
InnerPinchFunc16(
    void* refcon,
    A_long        xL,
    A_long        yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    PF_Err        err = PF_Err_NONE;
    PinchBulgeInfo* piP = reinterpret_cast<PinchBulgeInfo*>(refcon);

    if (piP) {
        // Get input layer dimensions
        float width = (float)piP->input->width;
        float height = (float)piP->input->height;

        // Calculate center position based on user parameter
        // Use a smaller scale factor (10000.0f instead of 1000.0f) to reduce sensitivity
        float centerX = width / 2.0f + (PF_FpLong)piP->center_x / 10000.0f;
        float centerY = height / 2.0f - (PF_FpLong)piP->center_y / 10000.0f;

        // Calculate normalized UV coordinates
        float uvX = (float)xL / width;
        float uvY = (float)yL / height;

        // Calculate center in normalized coordinates
        float centerNormX = centerX / width;
        float centerNormY = centerY / height;

        // Calculate edge feathering
        float featherAmount = piP->feather;
        float damping = 1.0f;

        if (featherAmount > 0.0f) {
            float edgeX1 = smoothstep(0.0f, featherAmount, uvX);
            float edgeX2 = smoothstep(1.0f, 1.0f - featherAmount, uvX);
            float edgeY1 = smoothstep(0.0f, featherAmount, uvY);
            float edgeY2 = smoothstep(1.0f, 1.0f - featherAmount, uvY);
            damping = edgeX1 * edgeX2 * edgeY1 * edgeY2;
        }

        // Offset UV by center
        float offsetX = uvX - centerNormX;
        float offsetY = uvY - centerNormY;

        // Adjust for aspect ratio (multiply X by aspect ratio)
        float aspectRatio = width / height;
        offsetX *= aspectRatio;

        // Calculate distance from center
        float dist = sqrt(offsetX * offsetX + offsetY * offsetY);

        // Store original coordinates
        float prevOffsetX = offsetX;
        float prevOffsetY = offsetY;

        // Apply distortion if within radius
        if (dist < piP->radius) {
            float p = dist / piP->radius;

            if (piP->strength > 0.0f) {
                // Bulge effect
                if (piP->useGaussian) {
                    // Gaussian smoothing
                    float factor = mix(1.0f, gsmooth(0.0f, piP->radius / dist, p), piP->strength * 0.75f);
                    offsetX *= factor;
                    offsetY *= factor;
                }
                else {
                    // Regular smoothstep
                    float factor = mix(1.0f, smoothstep(0.0f, piP->radius / dist, p), piP->strength * 0.75f);
                    offsetX *= factor;
                    offsetY *= factor;
                }
            }
            else {
                // Pinch effect
                float factor = mix(1.0f, pow(p, 1.0f + piP->strength * 0.75f) * piP->radius / dist, 1.0f - p);
                offsetX *= factor;
                offsetY *= factor;
            }

            // Apply edge feathering
            offsetX = prevOffsetX * (1.0f - damping) + offsetX * damping;
            offsetY = prevOffsetY * (1.0f - damping) + offsetY * damping;
        }

        // Undo aspect ratio adjustment
        offsetX /= aspectRatio;

        // Calculate final sampling coordinates
        float srcX = (offsetX + centerNormX) * width;
        float srcY = (offsetY + centerNormY) * height;

        // Sample the input at the calculated position
        SampleBilinear<PF_Pixel16>(piP->input, srcX, srcY, outP);
    }
    else {
        *outP = *inP;
    }

    return err;
}

static PF_Err
InnerPinchFuncFloat(
    void* refcon,
    A_long        xL,
    A_long        yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    PF_Err        err = PF_Err_NONE;
    PinchBulgeInfo* piP = reinterpret_cast<PinchBulgeInfo*>(refcon);

    if (piP) {
        // Get input layer dimensions
        float width = (float)piP->input->width;
        float height = (float)piP->input->height;

        // Calculate center position based on user parameter
        // Use a smaller scale factor (10000.0f instead of 1000.0f) to reduce sensitivity
        float centerX = width / 2.0f + (PF_FpLong)piP->center_x / 10000.0f;
        float centerY = height / 2.0f - (PF_FpLong)piP->center_y / 10000.0f;

        // Calculate normalized UV coordinates
        float uvX = (float)xL / width;
        float uvY = (float)yL / height;

        // Calculate center in normalized coordinates
        float centerNormX = centerX / width;
        float centerNormY = centerY / height;

        // Calculate edge feathering
        float featherAmount = piP->feather;
        float damping = 1.0f;

        if (featherAmount > 0.0f) {
            float edgeX1 = smoothstep(0.0f, featherAmount, uvX);
            float edgeX2 = smoothstep(1.0f, 1.0f - featherAmount, uvX);
            float edgeY1 = smoothstep(0.0f, featherAmount, uvY);
            float edgeY2 = smoothstep(1.0f, 1.0f - featherAmount, uvY);
            damping = edgeX1 * edgeX2 * edgeY1 * edgeY2;
        }

        // Offset UV by center
        float offsetX = uvX - centerNormX;
        float offsetY = uvY - centerNormY;

        // Adjust for aspect ratio (multiply X by aspect ratio)
        float aspectRatio = width / height;
        offsetX *= aspectRatio;

        // Calculate distance from center
        float dist = sqrt(offsetX * offsetX + offsetY * offsetY);

        // Store original coordinates
        float prevOffsetX = offsetX;
        float prevOffsetY = offsetY;

        // Apply distortion if within radius
        if (dist < piP->radius) {
            float p = dist / piP->radius;

            if (piP->strength > 0.0f) {
                // Bulge effect
                if (piP->useGaussian) {
                    // Gaussian smoothing
                    float factor = mix(1.0f, gsmooth(0.0f, piP->radius / dist, p), piP->strength * 0.75f);
                    offsetX *= factor;
                    offsetY *= factor;
                }
                else {
                    // Regular smoothstep
                    float factor = mix(1.0f, smoothstep(0.0f, piP->radius / dist, p), piP->strength * 0.75f);
                    offsetX *= factor;
                    offsetY *= factor;
                }
            }
            else {
                // Pinch effect
                float factor = mix(1.0f, pow(p, 1.0f + piP->strength * 0.75f) * piP->radius / dist, 1.0f - p);
                offsetX *= factor;
                offsetY *= factor;
            }

            // Apply edge feathering
            offsetX = prevOffsetX * (1.0f - damping) + offsetX * damping;
            offsetY = prevOffsetY * (1.0f - damping) + offsetY * damping;
        }

        // Undo aspect ratio adjustment
        offsetX /= aspectRatio;

        // Calculate final sampling coordinates
        float srcX = (offsetX + centerNormX) * width;
        float srcY = (offsetY + centerNormY) * height;

        // Sample the input at the calculated position
        SampleBilinear<PF_PixelFloat>(piP->input, srcX, srcY, outP);
    }
    else {
        *outP = *inP;
    }

    return err;
}


static PF_Err
SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    // Initialize effect info structure
    PinchBulgeInfo info;
    AEFX_CLR_STRUCT(info);

    PF_ParamDef param_copy;
    AEFX_CLR_STRUCT(param_copy);

    // Initialize max_result_rect to the current output request rect
    extra->output->max_result_rect = extra->input->output_request.rect;

    // Get center parameter
    ERR(PF_CHECKOUT_PARAM(in_data, INNER_PINCH_BULGE_CENTER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) {
        info.center_x = param_copy.u.td.x_value;
        info.center_y = param_copy.u.td.y_value;
    }

    // Get strength parameter
    ERR(PF_CHECKOUT_PARAM(in_data, INNER_PINCH_BULGE_STRENGTH, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) info.strength = param_copy.u.fs_d.value;

    // Get radius parameter
    ERR(PF_CHECKOUT_PARAM(in_data, INNER_PINCH_BULGE_RADIUS, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) info.radius = param_copy.u.fs_d.value;

    // Calculate buffer expansion based on radius and strength
    A_long expansion = 0;
    expansion = ceil(fabs(info.strength) * info.radius * 100.0f);

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
        INNER_PINCH_BULGE_INPUT,
        INNER_PINCH_BULGE_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &checkout));

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

    return err;
}

static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Checkout input layer pixels
    PF_EffectWorld* input_worldP = NULL;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, INNER_PINCH_BULGE_INPUT, &input_worldP));

    if (!err && input_worldP) {
        // Checkout output buffer
        PF_EffectWorld* output_worldP = NULL;
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (!err && output_worldP) {
            PF_ParamDef param_copy;
            AEFX_CLR_STRUCT(param_copy);

            PinchBulgeInfo info;
            AEFX_CLR_STRUCT(info);

            // Get center parameter
            ERR(PF_CHECKOUT_PARAM(in_data, INNER_PINCH_BULGE_CENTER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
            if (!err) {
                info.center_x = param_copy.u.td.x_value;
                info.center_y = param_copy.u.td.y_value;
            }

            // Get strength parameter
            ERR(PF_CHECKOUT_PARAM(in_data, INNER_PINCH_BULGE_STRENGTH, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
            if (!err) info.strength = param_copy.u.fs_d.value;

            // Get radius parameter
            ERR(PF_CHECKOUT_PARAM(in_data, INNER_PINCH_BULGE_RADIUS, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
            if (!err) info.radius = param_copy.u.fs_d.value;

            // Get feather parameter
            ERR(PF_CHECKOUT_PARAM(in_data, INNER_PINCH_BULGE_FEATHER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
            if (!err) info.feather = param_copy.u.fs_d.value;

            // Get gaussian parameter
            ERR(PF_CHECKOUT_PARAM(in_data, INNER_PINCH_BULGE_GAUSSIAN, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
            if (!err) info.useGaussian = param_copy.u.bd.value;

            info.input = input_worldP;

            // Clear output buffer
            PF_Pixel empty_pixel = { 0, 0, 0, 0 };
            ERR(suites.FillMatteSuite2()->fill(
                in_data->effect_ref,
                &empty_pixel,
                NULL,
                output_worldP));

            if (!err) {
                // Determine pixel depth by examining the world
                double bytesPerPixel = (double)input_worldP->rowbytes / (double)input_worldP->width;
                bool is16bit = false;
                bool is32bit = false;

                if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
                    is32bit = true;
                }
                else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
                    is16bit = true;
                }

                if (is32bit) {
                    // Use the IterateFloatSuite for 32-bit processing
                    ERR(suites.IterateFloatSuite1()->iterate(
                        in_data,
                        0,                // progress base
                        output_worldP->height,  // progress final
                        input_worldP,     // src
                        NULL,             // area - null for all pixels
                        (void*)&info,     // refcon - custom data
                        InnerPinchFuncFloat,   // pixel function pointer
                        output_worldP));
                }
                else if (is16bit) {
                    // Process with 16-bit iterate suite
                    ERR(suites.Iterate16Suite2()->iterate(
                        in_data,
                        0,                // progress base
                        output_worldP->height,  // progress final
                        input_worldP,     // src
                        NULL,             // area - null for all pixels
                        (void*)&info,     // refcon - custom data
                        InnerPinchFunc16,      // pixel function pointer
                        output_worldP));
                }
                else {
                    // Process with 8-bit iterate suite
                    ERR(suites.Iterate8Suite2()->iterate(
                        in_data,
                        0,                // progress base
                        output_worldP->height,  // progress final
                        input_worldP,     // src
                        NULL,             // area - null for all pixels
                        (void*)&info,     // refcon - custom data
                        InnerPinchFunc8,       // pixel function pointer
                        output_worldP));
                }
            }
        }
    }

    // Check in the input layer pixels
    if (input_worldP) {
        ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, INNER_PINCH_BULGE_INPUT));
    }

    return err;
}

static PF_Err
Render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err                err = PF_Err_NONE;
    AEGP_SuiteHandler    suites(in_data->pica_basicP);

    PinchBulgeInfo            piP;
    AEFX_CLR_STRUCT(piP);

    // Get parameter values
    piP.center_x = params[INNER_PINCH_BULGE_CENTER]->u.td.x_value;
    piP.center_y = params[INNER_PINCH_BULGE_CENTER]->u.td.y_value;
    piP.strength = params[INNER_PINCH_BULGE_STRENGTH]->u.fs_d.value;
    piP.radius = params[INNER_PINCH_BULGE_RADIUS]->u.fs_d.value;
    piP.feather = params[INNER_PINCH_BULGE_FEATHER]->u.fs_d.value;
    piP.useGaussian = params[INNER_PINCH_BULGE_GAUSSIAN]->u.bd.value;
    piP.input = &params[INNER_PINCH_BULGE_INPUT]->u.ld;

    // Determine pixel depth by examining the world
    double bytesPerPixel = (double)piP.input->rowbytes / (double)piP.input->width;
    bool is16bit = false;
    bool is32bit = false;

    if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
        is32bit = true;
    }
    else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
        is16bit = true;
    }

    if (is32bit) {
        // Use the IterateFloatSuite for 32-bit processing
        ERR(suites.IterateFloatSuite1()->iterate(
            in_data,
            0,                                // progress base
            output->height,                   // progress final
            &params[INNER_PINCH_BULGE_INPUT]->u.ld,  // src 
            NULL,                            // area - null for all pixels
            (void*)&piP,                    // refcon - custom data
            InnerPinchFuncFloat,             // pixel function pointer
            output));
    }
    else if (is16bit) {
        // Process with 16-bit iterate suite
        ERR(suites.Iterate16Suite2()->iterate(
            in_data,
            0,                                // progress base
            output->height,                   // progress final
            &params[INNER_PINCH_BULGE_INPUT]->u.ld,  // src 
            NULL,                            // area - null for all pixels
            (void*)&piP,                    // refcon - custom data
            InnerPinchFunc16,                // pixel function pointer
            output));
    }
    else {
        // Process with 8-bit iterate suite
        ERR(suites.Iterate8Suite2()->iterate(
            in_data,
            0,                                // progress base
            output->height,                   // progress final
            &params[INNER_PINCH_BULGE_INPUT]->u.ld,  // src 
            NULL,                            // area - null for all pixels
            (void*)&piP,                    // refcon - custom data
            InnerPinchFunc8,                  // pixel function pointer
            output));
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
        "Inner Pinch/Bulge",                    // Name
        "DKT Inner Pinch Bulge",                // Match Name
        "DKT Effects",                // Category
        AE_RESERVED_INFO,                // Reserved Info
        "EffectMain",                    // Entry point
        "");        // support URL

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
    PF_Err        err = PF_Err_NONE;

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
            err = Render(in_data, out_data, params, output);
            break;
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    return err;
}

