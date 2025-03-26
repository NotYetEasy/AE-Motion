#include "StretchAxis.h"
#include <stdio.h>
#include <math.h>

// String definitions directly in the code
#define STR_NAME "Stretch Axis"
#define STR_DESCRIPTION "A plugin that stretches your layers along an angle."
#define STR_SCALE_PARAM_NAME "Scale"
#define STR_ANGLE_PARAM_NAME "Angle"
#define STR_CONTENT_ONLY_PARAM_NAME "Mask to Layer"

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

static PF_Err
About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
        "%s v%d.%d\r%s",
        STR_NAME,
        MAJOR_VERSION,
        MINOR_VERSION,
        STR_DESCRIPTION);
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

    // Added more flags like in your working plugins
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
        PF_OutFlag_PIX_INDEPENDENT |
        PF_OutFlag_I_EXPAND_BUFFER;

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
    PF_Err		err = PF_Err_NONE;
    PF_ParamDef	def;

    AEFX_CLR_STRUCT(def);

    // Scale parameter
    PF_ADD_FLOAT_SLIDERX(STR_SCALE_PARAM_NAME,
        STRETCH_SCALE_MIN,
        STRETCH_SCALE_MAX,
        STRETCH_SCALE_MIN,
        STRETCH_SCALE_MAX,
        STRETCH_SCALE_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        SCALE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Angle parameter - now as a slider instead of an angle control
    PF_ADD_FLOAT_SLIDERX(STR_ANGLE_PARAM_NAME,
        0,
        3600,
        0,
        180,
        0,
        PF_Precision_TENTHS,
        0,
        0,
        ANGLE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Content Only parameter
    PF_ADD_CHECKBOX(STR_CONTENT_ONLY_PARAM_NAME,
        "",
        FALSE,
        0,
        CONTENT_ONLY_DISK_ID);

    out_data->num_params = STRETCH_NUM_PARAMS;

    return err;
}

// Function to sample pixel with bilinear interpolation
template <typename PixelType>
static void
SampleBilinear(
    void* src_data,
    A_long rowbytes,
    A_long width,
    A_long height,
    float x,
    float y,
    PixelType* outP)
{
    // Check if x is out of bounds - if so, make transparent
    if (x < 0 || x >= width) {
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
    y1 = MIN(MAX(y1, 0), height - 1);
    y2 = MIN(MAX(y2, 0), height - 1);

    // Clamp x only for accessing the buffer (not for transparency check)
    int x1_clamped = MIN(MAX(x1, 0), width - 1);
    int x2_clamped = MIN(MAX(x2, 0), width - 1);

    // Get pointers to pixels
    PixelType* p11, * p12, * p21, * p22;

    // Compute the size of a single pixel based on the pixel type
    size_t pixelSize = sizeof(PixelType);

    // Get base pointer
    PixelType* base = reinterpret_cast<PixelType*>(src_data);

    // Calculate pointers to the four pixels for bilinear interpolation
    p11 = reinterpret_cast<PixelType*>((char*)base + (y1 * rowbytes)) + x1_clamped;
    p12 = reinterpret_cast<PixelType*>((char*)base + (y2 * rowbytes)) + x1_clamped;
    p21 = reinterpret_cast<PixelType*>((char*)base + (y1 * rowbytes)) + x2_clamped;
    p22 = reinterpret_cast<PixelType*>((char*)base + (y2 * rowbytes)) + x2_clamped;

    // Check edge conditions for x
    bool x1_valid = (x1 >= 0 && x1 < width);
    bool x2_valid = (x2 >= 0 && x2 < width);

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
StretchFunc8(
    void* refcon,
    A_long		xL,
    A_long		yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    PF_Err			err = PF_Err_NONE;
    StretchInfo* siP = reinterpret_cast<StretchInfo*>(refcon);

    if (!siP) {
        *outP = *inP;  // Fallback to input pixel
        return PF_Err_BAD_CALLBACK_PARAM;
    }

    // Get dimensions and parameters
    float width = (float)siP->width;
    float height = (float)siP->height;
    float scale = (float)siP->scale;
    float angle = (float)siP->angle;
    bool content_only = siP->content_only;

    // Calculate normalized UV coordinates
    float uvX = (float)xL / width;
    float uvY = (float)yL / height;

    // Center coordinates (0.5, 0.5 is center)
    float centerX = 0.5f;
    float centerY = 0.5f;

    // Offset UV by center
    float offsetX = uvX - centerX;
    float offsetY = uvY - centerY;

    // Convert angle to radians
    const float PI = 3.14159265358979323846f;
    float rad = -angle * PI / 180.0f;

    // Rotation matrices
    float cos_rad = cosf(rad);
    float sin_rad = sinf(rad);

    // Apply rotation
    float x_rot = offsetX * cos_rad - offsetY * sin_rad;
    float y_rot = offsetX * sin_rad + offsetY * cos_rad;

    // Apply scale (only to x-coordinate)
    if (scale != 0.0f) {
        x_rot = x_rot / scale;
    }

    // Inverse rotation
    float x_inv_rot = x_rot * cos_rad + y_rot * sin_rad;
    float y_inv_rot = -x_rot * sin_rad + y_rot * cos_rad;

    // Back to absolute coordinates
    float sampleX = (x_inv_rot + centerX) * width;
    float sampleY = (y_inv_rot + centerY) * height;

    // Using the bilinear sampling function
    if (content_only && inP->alpha == 0) {
        // If content_only is true and current pixel has no alpha, use input
        *outP = *inP;
    }
    else {
        // Sample the source pixel with bilinear interpolation
        SampleBilinear<PF_Pixel8>(siP->src, siP->rowbytes, siP->width, siP->height, sampleX, sampleY, outP);

        if (content_only) {
            // If content_only is true, preserve input alpha
            outP->alpha = inP->alpha;
        }
    }

    return err;
}

static PF_Err
StretchFunc16(
    void* refcon,
    A_long		xL,
    A_long		yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    PF_Err			err = PF_Err_NONE;
    StretchInfo* siP = reinterpret_cast<StretchInfo*>(refcon);

    if (!siP) {
        *outP = *inP;  // Fallback to input pixel
        return PF_Err_BAD_CALLBACK_PARAM;
    }

    // Get dimensions and parameters
    float width = (float)siP->width;
    float height = (float)siP->height;
    float scale = (float)siP->scale;
    float angle = (float)siP->angle;
    bool content_only = siP->content_only;

    // Calculate normalized UV coordinates
    float uvX = (float)xL / width;
    float uvY = (float)yL / height;

    // Center coordinates (0.5, 0.5 is center)
    float centerX = 0.5f;
    float centerY = 0.5f;

    // Offset UV by center
    float offsetX = uvX - centerX;
    float offsetY = uvY - centerY;

    // Convert angle to radians
    const float PI = 3.14159265358979323846f;
    float rad = -angle * PI / 180.0f;

    // Rotation matrices
    float cos_rad = cosf(rad);
    float sin_rad = sinf(rad);

    // Apply rotation
    float x_rot = offsetX * cos_rad - offsetY * sin_rad;
    float y_rot = offsetX * sin_rad + offsetY * cos_rad;

    // Apply scale (only to x-coordinate)
    if (scale != 0.0f) {
        x_rot = x_rot / scale;
    }

    // Inverse rotation
    float x_inv_rot = x_rot * cos_rad + y_rot * sin_rad;
    float y_inv_rot = -x_rot * sin_rad + y_rot * cos_rad;

    // Back to absolute coordinates
    float sampleX = (x_inv_rot + centerX) * width;
    float sampleY = (y_inv_rot + centerY) * height;

    // Using the bilinear sampling function
    if (content_only && inP->alpha == 0) {
        // If content_only is true and current pixel has no alpha, use input
        *outP = *inP;
    }
    else {
        // Sample the source pixel with bilinear interpolation
        SampleBilinear<PF_Pixel16>(siP->src, siP->rowbytes, siP->width, siP->height, sampleX, sampleY, outP);

        if (content_only) {
            // If content_only is true, preserve input alpha
            outP->alpha = inP->alpha;
        }
    }

    return err;
}

static PF_Err
StretchFuncFloat(
    void* refcon,
    A_long		xL,
    A_long		yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    PF_Err			err = PF_Err_NONE;
    StretchInfo* siP = reinterpret_cast<StretchInfo*>(refcon);

    if (!siP) {
        *outP = *inP;  // Fallback to input pixel
        return PF_Err_BAD_CALLBACK_PARAM;
    }

    // Get dimensions and parameters
    float width = (float)siP->width;
    float height = (float)siP->height;
    float scale = (float)siP->scale;
    float angle = (float)siP->angle;
    bool content_only = siP->content_only;

    // Calculate normalized UV coordinates
    float uvX = (float)xL / width;
    float uvY = (float)yL / height;

    // Center coordinates (0.5, 0.5 is center)
    float centerX = 0.5f;
    float centerY = 0.5f;

    // Offset UV by center
    float offsetX = uvX - centerX;
    float offsetY = uvY - centerY;

    // Convert angle to radians
    const float PI = 3.14159265358979323846f;
    float rad = -angle * PI / 180.0f;

    // Rotation matrices
    float cos_rad = cosf(rad);
    float sin_rad = sinf(rad);

    // Apply rotation
    float x_rot = offsetX * cos_rad - offsetY * sin_rad;
    float y_rot = offsetX * sin_rad + offsetY * cos_rad;

    // Apply scale (only to x-coordinate)
    if (scale != 0.0f) {
        x_rot = x_rot / scale;
    }

    // Inverse rotation
    float x_inv_rot = x_rot * cos_rad + y_rot * sin_rad;
    float y_inv_rot = -x_rot * sin_rad + y_rot * cos_rad;

    // Back to absolute coordinates
    float sampleX = (x_inv_rot + centerX) * width;
    float sampleY = (y_inv_rot + centerY) * height;

    // Using the bilinear sampling function
    if (content_only && inP->alpha == 0) {
        // If content_only is true and current pixel has no alpha, use input
        *outP = *inP;
    }
    else {
        // Sample the source pixel with bilinear interpolation
        SampleBilinear<PF_PixelFloat>(siP->src, siP->rowbytes, siP->width, siP->height, sampleX, sampleY, outP);

        if (content_only) {
            // If content_only is true, preserve input alpha
            outP->alpha = inP->alpha;
        }
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
    PF_Err				err = PF_Err_NONE;
    AEGP_SuiteHandler	suites(in_data->pica_basicP);

    StretchInfo			siP;
    AEFX_CLR_STRUCT(siP);

    // Get input layer dimensions
    siP.width = params[STRETCH_INPUT]->u.ld.width;
    siP.height = params[STRETCH_INPUT]->u.ld.height;

    // Get parameters
    siP.scale = params[STRETCH_SCALE]->u.fs_d.value;
    if (siP.scale <= 0.01) {
        siP.scale = STRETCH_SCALE_DFLT;
    }

    siP.angle = params[STRETCH_ANGLE]->u.fs_d.value; // Now using fs_d instead of ad since we changed to a slider
    siP.content_only = params[STRETCH_CONTENT_ONLY]->u.bd.value;

    A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

    // Get source pixels
    siP.src = params[STRETCH_INPUT]->u.ld.data;
    siP.rowbytes = params[STRETCH_INPUT]->u.ld.rowbytes;

    // Determine pixel depth by examining the world
    double bytesPerPixel = (double)siP.rowbytes / (double)siP.width;
    bool is16bit = false;
    bool is32bit = false;

    if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
        is32bit = true;
        ERR(suites.IterateFloatSuite1()->iterate(
            in_data,
            0,                              // progress base
            linesL,                         // progress final
            &params[STRETCH_INPUT]->u.ld,   // src 
            NULL,                           // area - null for all pixels
            (void*)&siP,                    // refcon - your custom data pointer
            StretchFuncFloat,               // pixel function pointer
            output));
    }
    else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
        is16bit = true;
        ERR(suites.Iterate16Suite2()->iterate(
            in_data,
            0,                              // progress base
            linesL,                         // progress final
            &params[STRETCH_INPUT]->u.ld,   // src 
            NULL,                           // area - null for all pixels
            (void*)&siP,                    // refcon - your custom data pointer
            StretchFunc16,                  // pixel function pointer
            output));
    }
    else {
        ERR(suites.Iterate8Suite2()->iterate(
            in_data,
            0,                              // progress base
            linesL,                         // progress final
            &params[STRETCH_INPUT]->u.ld,   // src 
            NULL,                           // area - null for all pixels
            (void*)&siP,                    // refcon - your custom data pointer
            StretchFunc8,                   // pixel function pointer
            output));
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

    // Initialize stretch info structure
    StretchInfo stretch;
    AEFX_CLR_STRUCT(stretch);

    PF_ParamDef param_copy;
    AEFX_CLR_STRUCT(param_copy);

    // Initialize max_result_rect to the current output request rect
    extra->output->max_result_rect = extra->input->output_request.rect;

    // Get parameters
    ERR(PF_CHECKOUT_PARAM(in_data, STRETCH_SCALE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) stretch.scale = param_copy.u.fs_d.value;

    ERR(PF_CHECKOUT_PARAM(in_data, STRETCH_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) stretch.angle = param_copy.u.fs_d.value;

    ERR(PF_CHECKOUT_PARAM(in_data, STRETCH_CONTENT_ONLY, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) stretch.content_only = param_copy.u.bd.value;

    // Calculate buffer expansion based on scale and angle
    A_long expansion = 0;
    expansion = (A_long)(fabs(1.0 - stretch.scale) * 100.0);

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
        STRETCH_INPUT,
        STRETCH_INPUT,
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
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, STRETCH_INPUT, &input_worldP));

    if (!err && input_worldP) {
        // Checkout output buffer
        PF_EffectWorld* output_worldP = NULL;
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (!err && output_worldP) {
            PF_ParamDef param_copy;
            AEFX_CLR_STRUCT(param_copy);

            StretchInfo stretch;
            AEFX_CLR_STRUCT(stretch);

            // Get parameters
            ERR(PF_CHECKOUT_PARAM(in_data, STRETCH_SCALE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
            if (!err) stretch.scale = param_copy.u.fs_d.value;

            ERR(PF_CHECKOUT_PARAM(in_data, STRETCH_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
            if (!err) stretch.angle = param_copy.u.fs_d.value;

            ERR(PF_CHECKOUT_PARAM(in_data, STRETCH_CONTENT_ONLY, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
            if (!err) stretch.content_only = param_copy.u.bd.value;

            // Set up stretch info
            stretch.width = input_worldP->width;
            stretch.height = input_worldP->height;
            stretch.src = input_worldP->data;
            stretch.rowbytes = input_worldP->rowbytes;
            stretch.input = input_worldP;

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
                    // Use the IterateFloatSuite for 32-bit processing
                    ERR(suites.IterateFloatSuite1()->iterate(
                        in_data,
                        0,                // progress base
                        output_worldP->height,  // progress final
                        input_worldP,     // src
                        NULL,             // area - null for all pixels
                        (void*)&stretch,   // refcon - custom data
                        StretchFuncFloat,  // pixel function pointer
                        output_worldP));
                }
                else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
                    is16bit = true;
                    // Process with 16-bit iterate suite
                    ERR(suites.Iterate16Suite2()->iterate(
                        in_data,
                        0,                // progress base
                        output_worldP->height,  // progress final
                        input_worldP,     // src
                        NULL,             // area - null for all pixels
                        (void*)&stretch,   // refcon - custom data
                        StretchFunc16,      // pixel function pointer
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
                        (void*)&stretch,   // refcon - custom data
                        StretchFunc8,       // pixel function pointer
                        output_worldP));
                }
            }
        }
    }

    // Check in the input layer pixels
    if (input_worldP) {
        ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, STRETCH_INPUT));
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
        "Stretch Axis", // Name
        "DKT Stretch Axis", // Match Name
        "DKT Effects", // Category
        AE_RESERVED_INFO, // Reserved Info
        "EffectMain",	// Entry point
        "");	// support URL

    return result;
}

PF_Err
EffectMain(
    PF_Cmd			cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err		err = PF_Err_NONE;

    try {
        switch (cmd) {
        case PF_Cmd_ABOUT:
            err = About(in_data,
                out_data,
                params,
                output);
            break;

        case PF_Cmd_GLOBAL_SETUP:
            err = GlobalSetup(in_data,
                out_data,
                params,
                output);
            break;

        case PF_Cmd_PARAMS_SETUP:
            err = ParamsSetup(in_data,
                out_data,
                params,
                output);
            break;

        case PF_Cmd_RENDER:
            err = Render(in_data,
                out_data,
                params,
                output);
            break;

        case PF_Cmd_SMART_PRE_RENDER:
            err = SmartPreRender(in_data,
                out_data,
                (PF_PreRenderExtra*)extra);
            break;

        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_data,
                out_data,
                (PF_SmartRenderExtra*)extra);
            break;

            // Handle other commands silently
        default:
            break;
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    catch (std::exception& e) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    return err;
}

