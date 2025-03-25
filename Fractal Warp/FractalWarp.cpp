/**
 * FractalWarp.cpp
 * After Effects plugin that warps layers using fractal noise patterns
 * Created by DKT
 */

#include "FractalWarp.h"
#include <cmath>
#include <mutex>

 // String definitions
const char* STR_EFFECT_NAME = "Fractal Warp";
const char* STR_EFFECT_DESCRIPTION = "Warps layers using fractal noise patterns";
const char* STR_POSITION_PARAM = "Position";
const char* STR_PARALLAX_PARAM = "Parallax";
const char* STR_MAGNITUDE_PARAM = "Magnitude";
const char* STR_DETAIL_PARAM = "Detail";
const char* STR_LACUNARITY_PARAM = "Lacunarity";
const char* STR_SCREENSPACE_PARAM = "Screen Space";
const char* STR_OCTAVES_PARAM = "Octaves";

#ifndef PF_WorldFlag_FLOAT
#define PF_WorldFlag_FLOAT 1
#endif

#ifndef PF_WORLD_IS_FLOAT
#define PF_WORLD_IS_FLOAT(W) ((W)->world_flags & PF_WorldFlag_FLOAT)
#endif

/**
 * About command handler - displays plugin information
 */
static PF_Err
About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
        "%s v%d.%d\r%s",
        STR_EFFECT_NAME,
        MAJOR_VERSION,
        MINOR_VERSION,
        STR_EFFECT_DESCRIPTION);
    return PF_Err_NONE;
}

/**
 * Global setup - registers plugin capabilities
 */
static PF_Err
GlobalSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    // Set version information - use the specific value that matches your PIPL
    out_data->my_version = 0x80000;

    // Set plugin flags - using the same flags as the Oscillate effect
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
static PF_Err
ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    PF_ParamDef def;

    // Clear parameter definition structure
    AEFX_CLR_STRUCT(def);

    // Position parameter
    PF_ADD_POINT(STR_POSITION_PARAM,
        0, 0,  // X and Y position
        0,     // Flags
        POSITION_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Parallax parameter
    PF_ADD_POINT(STR_PARALLAX_PARAM,
        0, 0,  // X and Y position
        0,     // Flags
        PARALLAX_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Magnitude parameter
    PF_ADD_FLOAT_SLIDERX(STR_MAGNITUDE_PARAM,
        -5,
        5,
        -0.05,
        0.05,
        0.2,
        PF_Precision_THOUSANDTHS,
        0,
        0,
        MAGNITUDE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Detail parameter
    PF_ADD_FLOAT_SLIDERX(STR_DETAIL_PARAM,
        FRACTALWARP_DETAIL_MIN,
        FRACTALWARP_DETAIL_MAX,
        FRACTALWARP_DETAIL_MIN,
        FRACTALWARP_DETAIL_MAX,
        FRACTALWARP_DETAIL_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        DETAIL_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Lacunarity parameter
    PF_ADD_FLOAT_SLIDERX(STR_LACUNARITY_PARAM,
        FRACTALWARP_LACUNARITY_MIN,
        FRACTALWARP_LACUNARITY_MAX,
        FRACTALWARP_LACUNARITY_MIN,
        FRACTALWARP_LACUNARITY_MAX,
        FRACTALWARP_LACUNARITY_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        LACUNARITY_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Screen Space checkbox
    PF_ADD_CHECKBOX(STR_SCREENSPACE_PARAM,
        "",
        FALSE,
        0,
        SCREENSPACE_DISK_ID);

    AEFX_CLR_STRUCT(def);

    // Octaves parameter
    PF_ADD_SLIDER(STR_OCTAVES_PARAM,
        FRACTALWARP_OCTAVES_MIN,
        FRACTALWARP_OCTAVES_MAX,
        FRACTALWARP_OCTAVES_MIN,
        FRACTALWARP_OCTAVES_MAX,
        FRACTALWARP_OCTAVES_DFLT,
        OCTAVES_DISK_ID);

    out_data->num_params = FRACTALWARP_NUM_PARAMS;

    return err;
}

// Define our own min/max functions to avoid std:: namespace issues
template <typename T>
inline T MinValue(T a, T b) {
    return (a < b) ? a : b;
}

template <typename T>
inline T MaxValue(T a, T b) {
    return (a > b) ? a : b;
}

// Helper function for linear interpolation - must be defined before using it
inline float Mix(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

/**
 * Helper function for fractional part
 */
static float
Fract(float x) {
    return x - floorf(x);
}

// Noise generation function
static float
Random(float x, float y) {
    return Fract(sinf(x * 12.9898f + y * 78.233f) * 43758.5453123f);
}

// Perlin-style noise function
static float
Noise(float x, float y) {
    float i = floorf(x);
    float j = floorf(y);
    float f = Fract(x);
    float g = Fract(y);

    // Four corners in 2D of a tile
    float a = Random(i, j);
    float b = Random(i + 1.0f, j);
    float c = Random(i, j + 1.0f);
    float d = Random(i + 1.0f, j + 1.0f);

    // Cubic Hermine Curve
    float u = f * f * (3.0f - 2.0f * f);
    float v = g * g * (3.0f - 2.0f * g);

    // Mix
    return Mix(Mix(a, b, u),
        Mix(c, d, u),
        v);
}

// Fractal Brownian Motion function
static float
FBM(float x, float y, float px, float py, int octaveCount, float intensity) {
    // Initial values
    float value = 0.0f;
    float amplitude = 0.5f;

    // Loop of octaves
    for (int i = 0; i < octaveCount; i++) {
        value += amplitude * Noise(x, y);


        x = x * 2.0f + px;
        y = y * 2.0f + py;

        amplitude *= intensity;
    }

    return value;
}


// Helper functions for min/max operations
inline float MinFloat(float a, float b) {
    return (a < b) ? a : b;
}

inline float MaxFloat(float a, float b) {
    return (a > b) ? a : b;
}

inline A_long MinLong(A_long a, A_long b) {
    return (a < b) ? a : b;
}

inline A_long MaxLong(A_long a, A_long b) {
    return (a > b) ? a : b;
}


static PF_Err
FractalWarpFunc8(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    PF_Err err = PF_Err_NONE;

    FractalWarpInfo* fwi = reinterpret_cast<FractalWarpInfo*>(refcon);

    if (fwi) {
        const float width = static_cast<float>(fwi->width);
        const float height = static_cast<float>(fwi->height);
        const float aspectRatio = width / height;

        // Normalize coordinates (0-1)
        float st_x, st_y;

        // Setup coordinate system
        if (fwi->screenSpace) {
            st_x = static_cast<float>(xL) / width;
            st_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
            st_x *= aspectRatio;  // Apply aspect ratio correction
        }
        else {
            st_x = static_cast<float>(xL) / width;
            st_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
            st_x *= aspectRatio;  // Apply aspect ratio correction
        }

        // Apply position offset
        st_x += (fwi->position.x / 1000.0f) * -1.0f;
        st_y += (fwi->position.y / 1000.0f) * -1.0f;  // Reversed sign

        // Calculate parallax values
        const float parallax_x = (fwi->parallax.x / 200.0f) * -1.0f;
        const float parallax_y = (fwi->parallax.y / 200.0f) * -1.0f;  // Reversed sign

        // Calculate displacement using FBM
        const float dx = FBM(
            ((st_x - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            ((st_y - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            parallax_x, parallax_y,
            fwi->octaves, fwi->lacunarity
        );

        const float dy = FBM(
            ((st_x + 25.3f - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            ((st_y + 12.9f - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            parallax_x, parallax_y,
            fwi->octaves, fwi->lacunarity
        );

        // Sample coordinates
        float sample_x, sample_y;
        if (fwi->screenSpace) {
            sample_x = static_cast<float>(xL) / width;
            sample_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
        }
        else {
            sample_x = static_cast<float>(xL) / width;
            sample_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
        }

        // Apply displacement
        sample_x += (dx - 0.5f) * fwi->magnitude;
        sample_y += (dy - 0.5f) * fwi->magnitude;

        // Boundary handling
        // If coordinates are outside the range [0,1], set to transparent
        if (sample_x < 0.0f || sample_x > 1.0f || sample_y < 0.0f || sample_y > 1.0f) {
            // Set to transparent or black pixel
            outP->alpha = 0;
            outP->red = outP->green = outP->blue = 0;
            return err;
        }

        // Convert to pixel coordinates
        A_long sampleXL = static_cast<A_long>(sample_x * width + 0.5f);
        A_long sampleYL = static_cast<A_long>((1.0f - sample_y) * height + 0.5f);  // Reversed Y back to screen coordinates

        // Ensure sample coordinates are within valid range
        sampleXL = MaxLong(0L, MinLong(sampleXL, fwi->width - 1));
        sampleYL = MaxLong(0L, MinLong(sampleYL, fwi->height - 1));

        // Sample the pixel
        const A_long rowBytes = fwi->rowbytes / sizeof(PF_Pixel8);
        PF_Pixel8* sampleP = static_cast<PF_Pixel8*>(fwi->inputP) + sampleYL * rowBytes + sampleXL;

        // Set output pixel
        *outP = *sampleP;
    }

    return err;
}


static PF_Err
FractalWarpFunc16(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    PF_Err err = PF_Err_NONE;

    FractalWarpInfo* fwi = reinterpret_cast<FractalWarpInfo*>(refcon);

    if (fwi) {
        const float width = static_cast<float>(fwi->width);
        const float height = static_cast<float>(fwi->height);
        const float aspectRatio = width / height;

        // Normalize coordinates (0-1)
        float st_x, st_y;

        // Setup coordinate system
        if (fwi->screenSpace) {
            st_x = static_cast<float>(xL) / width;
            st_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
            st_x *= aspectRatio;  // Apply aspect ratio correction
        }
        else {
            st_x = static_cast<float>(xL) / width;
            st_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
            st_x *= aspectRatio;  // Apply aspect ratio correction
        }

        // Apply position offset
        st_x += (fwi->position.x / 1000.0f) * -1.0f;
        st_y += (fwi->position.y / 1000.0f) * -1.0f;  // Reversed sign

        // Calculate parallax values
        const float parallax_x = (fwi->parallax.x / 200.0f) * -1.0f;
        const float parallax_y = (fwi->parallax.y / 200.0f) * -1.0f;  // Reversed sign

        // Calculate displacement using FBM
        const float dx = FBM(
            ((st_x - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            ((st_y - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            parallax_x, parallax_y,
            fwi->octaves, fwi->lacunarity
        );

        const float dy = FBM(
            ((st_x + 25.3f - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            ((st_y + 12.9f - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            parallax_x, parallax_y,
            fwi->octaves, fwi->lacunarity
        );

        // Sample coordinates
        float sample_x, sample_y;
        if (fwi->screenSpace) {
            sample_x = static_cast<float>(xL) / width;
            sample_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
        }
        else {
            sample_x = static_cast<float>(xL) / width;
            sample_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
        }

        // Apply displacement
        sample_x += (dx - 0.5f) * fwi->magnitude;
        sample_y += (dy - 0.5f) * fwi->magnitude;

        // Boundary handling
        // If coordinates are outside the range [0,1], set to transparent
        if (sample_x < 0.0f || sample_x > 1.0f || sample_y < 0.0f || sample_y > 1.0f) {
            // Set to transparent or black pixel
            outP->alpha = 0;
            outP->red = outP->green = outP->blue = 0;
            return err;
        }

        // Convert to pixel coordinates
        A_long sampleXL = static_cast<A_long>(sample_x * width + 0.5f);
        A_long sampleYL = static_cast<A_long>((1.0f - sample_y) * height + 0.5f);  // Reversed Y back to screen coordinates

        // Ensure sample coordinates are within valid range
        sampleXL = MaxLong(0L, MinLong(sampleXL, fwi->width - 1));
        sampleYL = MaxLong(0L, MinLong(sampleYL, fwi->height - 1));

        // Sample the pixel
        A_long rowBytes16 = fwi->rowbytes / sizeof(PF_Pixel16);
        PF_Pixel16* sampleP = static_cast<PF_Pixel16*>(fwi->inputP) + sampleYL * rowBytes16 + sampleXL;

        // Set output pixel
        *outP = *sampleP;
    }

    return err;
}

static PF_Err
FractalWarpFunc32(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    PF_Err err = PF_Err_NONE;

    FractalWarpInfo* fwi = reinterpret_cast<FractalWarpInfo*>(refcon);

    if (fwi) {
        const float width = static_cast<float>(fwi->width);
        const float height = static_cast<float>(fwi->height);
        const float aspectRatio = width / height;

        // Normalize coordinates (0-1)
        float st_x, st_y;

        // Setup coordinate system
        if (fwi->screenSpace) {
            st_x = static_cast<float>(xL) / width;
            st_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
            st_x *= aspectRatio;  // Apply aspect ratio correction
        }
        else {
            st_x = static_cast<float>(xL) / width;
            st_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
            st_x *= aspectRatio;  // Apply aspect ratio correction
        }

        // Apply position offset
        st_x += (fwi->position.x / 1000.0f) * -1.0f;
        st_y += (fwi->position.y / 1000.0f) * -1.0f;  // Reversed sign

        // Calculate parallax values
        const float parallax_x = (fwi->parallax.x / 200.0f) * -1.0f;
        const float parallax_y = (fwi->parallax.y / 200.0f) * -1.0f;  // Reversed sign

        // Calculate displacement using FBM
        const float dx = FBM(
            ((st_x - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            ((st_y - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            parallax_x, parallax_y,
            fwi->octaves, fwi->lacunarity
        );

        const float dy = FBM(
            ((st_x + 25.3f - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            ((st_y + 12.9f - 0.5f) * 3.0f * fwi->detail) + 0.5f,
            parallax_x, parallax_y,
            fwi->octaves, fwi->lacunarity
        );

        // Sample coordinates
        float sample_x, sample_y;
        if (fwi->screenSpace) {
            sample_x = static_cast<float>(xL) / width;
            sample_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
        }
        else {
            sample_x = static_cast<float>(xL) / width;
            sample_y = 1.0f - (static_cast<float>(yL) / height);  // Reversed Y
        }

        // Apply displacement
        sample_x += (dx - 0.5f) * fwi->magnitude;
        sample_y += (dy - 0.5f) * fwi->magnitude;

        // Boundary handling
        // If coordinates are outside the range [0,1], set to transparent
        if (sample_x < 0.0f || sample_x > 1.0f || sample_y < 0.0f || sample_y > 1.0f) {
            // Set to transparent or black pixel
            outP->alpha = 0.0f;
            outP->red = outP->green = outP->blue = 0.0f;
            return err;
        }

        // Convert to pixel coordinates
        A_long sampleXL = static_cast<A_long>(sample_x * width + 0.5f);
        A_long sampleYL = static_cast<A_long>((1.0f - sample_y) * height + 0.5f);  // Reversed Y back to screen coordinates

        // Ensure sample coordinates are within valid range
        sampleXL = MaxLong(0L, MinLong(sampleXL, fwi->width - 1));
        sampleYL = MaxLong(0L, MinLong(sampleYL, fwi->height - 1));

        // Sample the pixel
        A_long rowBytes32 = fwi->rowbytes / sizeof(PF_PixelFloat);
        PF_PixelFloat* sampleP = static_cast<PF_PixelFloat*>(fwi->inputP) + sampleYL * rowBytes32 + sampleXL;

        // Set output pixel
        *outP = *sampleP;
    }

    return err;
}



/**
 * Data structure for thread-local rendering information
 */
typedef struct {
    FractalWarpInfo info;
    PF_InData* in_data;
    PF_EffectWorld* input_worldP;
    PF_EffectWorld* output_worldP;
} ThreadRenderData;

/**
 * Smart PreRender function - prepares for rendering
 */
static PF_Err
SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    // Initialize effect info structure
    FractalWarpInfo info;
    AEFX_CLR_STRUCT(info);

    PF_ParamDef param_copy;
    AEFX_CLR_STRUCT(param_copy);

    // Initialize max_result_rect to the current output request rect
    extra->output->max_result_rect = extra->input->output_request.rect;

    // Get magnitude parameter
    ERR(PF_CHECKOUT_PARAM(in_data, MAGNITUDE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) info.magnitude = param_copy.u.fs_d.value;

    // Calculate buffer expansion based on magnitude
    A_long expansion = 0;
    // Expand buffer based on magnitude
    expansion = ceil(info.magnitude * 1.5);

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
        FRACTALWARP_INPUT,
        FRACTALWARP_INPUT,
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
 * Smart Render function - performs the actual effect rendering
 */
static PF_Err
SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Use thread_local to ensure each thread has its own render data
    thread_local ThreadRenderData render_data;
    AEFX_CLR_STRUCT(render_data);
    render_data.in_data = in_data;

    // Checkout input layer pixels
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, FRACTALWARP_INPUT, &render_data.input_worldP));
    if (!err) {
        // Checkout output buffer
        ERR(extra->cb->checkout_output(in_data->effect_ref, &render_data.output_worldP));
    }

    if (!err && render_data.input_worldP && render_data.output_worldP) {
        PF_ParamDef param_copy;
        AEFX_CLR_STRUCT(param_copy);

        // Get all effect parameters
        // Position parameter
        ERR(PF_CHECKOUT_PARAM(in_data, FRACTALWARP_POSITION, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) {
            render_data.info.position.x = param_copy.u.td.x_value;
            render_data.info.position.y = param_copy.u.td.y_value;
        }

        // Parallax parameter
        ERR(PF_CHECKOUT_PARAM(in_data, FRACTALWARP_PARALLAX, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) {
            render_data.info.parallax.x = param_copy.u.td.x_value;
            render_data.info.parallax.y = param_copy.u.td.y_value;
        }

        // Magnitude parameter
        ERR(PF_CHECKOUT_PARAM(in_data, MAGNITUDE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.magnitude = param_copy.u.fs_d.value;

        // Detail parameter
        ERR(PF_CHECKOUT_PARAM(in_data, DETAIL_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.detail = param_copy.u.fs_d.value;

        // Lacunarity parameter
        ERR(PF_CHECKOUT_PARAM(in_data, LACUNARITY_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.lacunarity = param_copy.u.fs_d.value;

        // Screen Space parameter
        ERR(PF_CHECKOUT_PARAM(in_data, SCREENSPACE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.screenSpace = param_copy.u.bd.value;

        // Octaves parameter
        ERR(PF_CHECKOUT_PARAM(in_data, OCTAVES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.octaves = param_copy.u.sd.value;

        if (!err) {
            // Set up dimensions
            render_data.info.width = render_data.output_worldP->width;
            render_data.info.height = render_data.output_worldP->height;
            render_data.info.inputP = render_data.input_worldP->data;
            render_data.info.rowbytes = render_data.input_worldP->rowbytes;

            // Clear output buffer before processing
            PF_Pixel empty_pixel = { 0, 0, 0, 0 };
            ERR(suites.FillMatteSuite2()->fill(
                in_data->effect_ref,
                &empty_pixel,
                NULL,
                render_data.output_worldP));

            if (!err) {
                // Calculate lines to process
                A_long linesL = render_data.output_worldP->height;

                // Determine bit depth based on rowbytes
                A_long pixels_per_row = render_data.input_worldP->width;
                A_long bytes_per_pixel = render_data.input_worldP->rowbytes / pixels_per_row;

                // Determine bit depth based on bytes per pixel
                bool is_8bit = (bytes_per_pixel <= 4);
                bool is_16bit = (bytes_per_pixel > 4 && bytes_per_pixel <= 8);
                bool is_32bit = (bytes_per_pixel > 8);

                // Process based on calculated bit depth
                if (is_32bit) {
                    // 32-bit float processing
                    ERR(suites.IterateFloatSuite1()->iterate(
                        in_data,
                        0,                                // progress base
                        linesL,                           // progress final
                        render_data.input_worldP,         // src 
                        NULL,                             // area - null for all pixels
                        (void*)&render_data.info,         // refcon - custom data pointer
                        FractalWarpFunc32,                // pixel function pointer
                        render_data.output_worldP));
                }
                else if (is_16bit) {
                    // 16-bit processing
                    ERR(suites.Iterate16Suite2()->iterate(
                        in_data,
                        0,                                // progress base
                        linesL,                           // progress final
                        render_data.input_worldP,         // src 
                        NULL,                             // area - null for all pixels
                        (void*)&render_data.info,         // refcon - custom data pointer
                        FractalWarpFunc16,                // pixel function pointer
                        render_data.output_worldP));
                }
                else {
                    // 8-bit processing
                    ERR(suites.Iterate8Suite2()->iterate(
                        in_data,
                        0,                                // progress base
                        linesL,                           // progress final
                        render_data.input_worldP,         // src 
                        NULL,                             // area - null for all pixels
                        (void*)&render_data.info,         // refcon - custom data pointer
                        FractalWarpFunc8,                 // pixel function pointer
                        render_data.output_worldP));
                }
            }
        }
    }

    // Check in the input layer pixels
    if (render_data.input_worldP) {
        ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, FRACTALWARP_INPUT));
    }

    return err;
}



static PF_Err
LegacyRender(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Set up FractalWarpInfo
    FractalWarpInfo fwi;
    AEFX_CLR_STRUCT(fwi);

    // Get input layer
    PF_LayerDef* input_layer = &params[FRACTALWARP_INPUT]->u.ld;

    // Get position parameter
    fwi.position.x = params[FRACTALWARP_POSITION]->u.td.x_value;
    fwi.position.y = params[FRACTALWARP_POSITION]->u.td.y_value;

    // Get parallax parameter
    fwi.parallax.x = params[FRACTALWARP_PARALLAX]->u.td.x_value;
    fwi.parallax.y = params[FRACTALWARP_PARALLAX]->u.td.y_value;

    // Get magnitude parameter
    fwi.magnitude = params[MAGNITUDE_DISK_ID]->u.fs_d.value;

    // Get detail parameter
    fwi.detail = params[DETAIL_DISK_ID]->u.fs_d.value;

    // Get lacunarity parameter
    fwi.lacunarity = params[LACUNARITY_DISK_ID]->u.fs_d.value;

    // Get screen space parameter
    fwi.screenSpace = params[SCREENSPACE_DISK_ID]->u.bd.value;

    // Get octaves parameter
    fwi.octaves = params[OCTAVES_DISK_ID]->u.sd.value;

    // Set up dimensions
    fwi.width = output->width;
    fwi.height = output->height;
    fwi.inputP = input_layer->data;
    fwi.rowbytes = input_layer->rowbytes;

    // Calculate lines to process
    A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

    // Process based on bit depth
    // Check for world flags properly
    PF_Boolean deep_flag = PF_WORLD_IS_DEEP(output);
    PF_Boolean float_flag = PF_WORLD_IS_FLOAT(output);

    if (deep_flag) {
        if (float_flag) {
            // 32-bit float processing
            err = suites.IterateFloatSuite1()->iterate(
                in_data,
                0,                              // progress base
                linesL,                         // progress final
                input_layer,                    // src 
                NULL,                           // area - null for all pixels
                (void*)&fwi,                    // refcon - custom data pointer
                FractalWarpFunc32,              // pixel function pointer
                output);
        }
        else {
            // 16-bit processing
            err = suites.Iterate16Suite2()->iterate(
                in_data,
                0,                              // progress base
                linesL,                         // progress final
                input_layer,                    // src 
                NULL,                           // area - null for all pixels
                (void*)&fwi,                    // refcon - custom data pointer
                FractalWarpFunc16,              // pixel function pointer
                output);
        }
    }
    else {
        // 8-bit processing
        err = suites.Iterate8Suite2()->iterate(
            in_data,
            0,                              // progress base
            linesL,                         // progress final
            input_layer,                    // src 
            NULL,                           // area - null for all pixels
            (void*)&fwi,                    // refcon - custom data pointer
            FractalWarpFunc8,               // pixel function pointer
            output);
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

    // Register the effect with After Effects
    result = PF_REGISTER_EFFECT(
        inPtr,
        inPluginDataCallBackPtr,
        "Fractal Warp",          // Effect name
        "DKT FractalWarp",       // Match name - make sure this is unique
        "DKT Effects",                // Category
        AE_RESERVED_INFO
    );

    return result;
}

/**
 * Main entry point for the effect
 * Handles all command dispatching
 */
PF_Err
EffectMain(
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

