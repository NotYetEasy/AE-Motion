/**
 * WaveWarp.cpp
 * After Effects plugin that creates wave-based distortion effects
 * Created by DKT
 */

#include "WaveWarp.h"
#include <string.h>
#include <stdio.h>  // For sprintf
#include <math.h>
#include <mutex>

#ifndef PF_WorldFlag_FLOAT
#define PF_WorldFlag_FLOAT 1
#endif

#ifndef PF_WORLD_IS_FLOAT
#define PF_WORLD_IS_FLOAT(W) ((W)->world_flags & PF_WorldFlag_FLOAT)
#endif

 // Define min/max macros since AEFX_MIN/MAX aren't found
#define WW_MIN(a, b) ((a) < (b) ? (a) : (b))
#define WW_MAX(a, b) ((a) > (b) ? (a) : (b))

// Helper function to convert degrees to radians
static PF_FpLong DegreesToRadians(PF_FpLong degrees) {
    return degrees * 0.0174533;
}

// Helper function to convert radians to degrees
static PF_FpLong RadiansToDegrees(PF_FpLong radians) {
    return radians * 180.0 / PF_PI;
}

/**
 * Performs bilinear interpolation for 8-bit pixels
 */
PF_Pixel8 SampleBilinear8(PF_Pixel8* src, PF_FpLong x, PF_FpLong y, A_long width, A_long height, A_long rowbytes) {
    // Get the integer and fractional parts
    A_long x0 = (A_long)x;
    A_long y0 = (A_long)y;
    A_long x1 = WW_MIN(x0 + 1, width - 1);
    A_long y1 = WW_MIN(y0 + 1, height - 1);

    PF_FpLong fx = x - x0;
    PF_FpLong fy = y - y0;

    // Get the four surrounding pixels
    PF_Pixel8* p00 = (PF_Pixel8*)((char*)src + y0 * rowbytes + x0 * sizeof(PF_Pixel8));
    PF_Pixel8* p01 = (PF_Pixel8*)((char*)src + y0 * rowbytes + x1 * sizeof(PF_Pixel8));
    PF_Pixel8* p10 = (PF_Pixel8*)((char*)src + y1 * rowbytes + x0 * sizeof(PF_Pixel8));
    PF_Pixel8* p11 = (PF_Pixel8*)((char*)src + y1 * rowbytes + x1 * sizeof(PF_Pixel8));

    // Bilinear interpolation for each channel
    PF_Pixel8 result;
    result.alpha = (A_u_char)(
        p00->alpha * (1 - fx) * (1 - fy) +
        p01->alpha * fx * (1 - fy) +
        p10->alpha * (1 - fx) * fy +
        p11->alpha * fx * fy);

    result.red = (A_u_char)(
        p00->red * (1 - fx) * (1 - fy) +
        p01->red * fx * (1 - fy) +
        p10->red * (1 - fx) * fy +
        p11->red * fx * fy);

    result.green = (A_u_char)(
        p00->green * (1 - fx) * (1 - fy) +
        p01->green * fx * (1 - fy) +
        p10->green * (1 - fx) * fy +
        p11->green * fx * fy);

    result.blue = (A_u_char)(
        p00->blue * (1 - fx) * (1 - fy) +
        p01->blue * fx * (1 - fy) +
        p10->blue * (1 - fx) * fy +
        p11->blue * fx * fy);

    return result;
}

/**
 * Performs bilinear interpolation for 16-bit pixels
 */
PF_Pixel16 SampleBilinear16(PF_Pixel16* src, PF_FpLong x, PF_FpLong y, A_long width, A_long height, A_long rowbytes) {
    // Get the integer and fractional parts
    A_long x0 = (A_long)x;
    A_long y0 = (A_long)y;
    A_long x1 = WW_MIN(x0 + 1, width - 1);
    A_long y1 = WW_MIN(y0 + 1, height - 1);

    PF_FpLong fx = x - x0;
    PF_FpLong fy = y - y0;

    // Get the four surrounding pixels
    PF_Pixel16* p00 = (PF_Pixel16*)((char*)src + y0 * rowbytes + x0 * sizeof(PF_Pixel16));
    PF_Pixel16* p01 = (PF_Pixel16*)((char*)src + y0 * rowbytes + x1 * sizeof(PF_Pixel16));
    PF_Pixel16* p10 = (PF_Pixel16*)((char*)src + y1 * rowbytes + x0 * sizeof(PF_Pixel16));
    PF_Pixel16* p11 = (PF_Pixel16*)((char*)src + y1 * rowbytes + x1 * sizeof(PF_Pixel16));

    // Bilinear interpolation for each channel
    PF_Pixel16 result;
    result.alpha = (A_u_short)(
        p00->alpha * (1 - fx) * (1 - fy) +
        p01->alpha * fx * (1 - fy) +
        p10->alpha * (1 - fx) * fy +
        p11->alpha * fx * fy);

    result.red = (A_u_short)(
        p00->red * (1 - fx) * (1 - fy) +
        p01->red * fx * (1 - fy) +
        p10->red * (1 - fx) * fy +
        p11->red * fx * fy);

    result.green = (A_u_short)(
        p00->green * (1 - fx) * (1 - fy) +
        p01->green * fx * (1 - fy) +
        p10->green * (1 - fx) * fy +
        p11->green * fx * fy);

    result.blue = (A_u_short)(
        p00->blue * (1 - fx) * (1 - fy) +
        p01->blue * fx * (1 - fy) +
        p10->blue * (1 - fx) * fy +
        p11->blue * fx * fy);

    return result;
}

/**
 * Performs bilinear interpolation for float pixels
 */
PF_PixelFloat SampleBilinearFloat(PF_PixelFloat* src, PF_FpLong x, PF_FpLong y, A_long width, A_long height, A_long rowbytes) {
    // Get the integer and fractional parts
    A_long x0 = (A_long)x;
    A_long y0 = (A_long)y;
    A_long x1 = WW_MIN(x0 + 1, width - 1);
    A_long y1 = WW_MIN(y0 + 1, height - 1);

    PF_FpLong fx = x - x0;
    PF_FpLong fy = y - y0;

    // Get the four surrounding pixels
    PF_PixelFloat* p00 = (PF_PixelFloat*)((char*)src + y0 * rowbytes + x0 * sizeof(PF_PixelFloat));
    PF_PixelFloat* p01 = (PF_PixelFloat*)((char*)src + y0 * rowbytes + x1 * sizeof(PF_PixelFloat));
    PF_PixelFloat* p10 = (PF_PixelFloat*)((char*)src + y1 * rowbytes + x0 * sizeof(PF_PixelFloat));
    PF_PixelFloat* p11 = (PF_PixelFloat*)((char*)src + y1 * rowbytes + x1 * sizeof(PF_PixelFloat));

    // Bilinear interpolation for each channel
    PF_PixelFloat result;
    result.alpha = (PF_FpShort)(
        p00->alpha * (1 - fx) * (1 - fy) +
        p01->alpha * fx * (1 - fy) +
        p10->alpha * (1 - fx) * fy +
        p11->alpha * fx * fy);

    result.red = (PF_FpShort)(
        p00->red * (1 - fx) * (1 - fy) +
        p01->red * fx * (1 - fy) +
        p10->red * (1 - fx) * fy +
        p11->red * fx * fy);

    result.green = (PF_FpShort)(
        p00->green * (1 - fx) * (1 - fy) +
        p01->green * fx * (1 - fy) +
        p10->green * (1 - fx) * fy +
        p11->green * fx * fy);

    result.blue = (PF_FpShort)(
        p00->blue * (1 - fx) * (1 - fy) +
        p01->blue * fx * (1 - fy) +
        p10->blue * (1 - fx) * fy +
        p11->blue * fx * fy);

    return result;
}

/**
 * About command handler - displays plugin information
 */
static PF_Err
About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
        "Wave Warp v%d.%d\r"
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
static PF_Err
GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
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
        PF_OutFlag_I_EXPAND_BUFFER |
        PF_OutFlag_WIDE_TIME_INPUT |
        PF_OutFlag_NON_PARAM_VARY |
        PF_OutFlag_FORCE_RERENDER;

    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
        PF_OutFlag2_FLOAT_COLOR_AWARE |
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

    return PF_Err_NONE;
}

/**
 * Sets up the parameters for the effect
 */
static PF_Err
ParamsSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err      err = PF_Err_NONE;
    PF_ParamDef def;

    AEFX_CLR_STRUCT(def);

    // Phase parameter
    PF_ADD_FLOAT_SLIDERX("Phase",
        0,
        500,
        0,
        5,
        0,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        PHASE_DISK_ID);

    // Direction Angle parameter - now as a slider in degrees
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Angle",
        -3600,
        3600,
        -180,
        180,
        0,
        PF_Precision_INTEGER,
        0,
        0,
        ANGLE_DISK_ID);

    // Spacing parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Spacing",
        WAVEWARP_SPACING_MIN,
        WAVEWARP_SPACING_MAX,
        WAVEWARP_SPACING_MIN,
        WAVEWARP_SPACING_MAX,
        WAVEWARP_SPACING_DFLT,
        PF_Precision_TENTHS,
        0,
        0,
        SPACING_DISK_ID);

    // Magnitude parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Magnitude",
        WAVEWARP_MAGNITUDE_MIN,
        WAVEWARP_MAGNITUDE_MAX,
        WAVEWARP_MAGNITUDE_MIN,
        WAVEWARP_MAGNITUDE_MAX,
        WAVEWARP_MAGNITUDE_DFLT,
        PF_Precision_TENTHS,
        0,
        0,
        MAGNITUDE_DISK_ID);

    // Warp Angle parameter - now as a slider in degrees
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Warp Angle",
        -180,
        180,
        -180,
        180,
        90,  // Default to 90 degrees for waving flag effect
        PF_Precision_INTEGER,
        0,
        0,
        WARPANGLE_DISK_ID);

    // Damping parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Damping",
        WAVEWARP_DAMPING_MIN,
        WAVEWARP_DAMPING_MAX,
        WAVEWARP_DAMPING_MIN,
        WAVEWARP_DAMPING_MAX,
        WAVEWARP_DAMPING_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        DAMPING_DISK_ID);

    // Damping Space parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Damping Space",
        WAVEWARP_DAMPINGSPACE_MIN,
        WAVEWARP_DAMPINGSPACE_MAX,
        WAVEWARP_DAMPINGSPACE_MIN,
        WAVEWARP_DAMPINGSPACE_MAX,
        WAVEWARP_DAMPINGSPACE_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        DAMPINGSPACE_DISK_ID);

    // Damping Origin parameter
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Damping Origin",
        WAVEWARP_DAMPINGORIGIN_MIN,
        WAVEWARP_DAMPINGORIGIN_MAX,
        WAVEWARP_DAMPINGORIGIN_MIN,
        WAVEWARP_DAMPINGORIGIN_MAX,
        WAVEWARP_DAMPINGORIGIN_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        DAMPINGORIGIN_DISK_ID);

    // Screen Space checkbox
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Screen Space",
        "Screen Space",
        FALSE,
        0,
        SCREENSPACE_DISK_ID);

    out_data->num_params = WAVEWARP_NUM_PARAMS;

    return err;
}

/**
 * Pixel processing function for 8-bit color depth
 */
static PF_Err
WaveWarpFunc8(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    PF_Err      err = PF_Err_NONE;

    WaveWarpInfo* wiP = reinterpret_cast<WaveWarpInfo*>(refcon);

    if (wiP) {
        // Convert to normalized coordinates (0.0 to 1.0)
        PF_FpLong x_norm = (PF_FpLong)xL / wiP->width;
        PF_FpLong y_norm = (PF_FpLong)yL / wiP->height;

        // REVERSE both angles by negating them
        PF_FpLong direction_deg = -wiP->direction;
        PF_FpLong warpangle_deg = -wiP->offset;

        // Calculate angles
        PF_FpLong a1 = direction_deg * 0.0174533;
        PF_FpLong a2 = (direction_deg + warpangle_deg) * 0.0174533;

        // Get normalized coords
        PF_FpLong st_x, st_y;
        if (wiP->screenSpace) {
            st_x = x_norm;
            st_y = y_norm;
        }
        else {
            st_x = x_norm;
            st_y = y_norm;
        }

        // Calculate raw vector
        PF_FpLong raw_v_x = cos(a1);
        PF_FpLong raw_v_y = -sin(a1);

        // Calculate raw position - dot product
        PF_FpLong raw_p = (st_x * raw_v_x) + (st_y * raw_v_y);

        // Calculate space damping
        PF_FpLong space_damp = 1.0;
        if (wiP->dampingSpace < 0.0) {
            space_damp = 1.0 - (WW_MIN(fabs(raw_p - wiP->dampingOrigin), 1.0) * (0.0 - wiP->dampingSpace));
        }
        else if (wiP->dampingSpace > 0.0) {
            space_damp = 1.0 - ((1.0 - WW_MIN(fabs(raw_p - wiP->dampingOrigin), 1.0)) * wiP->dampingSpace);
        }

        // Calculate space
        PF_FpLong space = wiP->spacing * space_damp;

        // Calculate vector
        PF_FpLong v_x = cos(a1) * space;
        PF_FpLong v_y = -sin(a1) * space;

        // Calculate position - dot product
        PF_FpLong p = (st_x * v_x) + (st_y * v_y);

        // Calculate distance
        PF_FpLong ddist = fabs(p / space);

        // Calculate damping
        PF_FpLong damp = 1.0;
        if (wiP->damping < 0.0) {
            damp = 1.0 - (WW_MIN(fabs(ddist - wiP->dampingOrigin), 1.0) * (0.0 - wiP->damping));
        }
        else if (wiP->damping > 0.0) {
            damp = 1.0 - ((1.0 - WW_MIN(fabs(ddist - wiP->dampingOrigin), 1.0)) * wiP->damping);
        }

        // Calculate offset
        PF_FpLong offs_x = cos(a2) * (wiP->magnitude * damp) / 100.0;
        PF_FpLong offs_y = -sin(a2) * (wiP->magnitude * damp) / 100.0;

        // Apply sine wave with phase
        PF_FpLong wave = sin(p + wiP->phase * 6.28318);
        offs_x *= wave;
        offs_y *= wave;

        // Calculate sample coordinates in normalized space
        PF_FpLong sample_x_norm = x_norm + offs_x;
        PF_FpLong sample_y_norm = y_norm + offs_y;

        // Convert to pixel coordinates
        PF_FpLong sample_x_f = sample_x_norm * wiP->width;
        PF_FpLong sample_y_f = sample_y_norm * wiP->height;

        // Only check horizontal boundaries to make pixels disappear
        // For vertical boundaries, clamp to keep them visible
        if (sample_x_f < 0 || sample_x_f >= wiP->width) {
            // Set output to transparent black (invisible) for horizontal out-of-bounds
            outP->alpha = 0;
            outP->red = 0;
            outP->green = 0;
            outP->blue = 0;
        }
        else {
            // Clamp vertical coordinates to keep them visible
            sample_y_f = WW_MAX(0.0, WW_MIN(sample_y_f, wiP->height - 1.0));

            // Use the correct source pointer from the srcData field
            PF_Pixel8* srcP = (PF_Pixel8*)(wiP->srcData);

            // Use bilinear interpolation for points inside the image
            *outP = SampleBilinear8(srcP, sample_x_f, sample_y_f, wiP->width, wiP->height, wiP->rowbytes);
        }
    }
    else {
        // If wiP is null, just copy input to output
        *outP = *inP;
    }

    return err;
}

/**
 * Pixel processing function for 16-bit color depth
 */
static PF_Err
WaveWarpFunc16(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    PF_Err      err = PF_Err_NONE;

    WaveWarpInfo* wiP = reinterpret_cast<WaveWarpInfo*>(refcon);

    if (wiP) {
        // Convert to normalized coordinates (0.0 to 1.0)
        PF_FpLong x_norm = (PF_FpLong)xL / wiP->width;
        PF_FpLong y_norm = (PF_FpLong)yL / wiP->height;

        // REVERSE both angles by negating them
        PF_FpLong direction_deg = -wiP->direction;
        PF_FpLong warpangle_deg = -wiP->offset;

        // Calculate angles
        PF_FpLong a1 = direction_deg * 0.0174533;
        PF_FpLong a2 = (direction_deg + warpangle_deg) * 0.0174533;

        // Get normalized coords
        PF_FpLong st_x, st_y;
        if (wiP->screenSpace) {
            st_x = x_norm;
            st_y = y_norm;
        }
        else {
            st_x = x_norm;
            st_y = y_norm;
        }

        // Calculate raw vector
        PF_FpLong raw_v_x = cos(a1);
        PF_FpLong raw_v_y = -sin(a1);

        // Calculate raw position - dot product
        PF_FpLong raw_p = (st_x * raw_v_x) + (st_y * raw_v_y);

        // Calculate space damping
        PF_FpLong space_damp = 1.0;
        if (wiP->dampingSpace < 0.0) {
            space_damp = 1.0 - (WW_MIN(fabs(raw_p - wiP->dampingOrigin), 1.0) * (0.0 - wiP->dampingSpace));
        }
        else if (wiP->dampingSpace > 0.0) {
            space_damp = 1.0 - ((1.0 - WW_MIN(fabs(raw_p - wiP->dampingOrigin), 1.0)) * wiP->dampingSpace);
        }

        // Calculate space
        PF_FpLong space = wiP->spacing * space_damp;

        // Calculate vector
        PF_FpLong v_x = cos(a1) * space;
        PF_FpLong v_y = -sin(a1) * space;

        // Calculate position - dot product
        PF_FpLong p = (st_x * v_x) + (st_y * v_y);

        // Calculate distance
        PF_FpLong ddist = fabs(p / space);

        // Calculate damping
        PF_FpLong damp = 1.0;
        if (wiP->damping < 0.0) {
            damp = 1.0 - (WW_MIN(fabs(ddist - wiP->dampingOrigin), 1.0) * (0.0 - wiP->damping));
        }
        else if (wiP->damping > 0.0) {
            damp = 1.0 - ((1.0 - WW_MIN(fabs(ddist - wiP->dampingOrigin), 1.0)) * wiP->damping);
        }

        // Calculate offset
        PF_FpLong offs_x = cos(a2) * (wiP->magnitude * damp) / 100.0;
        PF_FpLong offs_y = -sin(a2) * (wiP->magnitude * damp) / 100.0;

        // Apply sine wave with phase
        PF_FpLong wave = sin(p + wiP->phase * 6.28318);
        offs_x *= wave;
        offs_y *= wave;

        // Calculate sample coordinates in normalized space
        PF_FpLong sample_x_norm = x_norm + offs_x;
        PF_FpLong sample_y_norm = y_norm + offs_y;

        // Convert to pixel coordinates
        PF_FpLong sample_x_f = sample_x_norm * wiP->width;
        PF_FpLong sample_y_f = sample_y_norm * wiP->height;

        // Only check horizontal boundaries to make pixels disappear
        // For vertical boundaries, clamp to keep them visible
        if (sample_x_f < 0 || sample_x_f >= wiP->width) {
            // Set output to transparent black (invisible) for horizontal out-of-bounds
            outP->alpha = 0;
            outP->red = 0;
            outP->green = 0;
            outP->blue = 0;
        }
        else {
            // Clamp vertical coordinates to keep them visible
            sample_y_f = WW_MAX(0.0, WW_MIN(sample_y_f, wiP->height - 1.0));

            // Use the correct source pointer from the srcData field
            PF_Pixel16* srcP = (PF_Pixel16*)(wiP->srcData);

            // Use bilinear interpolation for points inside the image
            *outP = SampleBilinear16(srcP, sample_x_f, sample_y_f, wiP->width, wiP->height, wiP->rowbytes);
        }
    }
    else {
        // If wiP is null, just copy input to output
        *outP = *inP;
    }

    return err;
}

/**
 * Pixel processing function for 32-bit float color depth
 */
static PF_Err
WaveWarpFunc32(
    void* refcon,
    A_long      xL,
    A_long      yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    PF_Err      err = PF_Err_NONE;

    WaveWarpInfo* wiP = reinterpret_cast<WaveWarpInfo*>(refcon);

    if (wiP) {
        // Convert to normalized coordinates (0.0 to 1.0)
        PF_FpLong x_norm = (PF_FpLong)xL / wiP->width;
        PF_FpLong y_norm = (PF_FpLong)yL / wiP->height;

        // REVERSE both angles by negating them
        PF_FpLong direction_deg = -wiP->direction;
        PF_FpLong warpangle_deg = -wiP->offset;

        // Calculate angles
        PF_FpLong a1 = direction_deg * 0.0174533;
        PF_FpLong a2 = (direction_deg + warpangle_deg) * 0.0174533;

        // Get normalized coords
        PF_FpLong st_x, st_y;
        if (wiP->screenSpace) {
            st_x = x_norm;
            st_y = y_norm;
        }
        else {
            st_x = x_norm;
            st_y = y_norm;
        }

        // Calculate raw vector
        PF_FpLong raw_v_x = cos(a1);
        PF_FpLong raw_v_y = -sin(a1);

        // Calculate raw position - dot product
        PF_FpLong raw_p = (st_x * raw_v_x) + (st_y * raw_v_y);

        // Calculate space damping
        PF_FpLong space_damp = 1.0;
        if (wiP->dampingSpace < 0.0) {
            space_damp = 1.0 - (WW_MIN(fabs(raw_p - wiP->dampingOrigin), 1.0) * (0.0 - wiP->dampingSpace));
        }
        else if (wiP->dampingSpace > 0.0) {
            space_damp = 1.0 - ((1.0 - WW_MIN(fabs(raw_p - wiP->dampingOrigin), 1.0)) * wiP->dampingSpace);
        }

        // Calculate space
        PF_FpLong space = wiP->spacing * space_damp;

        // Calculate vector
        PF_FpLong v_x = cos(a1) * space;
        PF_FpLong v_y = -sin(a1) * space;

        // Calculate position - dot product
        PF_FpLong p = (st_x * v_x) + (st_y * v_y);

        // Calculate distance
        PF_FpLong ddist = fabs(p / space);

        // Calculate damping
        PF_FpLong damp = 1.0;
        if (wiP->damping < 0.0) {
            damp = 1.0 - (WW_MIN(fabs(ddist - wiP->dampingOrigin), 1.0) * (0.0 - wiP->damping));
        }
        else if (wiP->damping > 0.0) {
            damp = 1.0 - ((1.0 - WW_MIN(fabs(ddist - wiP->dampingOrigin), 1.0)) * wiP->damping);
        }

        // Calculate offset
        PF_FpLong offs_x = cos(a2) * (wiP->magnitude * damp) / 100.0;
        PF_FpLong offs_y = -sin(a2) * (wiP->magnitude * damp) / 100.0;

        // Apply sine wave with phase
        PF_FpLong wave = sin(p + wiP->phase * 6.28318);
        offs_x *= wave;
        offs_y *= wave;

        // Calculate sample coordinates in normalized space
        PF_FpLong sample_x_norm = x_norm + offs_x;
        PF_FpLong sample_y_norm = y_norm + offs_y;

        // Convert to pixel coordinates
        PF_FpLong sample_x_f = sample_x_norm * wiP->width;
        PF_FpLong sample_y_f = sample_y_norm * wiP->height;

        // Only check horizontal boundaries to make pixels disappear
        // For vertical boundaries, clamp to keep them visible
        if (sample_x_f < 0 || sample_x_f >= wiP->width) {
            // Set output to transparent black (invisible) for horizontal out-of-bounds
            outP->alpha = 0.0f;
            outP->red = 0.0f;
            outP->green = 0.0f;
            outP->blue = 0.0f;
        }
        else {
            // Clamp vertical coordinates to keep them visible
            sample_y_f = WW_MAX(0.0, WW_MIN(sample_y_f, wiP->height - 1.0));

            // Use the correct source pointer from the srcData field
            PF_PixelFloat* srcP = (PF_PixelFloat*)(wiP->srcData);

            // Use bilinear interpolation for points inside the image
            *outP = SampleBilinearFloat(srcP, sample_x_f, sample_y_f, wiP->width, wiP->height, wiP->rowbytes);
        }
    }
    else {
        // If wiP is null, just copy input to output
        *outP = *inP;
    }

    return err;
}

/**
 * Data structure for thread-local rendering information
 */
typedef struct {
    WaveWarpInfo info;
    PF_InData* in_data;
    PF_EffectWorld* input_worldP;
    PF_EffectWorld* output_worldP;
} ThreadRenderData;

/**
 * Smart PreRender function - prepares for rendering
 */
static PF_Err SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    // Initialize effect info structure
    WaveWarpInfo info;
    AEFX_CLR_STRUCT(info);

    PF_ParamDef param_copy;
    AEFX_CLR_STRUCT(param_copy);

    // Initialize max_result_rect to the current output request rect
    extra->output->max_result_rect = extra->input->output_request.rect;

    // Get magnitude parameter
    ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_MAGNITUDE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
    if (!err) info.magnitude = param_copy.u.fs_d.value;

    // Calculate buffer expansion based on magnitude
    A_long expansion = 0;
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
        WAVEWARP_INPUT,
        WAVEWARP_INPUT,
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

/**
 * Smart Render function - performs the actual effect rendering
 */
static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Use thread_local to ensure each thread has its own render data
    thread_local ThreadRenderData render_data;
    AEFX_CLR_STRUCT(render_data);
    render_data.in_data = in_data;

    // Checkout input layer pixels
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, WAVEWARP_INPUT, &render_data.input_worldP));
    if (!err) {
        // Checkout output buffer
        ERR(extra->cb->checkout_output(in_data->effect_ref, &render_data.output_worldP));
    }

    if (!err && render_data.input_worldP && render_data.output_worldP) {
        PF_ParamDef param_copy;
        AEFX_CLR_STRUCT(param_copy);

        // Get all effect parameters
        // Phase parameter
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_PHASE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.phase = param_copy.u.fs_d.value;

        // Angle parameter
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.direction = param_copy.u.fs_d.value;

        // Spacing parameter
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_SPACING, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.spacing = param_copy.u.fs_d.value;

        // Magnitude parameter
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_MAGNITUDE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.magnitude = param_copy.u.fs_d.value;

        // Warp Angle parameter
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_WARPANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.offset = param_copy.u.fs_d.value;

        // Damping parameter
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_DAMPING, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.damping = param_copy.u.fs_d.value;

        // Damping Space parameter
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_DAMPINGSPACE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.dampingSpace = param_copy.u.fs_d.value;

        // Damping Origin parameter
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_DAMPINGORIGIN, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.dampingOrigin = param_copy.u.fs_d.value;

        // Screen Space checkbox
        ERR(PF_CHECKOUT_PARAM(in_data, WAVEWARP_SCREENSPACE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
        if (!err) render_data.info.screenSpace = param_copy.u.bd.value;

        if (!err) {
            // Get layer dimensions
            render_data.info.width = render_data.input_worldP->width;
            render_data.info.height = render_data.input_worldP->height;
            render_data.info.rowbytes = render_data.input_worldP->rowbytes;

            // Set the source data pointer
            render_data.info.srcData = render_data.input_worldP->data;

            // Process the image based on bit depth
            A_long linesL = render_data.output_worldP->height;

            // Use bytes-per-pixel method for bit depth detection
            A_long bytes_per_pixel = render_data.output_worldP->rowbytes / render_data.output_worldP->width;
            bool is_8bit = (bytes_per_pixel <= 4);
            bool is_16bit = (bytes_per_pixel > 4 && bytes_per_pixel <= 8);
            bool is_32bit = (bytes_per_pixel > 8);

            if (is_32bit) {
                // 32-bit float processing
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,                          // progress base
                    linesL,                     // progress final
                    render_data.input_worldP,   // src
                    NULL,                       // area - null for all pixels
                    (void*)&render_data.info,   // refcon - our parameter struct
                    WaveWarpFunc32,             // pixel function for 32-bit float
                    render_data.output_worldP));
            }
            else if (is_16bit) {
                // 16-bit processing
                ERR(suites.Iterate16Suite2()->iterate(
                    in_data,
                    0,                          // progress base
                    linesL,                     // progress final
                    render_data.input_worldP,   // src
                    NULL,                       // area - null for all pixels
                    (void*)&render_data.info,   // refcon - our parameter struct
                    WaveWarpFunc16,             // pixel function for 16-bit
                    render_data.output_worldP));
            }
            else {
                // 8-bit processing
                ERR(suites.Iterate8Suite2()->iterate(
                    in_data,
                    0,                          // progress base
                    linesL,                     // progress final
                    render_data.input_worldP,   // src
                    NULL,                       // area - null for all pixels
                    (void*)&render_data.info,   // refcon - our parameter struct
                    WaveWarpFunc8,              // pixel function for 8-bit
                    render_data.output_worldP));
            }
        }
    }

    // Check in the input layer pixels
    if (render_data.input_worldP) {
        ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, WAVEWARP_INPUT));
    }

    return err;
}

/**
 * Legacy rendering function for older AE versions
 */
static PF_Err
Render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err              err = PF_Err_NONE;
    AEGP_SuiteHandler   suites(in_data->pica_basicP);

    // Setup the wave warp parameters
    WaveWarpInfo wiP;
    AEFX_CLR_STRUCT(wiP);

    wiP.phase = params[WAVEWARP_PHASE]->u.fs_d.value;

    // Get the direction and warp angles directly in degrees from sliders
    PF_FpLong direction_deg = params[WAVEWARP_ANGLE]->u.fs_d.value;
    PF_FpLong warpangle_deg = params[WAVEWARP_WARPANGLE]->u.fs_d.value;

    // Store the angles in degrees
    wiP.direction = direction_deg;
    wiP.offset = warpangle_deg;

    wiP.spacing = params[WAVEWARP_SPACING]->u.fs_d.value;
    wiP.magnitude = params[WAVEWARP_MAGNITUDE]->u.fs_d.value;
    wiP.damping = params[WAVEWARP_DAMPING]->u.fs_d.value;
    wiP.dampingSpace = params[WAVEWARP_DAMPINGSPACE]->u.fs_d.value;
    wiP.dampingOrigin = params[WAVEWARP_DAMPINGORIGIN]->u.fs_d.value;
    wiP.screenSpace = params[WAVEWARP_SCREENSPACE]->u.bd.value;

    // Get layer dimensions
    wiP.width = output->width;
    wiP.height = output->height;
    wiP.rowbytes = output->rowbytes;

    // Set the source data pointer
    wiP.srcData = params[WAVEWARP_INPUT]->u.ld.data;

    A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

    // Use bytes-per-pixel method for bit depth detection
    A_long bytes_per_pixel = output->rowbytes / output->width;
    bool is_8bit = (bytes_per_pixel <= 4);
    bool is_16bit = (bytes_per_pixel > 4 && bytes_per_pixel <= 8);
    bool is_32bit = (bytes_per_pixel > 8);

    // Process the image using our wave warp function
    if (is_32bit) {
        // 32-bit float processing
        ERR(suites.IterateFloatSuite1()->iterate(
            in_data,
            0,                              // progress base
            linesL,                         // progress final
            &params[WAVEWARP_INPUT]->u.ld,  // src
            NULL,                           // area - null for all pixels
            (void*)&wiP,                    // refcon - our parameter struct
            WaveWarpFunc32,                 // pixel function for 32-bit float
            output));
    }
    else if (is_16bit) {
        // 16-bit processing
        ERR(suites.Iterate16Suite2()->iterate(
            in_data,
            0,                              // progress base
            linesL,                         // progress final
            &params[WAVEWARP_INPUT]->u.ld,  // src
            NULL,                           // area - null for all pixels
            (void*)&wiP,                    // refcon - our parameter struct
            WaveWarpFunc16,                 // pixel function for 16-bit
            output));
    }
    else {
        // 8-bit processing
        ERR(suites.Iterate8Suite2()->iterate(
            in_data,
            0,                              // progress base
            linesL,                         // progress final
            &params[WAVEWARP_INPUT]->u.ld,  // src
            NULL,                           // area - null for all pixels
            (void*)&wiP,                    // refcon - our parameter struct
            WaveWarpFunc8,                  // pixel function for 8-bit
            output));
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
        "Alight Wave Warp",          // Effect name
        "DKT Alight Wave Warp",      // Match name - make sure this is unique
        "DKT Effects",        // Category
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
    PF_Cmd          cmd,
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output,
    void* extra)
{
    PF_Err      err = PF_Err_NONE;

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

