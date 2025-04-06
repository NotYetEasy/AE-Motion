#include "DirectionalBlur.h"
#include <stdio.h>
#include <math.h>
#include <cmath>

#define STR_NAME "DKT Directional Blur"
#define STR_DESCRIPTION "Blurs the layer in a specific direction"
#define STR_STRENGTH_PARAM_NAME "Strength"
#define STR_ANGLE_PARAM_NAME "Angle"

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

    out_data->out_flags |= PF_OutFlag_DEEP_COLOR_AWARE |
        PF_OutFlag_PIX_INDEPENDENT;

    out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_SMART_RENDER |
        PF_OutFlag2_FLOAT_COLOR_AWARE |
        PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
        PF_OutFlag2_REVEALS_ZERO_ALPHA;

    return PF_Err_NONE;
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

    PF_ADD_FLOAT_SLIDERX(STR_STRENGTH_PARAM_NAME,
        DBLUR_STRENGTH_MIN,
        DBLUR_STRENGTH_MAX,
        DBLUR_STRENGTH_MIN,
        DBLUR_STRENGTH_MAX,
        DBLUR_STRENGTH_DFLT,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        STRENGTH_DISK_ID);

    AEFX_CLR_STRUCT(def);

    PF_ADD_FLOAT_SLIDERX(STR_ANGLE_PARAM_NAME,
        0,
        3600,
        0,
        360,
        0,
        PF_Precision_TENTHS,
        0,
        0,
        ANGLE_DISK_ID);

    out_data->num_params = DBLUR_NUM_PARAMS;

    return err;
}

template <typename PixelType>
static inline void SampleBilinear(
    const void* src_data,
    const A_long rowbytes,
    const A_long width,
    const A_long height,
    const float x,
    const float y,
    PixelType* outP)
{
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float fx = x - x0;
    const float fy = y - y0;

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w10 = fx * (1.0f - fy);
    const float w11 = fx * fy;

    float r00 = 0, g00 = 0, b00 = 0, a00 = 0;
    float r01 = 0, g01 = 0, b01 = 0, a01 = 0;
    float r10 = 0, g10 = 0, b10 = 0, a10 = 0;
    float r11 = 0, g11 = 0, b11 = 0, a11 = 0;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        const PF_PixelFloat* base = static_cast<const PF_PixelFloat*>(src_data);
        const A_long stride = rowbytes / sizeof(PF_PixelFloat);

        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            const PF_PixelFloat* p00 = &base[y0 * stride + x0];
            r00 = p00->red;
            g00 = p00->green;
            b00 = p00->blue;
            a00 = p00->alpha;
        }

        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
            const PF_PixelFloat* p01 = &base[y1 * stride + x0];
            r01 = p01->red;
            g01 = p01->green;
            b01 = p01->blue;
            a01 = p01->alpha;
        }

        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
            const PF_PixelFloat* p10 = &base[y0 * stride + x1];
            r10 = p10->red;
            g10 = p10->green;
            b10 = p10->blue;
            a10 = p10->alpha;
        }

        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            const PF_PixelFloat* p11 = &base[y1 * stride + x1];
            r11 = p11->red;
            g11 = p11->green;
            b11 = p11->blue;
            a11 = p11->alpha;
        }

        outP->red = r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11;
        outP->green = g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11;
        outP->blue = b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11;
        outP->alpha = a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        const PF_Pixel16* base = static_cast<const PF_Pixel16*>(src_data);
        const A_long stride = rowbytes / sizeof(PF_Pixel16);

        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            const PF_Pixel16* p00 = &base[y0 * stride + x0];
            r00 = p00->red;
            g00 = p00->green;
            b00 = p00->blue;
            a00 = p00->alpha;
        }

        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
            const PF_Pixel16* p01 = &base[y1 * stride + x0];
            r01 = p01->red;
            g01 = p01->green;
            b01 = p01->blue;
            a01 = p01->alpha;
        }

        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
            const PF_Pixel16* p10 = &base[y0 * stride + x1];
            r10 = p10->red;
            g10 = p10->green;
            b10 = p10->blue;
            a10 = p10->alpha;
        }

        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            const PF_Pixel16* p11 = &base[y1 * stride + x1];
            r11 = p11->red;
            g11 = p11->green;
            b11 = p11->blue;
            a11 = p11->alpha;
        }

        outP->red = static_cast<A_u_short>(r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11 + 0.5f);
        outP->green = static_cast<A_u_short>(g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11 + 0.5f);
        outP->blue = static_cast<A_u_short>(b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11 + 0.5f);
        outP->alpha = static_cast<A_u_short>(a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11 + 0.5f);
    }
    else {
        const PF_Pixel8* base = static_cast<const PF_Pixel8*>(src_data);
        const A_long stride = rowbytes / sizeof(PF_Pixel8);

        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height) {
            const PF_Pixel8* p00 = &base[y0 * stride + x0];
            r00 = p00->red;
            g00 = p00->green;
            b00 = p00->blue;
            a00 = p00->alpha;
        }

        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height) {
            const PF_Pixel8* p01 = &base[y1 * stride + x0];
            r01 = p01->red;
            g01 = p01->green;
            b01 = p01->blue;
            a01 = p01->alpha;
        }

        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height) {
            const PF_Pixel8* p10 = &base[y0 * stride + x1];
            r10 = p10->red;
            g10 = p10->green;
            b10 = p10->blue;
            a10 = p10->alpha;
        }

        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height) {
            const PF_Pixel8* p11 = &base[y1 * stride + x1];
            r11 = p11->red;
            g11 = p11->green;
            b11 = p11->blue;
            a11 = p11->alpha;
        }

        outP->red = static_cast<A_u_char>(r00 * w00 + r01 * w01 + r10 * w10 + r11 * w11 + 0.5f);
        outP->green = static_cast<A_u_char>(g00 * w00 + g01 * w01 + g10 * w10 + g11 * w11 + 0.5f);
        outP->blue = static_cast<A_u_char>(b00 * w00 + b01 * w01 + b10 * w10 + b11 * w11 + 0.5f);
        outP->alpha = static_cast<A_u_char>(a00 * w00 + a01 * w01 + a10 * w10 + a11 * w11 + 0.5f);
    }
}

template <typename PixelType>
static PF_Err DirectionalBlurGeneric(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    PF_Err err = PF_Err_NONE;
    BlurInfo* biP = reinterpret_cast<BlurInfo*>(refcon);

    if (!biP) {
        *outP = *inP;
        return PF_Err_BAD_CALLBACK_PARAM;
    }

    const float width = static_cast<float>(biP->width);
    const float height = static_cast<float>(biP->height);
    const float strength = static_cast<float>(biP->strength);
    const float angle_rad = static_cast<float>(-biP->angle * PF_RAD_PER_DEGREE);

    if (strength <= 0.001f) {
        outP->alpha = inP->alpha;
        outP->red = inP->red;
        outP->green = inP->green;
        outP->blue = inP->blue;
        return PF_Err_NONE;
    }

    const float velocity_x = cos(angle_rad) * strength;
    const float velocity_y = -sin(angle_rad) * strength;

    const float adjusted_velocity_x = velocity_x * (width / height);

    const float texelSize_x = 1.0f / width;
    const float texelSize_y = 1.0f / height;

    const float speed_x = adjusted_velocity_x / texelSize_x;
    const float speed_y = velocity_y / texelSize_y;
    const float speed = sqrtf(speed_x * speed_x + speed_y * speed_y);

    const int nSamples = static_cast<int>(MAX(MIN(speed, 100.01f), 1.01f));

    const float normX = static_cast<float>(xL) / width;
    const float normY = static_cast<float>(yL) / height;

    float accumR = static_cast<float>(inP->red);
    float accumG = static_cast<float>(inP->green);
    float accumB = static_cast<float>(inP->blue);
    float accumA = static_cast<float>(inP->alpha);

    const float inv_nSamples_minus_1 = 1.0f / static_cast<float>(nSamples - 1);
    const float inv_nSamples = 1.0f / static_cast<float>(nSamples);

    for (int i = 1; i < nSamples; i++) {
        const float t = static_cast<float>(i) * inv_nSamples_minus_1 - 0.5f;
        const float sample_norm_x = normX - adjusted_velocity_x * t;
        const float sample_norm_y = normY - velocity_y * t;
        const float sample_x = sample_norm_x * width;
        const float sample_y = sample_norm_y * height;

        PixelType sample;
        SampleBilinear<PixelType>(biP->src, biP->rowbytes, biP->width, biP->height,
            sample_x, sample_y, &sample);

        accumR += static_cast<float>(sample.red);
        accumG += static_cast<float>(sample.green);
        accumB += static_cast<float>(sample.blue);
        accumA += static_cast<float>(sample.alpha);
    }

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

    return err;
}

static PF_Err DirectionalBlur8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return DirectionalBlurGeneric<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err DirectionalBlur16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return DirectionalBlurGeneric<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err DirectionalBlurFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return DirectionalBlurGeneric<PF_PixelFloat>(refcon, xL, yL, inP, outP);
}

static PF_Err
Render(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    BlurInfo bi;
    AEFX_CLR_STRUCT(bi);

    PF_LayerDef* input_layer = &params[DBLUR_INPUT]->u.ld;
    A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

    bi.strength = params[DBLUR_STRENGTH]->u.fs_d.value;
    bi.angle = params[DBLUR_ANGLE]->u.fs_d.value;
    bi.width = input_layer->width;
    bi.height = input_layer->height;
    bi.src = input_layer->data;
    bi.rowbytes = input_layer->rowbytes;

    const double bytesPerPixel = static_cast<double>(bi.rowbytes) / static_cast<double>(bi.width);

    if (bytesPerPixel >= 16.0) {
        ERR(suites.IterateFloatSuite1()->iterate(
            in_data, 0, linesL, input_layer, NULL, &bi, DirectionalBlurFloat, output));
    }
    else if (bytesPerPixel >= 8.0) {
        ERR(suites.Iterate16Suite2()->iterate(
            in_data, 0, linesL, input_layer, NULL, &bi, DirectionalBlur16, output));
    }
    else {
        ERR(suites.Iterate8Suite2()->iterate(
            in_data, 0, linesL, input_layer, NULL, &bi, DirectionalBlur8, output));
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

    try {
        PF_ParamDef strength_param;
        AEFX_CLR_STRUCT(strength_param);
        ERR(PF_CHECKOUT_PARAM(in_data, DBLUR_STRENGTH, in_data->current_time, in_data->time_step, in_data->time_scale, &strength_param));

        PF_ParamDef angle_param;
        AEFX_CLR_STRUCT(angle_param);
        ERR(PF_CHECKOUT_PARAM(in_data, DBLUR_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &angle_param));

        PF_Rect request_rect = extra->input->output_request.rect;

        float strength = strength_param.u.fs_d.value;
        float angle_rad = static_cast<float>(-angle_param.u.fs_d.value * PF_RAD_PER_DEGREE);

        float velocity_x = cos(angle_rad) * strength;
        float velocity_y = -sin(angle_rad) * strength;

        float width = static_cast<float>(in_data->width);
        float height = static_cast<float>(in_data->height);
        float adjusted_velocity_x = velocity_x * (width / height);

        float expansion_x = fabs(adjusted_velocity_x) * 200.0f;
        float expansion_y = fabs(velocity_y) * 200.0f;

        PF_Rect expanded_rect = request_rect;
        expanded_rect.left -= static_cast<A_long>(expansion_x);
        expanded_rect.top -= static_cast<A_long>(expansion_y);
        expanded_rect.right += static_cast<A_long>(expansion_x);
        expanded_rect.bottom += static_cast<A_long>(expansion_y);

        PF_RenderRequest req = extra->input->output_request;
        req.preserve_rgb_of_zero_alpha = TRUE;

        PF_CheckoutResult checkout;
        ERR(extra->cb->checkout_layer(in_data->effect_ref,
            DBLUR_INPUT,
            DBLUR_INPUT,
            &req,
            in_data->current_time,
            in_data->time_step,
            in_data->time_scale,
            &checkout));

        if (!err) {
            extra->output->max_result_rect = expanded_rect;
            extra->output->result_rect = request_rect;
            extra->output->solid = FALSE;
            extra->output->flags = PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;
        }

        ERR(PF_CHECKIN_PARAM(in_data, &strength_param));
        ERR(PF_CHECKIN_PARAM(in_data, &angle_param));
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
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    try {
        PF_ParamDef strength_param, angle_param;
        AEFX_CLR_STRUCT(strength_param);
        AEFX_CLR_STRUCT(angle_param);

        ERR(PF_CHECKOUT_PARAM(in_data, DBLUR_STRENGTH, in_data->current_time, in_data->time_step, in_data->time_scale, &strength_param));
        ERR(PF_CHECKOUT_PARAM(in_data, DBLUR_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &angle_param));

        PF_EffectWorld* input_worldP = NULL;
        err = extra->cb->checkout_layer_pixels(in_data->effect_ref, DBLUR_INPUT, &input_worldP);
        if (err) return err;

        PF_EffectWorld* output_worldP = NULL;
        err = extra->cb->checkout_output(in_data->effect_ref, &output_worldP);
        if (err) return err;

        PF_PixelFormat pixelFormat;
        PF_WorldSuite2* wsP = NULL;
        ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
        ERR(wsP->PF_GetPixelFormat(output_worldP, &pixelFormat));
        ERR(suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2));

        if (input_worldP && output_worldP) {
            A_long orig_left = (output_worldP->width - input_worldP->width) / 2;
            A_long orig_top = (output_worldP->height - input_worldP->height) / 2;
            A_long orig_right = orig_left + input_worldP->width;
            A_long orig_bottom = orig_top + input_worldP->height;

            if (strength_param.u.fs_d.value <= 0.001f) {
                for (A_long y = orig_top; y < orig_bottom; y++) {
                    A_long src_y = y - orig_top;

                    switch (pixelFormat) {
                    case PF_PixelFormat_ARGB128:
                    {
                        PF_PixelFloat* srcRow = (PF_PixelFloat*)((char*)input_worldP->data + src_y * input_worldP->rowbytes);
                        PF_PixelFloat* dstRow = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                        for (A_long x = orig_left; x < orig_right; x++) {
                            A_long src_x = x - orig_left;
                            dstRow[x] = srcRow[src_x];
                        }
                        break;
                    }
                    case PF_PixelFormat_ARGB64:
                    {
                        PF_Pixel16* srcRow = (PF_Pixel16*)((char*)input_worldP->data + src_y * input_worldP->rowbytes);
                        PF_Pixel16* dstRow = (PF_Pixel16*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                        for (A_long x = orig_left; x < orig_right; x++) {
                            A_long src_x = x - orig_left;
                            dstRow[x] = srcRow[src_x];
                        }
                        break;
                    }
                    case PF_PixelFormat_ARGB32:
                    {
                        PF_Pixel8* srcRow = (PF_Pixel8*)((char*)input_worldP->data + src_y * input_worldP->rowbytes);
                        PF_Pixel8* dstRow = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                        for (A_long x = orig_left; x < orig_right; x++) {
                            A_long src_x = x - orig_left;
                            dstRow[x] = srcRow[src_x];
                        }
                        break;
                    }
                    }
                }

                ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, DBLUR_INPUT));
                ERR(PF_CHECKIN_PARAM(in_data, &strength_param));
                ERR(PF_CHECKIN_PARAM(in_data, &angle_param));
                return err;
            }

            const float width = static_cast<float>(input_worldP->width);
            const float height = static_cast<float>(input_worldP->height);
            float strength = static_cast<float>(strength_param.u.fs_d.value);
            strength = MAX(strength, 0.00000001f);

            const float angle_rad = static_cast<float>(-angle_param.u.fs_d.value * PF_RAD_PER_DEGREE);
            const float velocity_x = cos(angle_rad) * strength;
            const float velocity_y = -sin(angle_rad) * strength;
            const float adjusted_velocity_x = velocity_x * (width / height);
            const float texelSize_x = 1.0f / width;
            const float texelSize_y = 1.0f / height;
            const float speed_x = adjusted_velocity_x / texelSize_x;
            const float speed_y = velocity_y / texelSize_y;
            const float speed = sqrtf(speed_x * speed_x + speed_y * speed_y);
            const int nSamples = static_cast<int>(MAX(MIN(speed, 100.01f), 1.01f));
            const float inv_nSamples = 1.0f / static_cast<float>(nSamples);
            const float inv_nSamples_minus_1 = 1.0f / static_cast<float>(nSamples - 1);

            switch (pixelFormat) {
            case PF_PixelFormat_ARGB128:
            {
                for (A_long y = 0; y < output_worldP->height; y++) {
                    PF_PixelFloat* outRow = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                    float normY = (y - orig_top) / height;

                    for (A_long x = 0; x < output_worldP->width; x++) {
                        float normX = (x - orig_left) / width;
                        float accumR = 0, accumG = 0, accumB = 0, accumA = 0;

                        for (int i = 0; i < nSamples; i++) {
                            const float t = static_cast<float>(i) * inv_nSamples_minus_1 - 0.5f;
                            const float sample_norm_x = normX - adjusted_velocity_x * t;
                            const float sample_norm_y = normY - velocity_y * t;
                            const float sample_x = sample_norm_x * width + orig_left;
                            const float sample_y = sample_norm_y * height + orig_top;

                            if (sample_x >= orig_left && sample_x < orig_right &&
                                sample_y >= orig_top && sample_y < orig_bottom) {
                                A_long sx = static_cast<A_long>(sample_x) - orig_left;
                                A_long sy = static_cast<A_long>(sample_y) - orig_top;
                                PF_PixelFloat* inRow = (PF_PixelFloat*)((char*)input_worldP->data + sy * input_worldP->rowbytes);
                                accumA += inRow[sx].alpha;
                                accumR += inRow[sx].red;
                                accumG += inRow[sx].green;
                                accumB += inRow[sx].blue;
                            }
                        }

                        outRow[x].alpha = accumA * inv_nSamples;
                        outRow[x].red = accumR * inv_nSamples;
                        outRow[x].green = accumG * inv_nSamples;
                        outRow[x].blue = accumB * inv_nSamples;
                    }
                }
                break;
            }
            case PF_PixelFormat_ARGB64:
            {
                for (A_long y = 0; y < output_worldP->height; y++) {
                    PF_Pixel16* outRow = (PF_Pixel16*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                    float normY = (y - orig_top) / height;

                    for (A_long x = 0; x < output_worldP->width; x++) {
                        float normX = (x - orig_left) / width;
                        float accumR = 0, accumG = 0, accumB = 0, accumA = 0;

                        for (int i = 0; i < nSamples; i++) {
                            const float t = static_cast<float>(i) * inv_nSamples_minus_1 - 0.5f;
                            const float sample_norm_x = normX - adjusted_velocity_x * t;
                            const float sample_norm_y = normY - velocity_y * t;
                            const float sample_x = sample_norm_x * width + orig_left;
                            const float sample_y = sample_norm_y * height + orig_top;

                            if (sample_x >= orig_left && sample_x < orig_right &&
                                sample_y >= orig_top && sample_y < orig_bottom) {
                                A_long sx = static_cast<A_long>(sample_x) - orig_left;
                                A_long sy = static_cast<A_long>(sample_y) - orig_top;
                                PF_Pixel16* inRow = (PF_Pixel16*)((char*)input_worldP->data + sy * input_worldP->rowbytes);
                                accumA += inRow[sx].alpha;
                                accumR += inRow[sx].red;
                                accumG += inRow[sx].green;
                                accumB += inRow[sx].blue;
                            }
                        }

                        outRow[x].alpha = static_cast<A_u_short>(accumA * inv_nSamples + 0.5f);
                        outRow[x].red = static_cast<A_u_short>(accumR * inv_nSamples + 0.5f);
                        outRow[x].green = static_cast<A_u_short>(accumG * inv_nSamples + 0.5f);
                        outRow[x].blue = static_cast<A_u_short>(accumB * inv_nSamples + 0.5f);
                    }
                }
                break;
            }
            case PF_PixelFormat_ARGB32:
            {
                for (A_long y = 0; y < output_worldP->height; y++) {
                    PF_Pixel8* outRow = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);
                    float normY = (y - orig_top) / height;

                    for (A_long x = 0; x < output_worldP->width; x++) {
                        float normX = (x - orig_left) / width;
                        float accumR = 0, accumG = 0, accumB = 0, accumA = 0;

                        for (int i = 0; i < nSamples; i++) {
                            const float t = static_cast<float>(i) * inv_nSamples_minus_1 - 0.5f;
                            const float sample_norm_x = normX - adjusted_velocity_x * t;
                            const float sample_norm_y = normY - velocity_y * t;
                            const float sample_x = sample_norm_x * width + orig_left;
                            const float sample_y = sample_norm_y * height + orig_top;

                            if (sample_x >= orig_left && sample_x < orig_right &&
                                sample_y >= orig_top && sample_y < orig_bottom) {
                                A_long sx = static_cast<A_long>(sample_x) - orig_left;
                                A_long sy = static_cast<A_long>(sample_y) - orig_top;
                                PF_Pixel8* inRow = (PF_Pixel8*)((char*)input_worldP->data + sy * input_worldP->rowbytes);
                                accumA += inRow[sx].alpha;
                                accumR += inRow[sx].red;
                                accumG += inRow[sx].green;
                                accumB += inRow[sx].blue;
                            }
                        }

                        outRow[x].alpha = static_cast<A_u_char>(accumA * inv_nSamples + 0.5f);
                        outRow[x].red = static_cast<A_u_char>(accumR * inv_nSamples + 0.5f);
                        outRow[x].green = static_cast<A_u_char>(accumG * inv_nSamples + 0.5f);
                        outRow[x].blue = static_cast<A_u_char>(accumB * inv_nSamples + 0.5f);
                    }
                }
                break;
            }
            }
        }

        if (input_worldP) {
            ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, DBLUR_INPUT));
        }

        ERR(PF_CHECKIN_PARAM(in_data, &strength_param));
        ERR(PF_CHECKIN_PARAM(in_data, &angle_param));
    }
    catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
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
    PF_Err result = PF_REGISTER_EFFECT_EXT2(
        inPtr,
        inPluginDataCallBackPtr,
        "DKT Directional Blur",
        "DKT Directional Blur",
        "DKT Effects",
        AE_RESERVED_INFO,
        "EffectMain",
        "");

    return result;
}

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

        case PF_Cmd_RENDER:
            err = Render(in_data, out_data, params, output);
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

