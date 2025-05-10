#pragma once
#ifndef RasterTransform
#define RasterTransform

#include "RasterTransformKernel.cl.h"
#include "AEConfig.h"
#include "entry.h"
#include "AEFX_SuiteHelper.h"
#include "PrSDKAESupport.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_Macros.h"
#include "AEGP_SuiteHandler.h"
#include "String_Utils.h"
#include "Param_Utils.h"
#include "Smart_Utils.h"

#if _WIN32
#include <CL/cl.h>
#define HAS_HLSL 1
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_HLSL 0
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "RasterTransformKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define STR_NAME          "Transform"
#define STR_DESCRIPTION   "Raster Transform Effect\rCopyright 2023 Adobe Inc."
#define STR_SCALE_PARAM   "Scale"
#define STR_ANGLE_PARAM   "Angle"
#define STR_OFFSET_PARAM  "Offset"
#define STR_MASK_PARAM    "Mask To Layer"
#define STR_ALPHA_PARAM   "Alpha"
#define STR_FILL_PARAM    "Fill"
#define STR_SAMPLE_PARAM  "Sampling"
#define STR_NEAREST       "Nearest"
#define STR_LINEAR        "Linear"

#define MAJOR_VERSION    1
#define MINOR_VERSION    0
#define BUG_VERSION      0
#define STAGE_VERSION    PF_Stage_DEVELOP
#define BUILD_VERSION    1

#define TRANSFORM_SCALE_MIN     0.01
#define TRANSFORM_SCALE_MAX     50.0
#define TRANSFORM_SCALE_DFLT    1.0

#define TRANSFORM_ANGLE_MIN     0.0
#define TRANSFORM_ANGLE_MAX     3600.0
#define TRANSFORM_ANGLE_DFLT    0.0

#define TRANSFORM_ALPHA_MIN     0.0
#define TRANSFORM_ALPHA_MAX     1.0
#define TRANSFORM_ALPHA_DFLT    1.0

#define TRANSFORM_FILL_MIN      0.0
#define TRANSFORM_FILL_MAX      1.0
#define TRANSFORM_FILL_DFLT     0.0

#define TRANSFORM_SAMPLE_NEAREST 0
#define TRANSFORM_SAMPLE_LINEAR  1
#define TRANSFORM_SAMPLE_DFLT    TRANSFORM_SAMPLE_LINEAR

enum {
    TRANSFORM_INPUT = 0,
    TRANSFORM_SCALE,
    TRANSFORM_ANGLE,
    TRANSFORM_OFFSET,
    TRANSFORM_MASK_TO_LAYER,
    TRANSFORM_ALPHA,
    TRANSFORM_FILL,
    TRANSFORM_SAMPLE,
    TRANSFORM_NUM_PARAMS
};

enum {
    SCALE_DISK_ID = 1,
    ANGLE_DISK_ID,
    OFFSET_DISK_ID,
    MASK_DISK_ID,
    ALPHA_DISK_ID,
    FILL_DISK_ID,
    SAMPLE_DISK_ID,
};

typedef struct TransformInfo {
    float scale;
    float angle;
    float offset_x;
    float offset_y;
    int maskToLayer;
    float alpha;
    float fill;
    int sample;
    int input_width;
    int input_height;
    PF_EffectWorld* input_worldP;
} TransformInfo;

typedef struct TransformParams {
    int mSrcPitch;
    int mDstPitch;
    int m16f;
    int mWidth;
    int mHeight;
    float mScale;
    float mAngle;
    float mOffsetX;
    float mOffsetY;
    int mMaskToLayer;
    float mAlpha;
    float mFill;
    int mSample;
} TransformParams;

#if HAS_METAL
struct ScopedAutoreleasePool
{
    ScopedAutoreleasePool()
        : mPool([[NSAutoreleasePool alloc]init] )
    {
    }

    ~ScopedAutoreleasePool()
    {
        [mPool release] ;
    }

    NSAutoreleasePool* mPool;
};
#endif

extern "C" {
    DllExport
        PF_Err
        EffectMain(
            PF_Cmd          cmd,
            PF_InData* in_data,
            PF_OutData* out_data,
            PF_ParamDef* params[],
            PF_LayerDef* output,
            void* extra);
}

#endif  