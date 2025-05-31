#pragma once
#ifndef VIGNETTE_H
#define VIGNETTE_H

#include "VignetteKernel.cl.h"
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
#include "VignetteKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION    "\nCopyright 2018-2023 Adobe Inc.\rVignette effect."

#define NAME            "DKT Vignette"
#define MAJOR_VERSION   1
#define MINOR_VERSION   1
#define BUG_VERSION     0
#define STAGE_VERSION   PF_Stage_DEVELOP
#define BUILD_VERSION   1


enum {
    VIGNETTE_INPUT = 0,
    VIGNETTE_SCALE,
    VIGNETTE_ROUNDNESS,
    VIGNETTE_FEATHER,
    VIGNETTE_STRENGTH,
    VIGNETTE_TINT,
    VIGNETTE_COLOR,
    VIGNETTE_PUNCHOUT,
    VIGNETTE_NUM_PARAMS
};


#define VIGNETTE_SCALE_MIN     0.001f
#define VIGNETTE_SCALE_MAX     2.0f
#define VIGNETTE_SCALE_DFLT    0.9f

#define VIGNETTE_ROUNDNESS_MIN  0.0f
#define VIGNETTE_ROUNDNESS_MAX  9.9f
#define VIGNETTE_ROUNDNESS_DFLT 1.5f

#define VIGNETTE_FEATHER_MIN    0.001f
#define VIGNETTE_FEATHER_MAX    1.0f
#define VIGNETTE_FEATHER_DFLT   0.5f

#define VIGNETTE_STRENGTH_MIN   0.0f
#define VIGNETTE_STRENGTH_MAX   1.0f
#define VIGNETTE_STRENGTH_DFLT  0.8f

#define VIGNETTE_TINT_MIN       0.0f
#define VIGNETTE_TINT_MAX       1.0f
#define VIGNETTE_TINT_DFLT      0.2f


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

typedef struct
{
    float mScale;
    float mRoundness;
    float mFeather;
    float mStrength;
    float mTint;
    PF_Pixel mColor;
    PF_Boolean mPunchout;
    A_long mWidth;
    A_long mHeight;
} VignetteParams;

#endif  