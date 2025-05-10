#pragma once
#ifndef SDK_DirectionalBlur_H
#define SDK_DirectionalBlur_H

#include "DirectionalBlurKernel.cl.h"
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
#include <type_traits>

#if _WIN32
#include <CL/cl.h>
#define HAS_HLSL 1
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_HLSL 0
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "SDK_DirectionalBlur_Kernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023 Adobe Inc.\rDirectional Blur effect."

#define NAME			"DirectionalBlur"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

#define STR_NAME "DKT Directional Blur"
#define STR_DESCRIPTION "Blurs the layer in a specific direction"
#define STR_STRENGTH_PARAM_NAME "Strength"
#define STR_ANGLE_PARAM_NAME "Angle"

enum {
    DBLUR_INPUT = 0,
    DBLUR_STRENGTH,
    DBLUR_ANGLE,
    DBLUR_NUM_PARAMS
};

enum {
    STRENGTH_DISK_ID = 1,
    ANGLE_DISK_ID,
};

#define DBLUR_STRENGTH_MIN		0.0
#define DBLUR_STRENGTH_MAX		1.0
#define DBLUR_STRENGTH_DFLT		0.15

#define PF_RAD_PER_DEGREE 0.017453292519943295769236907684886

typedef struct {
    PF_FpLong       strength;
    PF_FpLong       angle;
    A_long          width;
    A_long          height;
    void* src;
    A_long          rowbytes;
    PF_EffectWorld* input;
} BlurInfo;

extern "C" {
    DllExport
        PF_Err
        EffectMain(
            PF_Cmd			cmd,
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

#endif  