#pragma once
#ifndef ReorientSphere_H
#define ReorientSphere_H

#include "ReorientSphereKernel.cl.h"
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
#include "ReorientSphereKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

typedef unsigned char        u_char;
typedef unsigned short       u_short;
typedef unsigned short       u_int16;
typedef unsigned long        u_long;
typedef short int            int16;
#define PF_TABLE_BITS    12
#define PF_TABLE_SZ_16    4096

#define PF_DEEP_COLOR_AWARE 1

#define STR_NAME          "ReorientSphere"
#define STR_DESCRIPTION   "Reorients 360-degree spherical images.\rCopyright 2025 DKT Effects."
#define STR_ORIENTATION_PARAM    "Orientation"
#define STR_ROTATION_PARAM       "Rotation"

#define MAJOR_VERSION    1
#define MINOR_VERSION    0
#define BUG_VERSION      0
#define STAGE_VERSION    PF_Stage_DEVELOP
#define BUILD_VERSION    1

enum {
    REORIENT_INPUT = 0,
    REORIENT_ORIENTATION,
    REORIENT_ROTATION,
    REORIENT_NUM_PARAMS
};

enum {
    ORIENTATION_DISK_ID = 1,
    ROTATION_DISK_ID,
};

typedef struct ReorientInfo {
    float     orientation[16];
    float     rotation[3];
    A_long    width;
    A_long    height;
    float     downsample_factor_x;
    float     downsample_factor_y;
    PF_EffectWorld* input_worldP;
} ReorientInfo, * ReorientInfoP, ** ReorientInfoH;

#define M_PI 3.14159265358979323846

typedef struct {
    float m[16];
} Matrix4x4;

typedef struct {
    A_long width;
    A_long height;
    void* worldP;
} SamplingInfo;

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