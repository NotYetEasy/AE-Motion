#pragma once
#ifndef StretchAxis
#define StretchAxis

#include "StretchAxisKernel.cl.h"
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
#include "StretchAxisKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION    "\nCopyright 2023.\rStretch Axis Effect."

#define NAME            "Stretch Axis"
#define MAJOR_VERSION    1
#define MINOR_VERSION    0
#define BUG_VERSION        0
#define STAGE_VERSION    PF_Stage_DEVELOP
#define BUILD_VERSION    1


enum {
    STRETCH_INPUT = 0,
    STRETCH_SCALE,
    STRETCH_ANGLE,
    STRETCH_CONTENT_ONLY,
    STRETCH_TILES_GROUP,
    STRETCH_X_TILES,
    STRETCH_Y_TILES,
    STRETCH_MIRROR,
    STRETCH_TILES_GROUP_END,
    STRETCH_NUM_PARAMS
};

#define STRETCH_SCALE_MIN        0.01
#define STRETCH_SCALE_MAX        50.0
#define STRETCH_SCALE_DFLT       1.0

#define STRETCH_ANGLE_MIN        -3600.0
#define STRETCH_ANGLE_MAX        3600.0
#define STRETCH_ANGLE_DFLT       0.0

#define SCALE_DISK_ID           1
#define ANGLE_DISK_ID           2
#define CONTENT_ONLY_DISK_ID    3
#define X_TILES_DISK_ID         4
#define Y_TILES_DISK_ID         5
#define MIRROR_DISK_ID          6

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
    float scale;
    float angle;
    bool content_only;
    bool x_tiles;
    bool y_tiles;
    bool mirror;
    bool params_changed;
} StretchParams;

typedef struct
{
    PF_InData* in_data;
    PF_EffectWorld* input;
    void* src;
    A_long rowbytes;
    A_long width;
    A_long height;
    PF_FpLong scale;
    PF_FpLong angle;
    PF_Boolean content_only;

    float input_center_x;
    float input_center_y;
    float output_center_x;
    float output_center_y;
    float offset_x;
    float offset_y;
    float last_scale;
    float last_angle;
    bool params_changed;
    PF_Boolean x_tiles;
    PF_Boolean y_tiles;
    PF_Boolean mirror;
} StretchInfo;

#endif  
