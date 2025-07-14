#pragma once
#ifndef AUTOSHAKE_H
#define AUTOSHAKE_H

#include "AutoShakeKernel.cl.h"
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
#include "SimplexNoise.h"

#if _WIN32
#include <CL/cl.h>
#define HAS_CUDA 1
#define HAS_HLSL 1
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_CUDA 0
#define HAS_HLSL 0
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "AutoShakeKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023-2024 DKT.\rGPU-accelerated Auto-Shake effect."

#define NAME			"Auto-Shake"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

#define AUTOSHAKE_MAGNITUDE_MIN      0
#define AUTOSHAKE_MAGNITUDE_MAX      2000
#define AUTOSHAKE_MAGNITUDE_DFLT     50

#define AUTOSHAKE_FREQUENCY_MIN      0
#define AUTOSHAKE_FREQUENCY_MAX      16
#define AUTOSHAKE_FREQUENCY_DFLT     2.0

#define AUTOSHAKE_EVOLUTION_MIN      0
#define AUTOSHAKE_EVOLUTION_MAX      2000
#define AUTOSHAKE_EVOLUTION_DFLT     0

#define AUTOSHAKE_SEED_MIN           0
#define AUTOSHAKE_SEED_MAX           5
#define AUTOSHAKE_SEED_DFLT          0

#define AUTOSHAKE_ANGLE_DFLT         45.0

#define AUTOSHAKE_SLACK_MIN          0
#define AUTOSHAKE_SLACK_MAX          1
#define AUTOSHAKE_SLACK_DFLT         0.25

#define AUTOSHAKE_ZSHAKE_MIN         0
#define AUTOSHAKE_ZSHAKE_MAX         2000
#define AUTOSHAKE_ZSHAKE_DFLT        0

enum {
    AUTOSHAKE_INPUT = 0,
    NORMAL_CHECKBOX_DISK_ID,
    AUTOSHAKE_MAGNITUDE,
    AUTOSHAKE_FREQUENCY,
    AUTOSHAKE_EVOLUTION,
    AUTOSHAKE_SEED,
    AUTOSHAKE_ANGLE,
    AUTOSHAKE_SLACK,
    AUTOSHAKE_ZSHAKE,
    TILES_GROUP_START,
    X_TILES_DISK_ID,
    Y_TILES_DISK_ID,
    MIRROR_DISK_ID,
    TILES_GROUP_END,
    COMPATIBILITY_GROUP_START,
    COMPATIBILITY_CHECKBOX_DISK_ID,
    COMPATIBILITY_MAGNITUDE_DISK_ID,
    COMPATIBILITY_SPEED_DISK_ID,
    COMPATIBILITY_EVOLUTION_DISK_ID,
    COMPATIBILITY_SEED_DISK_ID,
    COMPATIBILITY_ANGLE_DISK_ID,
    COMPATIBILITY_SLACK_DISK_ID,
    COMPATIBILITY_GROUP_END,
    AUTOSHAKE_NUM_PARAMS
};


typedef struct {
    PF_FpLong magnitude;
    PF_FpLong frequency;
    PF_FpLong evolution;
    PF_FpLong seed;
    PF_FpLong angle;
    PF_FpLong slack;
    PF_FpLong zshake;
    int x_tiles;
    int y_tiles;
    int mirror;
    PF_FpLong layer_start_seconds;
    int normal_mode;
    int compatibility_mode;
    PF_FpLong compatibility_magnitude;
    PF_FpLong compatibility_speed;
    PF_FpLong compatibility_evolution;
    PF_FpLong compatibility_seed;
    PF_FpLong compatibility_angle;
    PF_FpLong compatibility_slack;
} ShakeInfo;


struct ThreadRenderData {
    ShakeInfo info;
    A_long width;
    A_long height;
    void* input_data;
    A_long input_rowbytes;
    PF_FpLong current_time;
    PF_FpLong prev_time;
    PF_FpLong duration;
    PF_FpLong layer_start_seconds;
    PF_FpLong accumulated_phase;
    bool accumulated_phase_initialized;
    bool has_frequency_keyframes;
    int buffer_expand_x;
    int buffer_expand_y;
};





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

PF_Err NSError2PFErr(NSError* inError);
#endif 

#endif 
