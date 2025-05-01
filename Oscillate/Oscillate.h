#pragma once
#ifndef Oscillate_H
#define Oscillate_H

#include "OscillateKernel.cl.h"
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
#include "OscillateKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCreated by DKT with Unknown's help.\rUnder development!!\rDiscord: dkt0 and unknown1234\rContact us if you want to contribute or report bugs!"

#define NAME			"Oscillate"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

/* Parameter defaults */
#define ANGLE_MIN       -3600.0
#define ANGLE_MAX       3600.0
#define ANGLE_DFLT      45.0

#define FREQUENCY_MIN   0.0
#define FREQUENCY_MAX   16.0
#define FREQUENCY_DFLT  2.0

#define MAGNITUDE_MIN   0
#define MAGNITUDE_MAX   4000
#define MAGNITUDE_DFLT  25

#define PHASE_MIN       0.0
#define PHASE_MAX       1000.0
#define PHASE_DFLT      0.0

enum {
    RANDOMMOVE_INPUT = 0,
    DIRECTION_SLIDER,
    ANGLE_SLIDER,
    FREQUENCY_SLIDER,
    MAGNITUDE_SLIDER,
    WAVE_TYPE_SLIDER,
    PHASE_SLIDER,
    TILES_GROUP,        // Group start
    X_TILES_DISK_ID,
    Y_TILES_DISK_ID,
    MIRROR_DISK_ID,
    TILES_GROUP_END,    // Group end
    RANDOMMOVE_NUM_PARAMS
};

typedef struct RandomMoveInfo {
    A_long    direction;
    PF_FpLong angle;
    PF_FpLong frequency;
    PF_FpLong magnitude;
    A_long    wave_type;
    PF_FpLong phase;
    PF_Boolean x_tiles;    // X tiling parameter
    PF_Boolean y_tiles;    // Y tiling parameter
    PF_Boolean mirror;     // Mirror parameter
} RandomMoveInfo;

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
/*
 ** Plugins must not rely on a host autorelease pool.
 ** Create a pool if autorelease is used, or Cocoa convention calls, such as Metal, might internally autorelease.
 */
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

typedef struct {
    RandomMoveInfo info;
    PF_FpLong current_time;
    PF_InData* in_data;
    PF_EffectWorld* input_worldP;
    PF_EffectWorld* output_worldP;
} ThreadRenderData;

#endif // Oscillate_H
