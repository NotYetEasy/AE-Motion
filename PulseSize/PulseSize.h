#pragma once
#ifndef PULSESIZE_H
#define PULSESIZE_H

#include "PulseSizeKernel.cl.h"
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
#include "PulseSizeKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCreated by DKT with Unknown's help.\rUnder development!!\rDiscord: dkt0 and unknown1234\rContact us if you want to contribute or report bugs!"

#define NAME			"PulseSize"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

#define FREQUENCY_MIN   0.0
#define FREQUENCY_MAX   16.0
#define FREQUENCY_DFLT  2.0

#define SHRINK_MIN      0.0
#define SHRINK_MAX      1.0
#define SHRINK_DFLT     0.9

#define GROW_MIN        1.0
#define GROW_MAX        2.0
#define GROW_DFLT       1.1

#define PHASE_MIN       0.0
#define PHASE_MAX       2.0
#define PHASE_DFLT      0.0

enum {
    PULSESIZE_INPUT = 0,
    FREQUENCY_SLIDER,
    SHRINK_SLIDER,
    GROW_SLIDER,
    PHASE_SLIDER,
    WAVE_TYPE_SLIDER,
    TILES_GROUP,          
    X_TILES_DISK_ID,
    Y_TILES_DISK_ID,
    MIRROR_DISK_ID,
    TILES_GROUP_END,      
    PULSESIZE_NUM_PARAMS
};

typedef struct PulseSizeInfo {
    PF_FpLong frequency;
    PF_FpLong shrink;
    PF_FpLong grow;
    PF_FpLong phase;
    A_long    wave_type;
    PF_Boolean x_tiles;       
    PF_Boolean y_tiles;       
    PF_Boolean mirror;       
} PulseSizeInfo;

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

typedef struct {
    PulseSizeInfo info;
    PF_FpLong current_time;
    A_long width;
    A_long height;
    void* input_data;
    A_long input_rowbytes;
} ThreadRenderData;

#endif  
