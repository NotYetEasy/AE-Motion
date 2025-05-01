#pragma once
#ifndef INNER_PINCH_BULGE_H
#define INNER_PINCH_BULGE_H

#include "InnerPinchBulge_Kernel.cl.h"
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
#include "InnerPinchBulge_Kernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"Creates a pinch or bulge effect from a center point with adjustable radius and strength."

#define NAME			"Inner Pinch/Bulge"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
    INNER_PINCH_BULGE_INPUT = 0,
    INNER_PINCH_BULGE_CENTER,
    INNER_PINCH_BULGE_STRENGTH,
    INNER_PINCH_BULGE_RADIUS,
    INNER_PINCH_BULGE_FEATHER,
    INNER_PINCH_BULGE_GAUSSIAN,
    INNER_PINCH_BULGE_TILES_GROUP, // Group start
    INNER_PINCH_BULGE_X_TILES,
    INNER_PINCH_BULGE_Y_TILES,
    INNER_PINCH_BULGE_MIRROR,
    INNER_PINCH_BULGE_TILES_GROUP_END, // Group end
    INNER_PINCH_BULGE_NUM_PARAMS
};

/* Parameter disk IDs */
enum {
    CENTER_DISK_ID = 1,
    STRENGTH_DISK_ID,
    RADIUS_DISK_ID,
    FEATHER_DISK_ID,
    GAUSSIAN_DISK_ID,
    X_TILES_DISK_ID,
    Y_TILES_DISK_ID,
    MIRROR_DISK_ID
};

#define	INNER_PINCH_BULGE_CENTER_X_DFLT    0
#define	INNER_PINCH_BULGE_CENTER_Y_DFLT    0
#define	INNER_PINCH_BULGE_STRENGTH_MIN     -1.00
#define	INNER_PINCH_BULGE_STRENGTH_MAX     1.00
#define	INNER_PINCH_BULGE_STRENGTH_DFLT    0.5
#define	INNER_PINCH_BULGE_RADIUS_MIN       0.0
#define	INNER_PINCH_BULGE_RADIUS_MAX       2.5
#define	INNER_PINCH_BULGE_RADIUS_DFLT      0.3
#define	INNER_PINCH_BULGE_FEATHER_MIN      0.0
#define	INNER_PINCH_BULGE_FEATHER_MAX      1.0
#define	INNER_PINCH_BULGE_FEATHER_DFLT     0.1

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

typedef struct
{
    PF_FpLong            strength;
    PF_FpLong            radius;
    PF_FpLong            feather;
    PF_Boolean           useGaussian;
    PF_Fixed             center_x;
    PF_Fixed             center_y;
    PF_Boolean           x_tiles;
    PF_Boolean           y_tiles;
    PF_Boolean           mirror;
    PF_EffectWorld* input;
    PF_InData* in_data;
} PinchBulgeInfo;

#endif // INNER_PINCH_BULGE_H
