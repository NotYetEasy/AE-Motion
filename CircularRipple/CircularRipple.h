#pragma once
#ifndef CircularRipple_H
#define CircularRipple_H

#include "CircularRippleKernel.cl.h"
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
#include "CircularRippleKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023-2024.\rCircular Ripple GPU effect."

#define NAME			"Circular Ripple"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	CIRCULAR_RIPPLE_INPUT = 0,
	CIRCULAR_RIPPLE_CENTER,
	CIRCULAR_RIPPLE_FREQUENCY,
	CIRCULAR_RIPPLE_STRENGTH,
	CIRCULAR_RIPPLE_PHASE,
	CIRCULAR_RIPPLE_RADIUS,
	CIRCULAR_RIPPLE_FEATHER,
	CIRCULAR_RIPPLE_TILES_GROUP,   
	CIRCULAR_RIPPLE_X_TILES,
	CIRCULAR_RIPPLE_Y_TILES,
	CIRCULAR_RIPPLE_MIRROR,
	CIRCULAR_RIPPLE_TILES_GROUP_END,   
	CIRCULAR_RIPPLE_NUM_PARAMS
};

enum {
	CENTER_DISK_ID = 1,
	FREQUENCY_DISK_ID,
	STRENGTH_DISK_ID,
	PHASE_DISK_ID,
	RADIUS_DISK_ID,
	FEATHER_DISK_ID,
	X_TILES_DISK_ID,
	Y_TILES_DISK_ID,
	MIRROR_DISK_ID
};

#define	CIRCULAR_RIPPLE_FREQ_MIN		0
#define	CIRCULAR_RIPPLE_FREQ_MAX		100
#define	CIRCULAR_RIPPLE_FREQ_DFLT		20

#define	CIRCULAR_RIPPLE_STRENGTH_MIN	-1
#define	CIRCULAR_RIPPLE_STRENGTH_MAX	1
#define	CIRCULAR_RIPPLE_STRENGTH_DFLT	0.025

#define	CIRCULAR_RIPPLE_PHASE_MIN		-1000
#define	CIRCULAR_RIPPLE_PHASE_MAX		1000
#define	CIRCULAR_RIPPLE_PHASE_DFLT		0

#define	CIRCULAR_RIPPLE_RADIUS_MIN		0
#define	CIRCULAR_RIPPLE_RADIUS_MAX		0.8
#define	CIRCULAR_RIPPLE_RADIUS_DFLT		0.3

#define	CIRCULAR_RIPPLE_FEATHER_MIN		0.001
#define	CIRCULAR_RIPPLE_FEATHER_MAX		1.0
#define	CIRCULAR_RIPPLE_FEATHER_DFLT	0.1


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

typedef struct
{
	PF_Fixed        center_x;
	PF_Fixed        center_y;
	PF_FpLong       frequency;
	PF_FpLong       strength;
	PF_FpLong       phase;
	PF_FpLong       radius;
	PF_FpLong       feather;
	PF_Point        center;
	PF_FpLong       width;
	PF_FpLong       height;
	PF_PixelPtr     src;
	A_long          rowbytes;
	PF_Boolean      x_tiles;
	PF_Boolean      y_tiles;
	PF_Boolean      mirror;
} RippleInfo;


#endif  
