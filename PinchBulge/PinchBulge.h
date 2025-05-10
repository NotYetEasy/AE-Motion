#pragma once
#ifndef PinchBulge_H
#define PinchBulge_H

#include "PinchBulgeKernel.cl.h"
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
#include "SDK_Invert_ProcAmp_Kernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2018-2023 Adobe Inc.\rSample PinchBulge GPU effect."

#define NAME			"PinchBulge_GPU"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

enum {
	PINCH_INPUT = 0,
	PINCH_CENTER,
	PINCH_STRENGTH,
	PINCH_RADIUS,
	PINCH_TILES_GROUP,      
	PINCH_X_TILES,
	PINCH_Y_TILES,
	PINCH_MIRROR,
	PINCH_TILES_GROUP_END,   
	PINCH_NUM_PARAMS
};

enum {
	CENTER_DISK_ID = 1,
	STRENGTH_DISK_ID,
	RADIUS_DISK_ID,
	X_TILES_DISK_ID,
	Y_TILES_DISK_ID,
	MIRROR_DISK_ID
};

#define    PINCH_CENTER_X_DFLT    0
#define    PINCH_CENTER_Y_DFLT    0
#define    PINCH_STRENGTH_MIN     -1.00
#define    PINCH_STRENGTH_MAX     1.00
#define    PINCH_STRENGTH_DFLT    0.0
#define    PINCH_RADIUS_MIN       0.0
#define    PINCH_RADIUS_MAX       2.5
#define    PINCH_RADIUS_DFLT      0.3

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

struct PinchInfo {
	PF_InData* in_data;
	PF_FpLong center_x;
	PF_FpLong center_y;
	float strength;
	float radius;
	bool x_tiles;
	bool y_tiles;
	bool mirror;
	PF_EffectWorld* input;
};


typedef struct
{
	float mCenterX;
	float mCenterY;
	float mStrength;
	float mRadius;
	int mXTiles;
	int mYTiles;
	int mMirror;
} PinchBulgeParams;

#endif
