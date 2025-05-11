#pragma once
#ifndef Transforms
#define Transforms

#include "TransformsKernel.cl.h"
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
#include "TransformsKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2018-2023 Adobe Inc.\rGPU Accelerated Transform effect."

#define NAME			"DKT_GPU_Transform"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

enum {
	TRANSFORMS_INPUT = 0,
	TRANSFORMS_POSITION,
	TRANSFORMS_ROTATION,
	TRANSFORMS_SCALE,
	TRANSFORMS_GROUP_START,
	TRANSFORMS_X_TILES,
	TRANSFORMS_Y_TILES,
	TRANSFORMS_MIRROR,
	TRANSFORMS_GROUP_END,
	TRANSFORMS_NUM_PARAMS
};

#define POSITION_DISK_ID 1
#define ROTATION_DISK_ID 2
#define SCALE_DISK_ID 3
#define GROUP_START_DISK_ID 4
#define X_TILES_DISK_ID 5
#define Y_TILES_DISK_ID 6
#define MIRROR_DISK_ID 7
#define GROUP_END_DISK_ID 8

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
	float x_pos;
	float y_pos;
	float rotation;
	float scale;
	bool x_tiles;
	bool y_tiles;
	bool mirror;
} TransformParams;

typedef struct TransformInfo {
	PF_Fixed        x_pos;
	PF_Fixed        y_pos;
	PF_Fixed        rotation;
	PF_FpLong       scale;
	bool            x_tiles;
	bool            y_tiles;
	bool            mirror;
	PF_EffectWorld* input_worldP;
} TransformInfo, * TransformInfoP, ** TransformInfoH;

#endif