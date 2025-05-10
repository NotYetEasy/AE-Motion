#pragma once
#ifndef Squeeze
#define Squeeze

#include "SqueezeKernel.cl.h"
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
#include "SqueezeKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nDistorts an image by squeezing it horizontally or vertically."

#define NAME			"Squeeze"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	SQUEEZE_INPUT = 0,
	SQUEEZE_STRENGTH,
	SQUEEZE_TILES_GROUP,
	SQUEEZE_X_TILES,
	SQUEEZE_Y_TILES,
	SQUEEZE_MIRROR,
	SQUEEZE_TILES_GROUP_END,
	SQUEEZE_NUM_PARAMS
};


#define	SQUEEZE_STRENGTH_MIN		-2.0
#define	SQUEEZE_STRENGTH_MAX		2.0
#define	SQUEEZE_STRENGTH_DFLT		0.0


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
	float strength;
	A_long width;
	A_long height;
	A_long rowbytes;
	void* src;
	PF_EffectWorld* input;
	PF_Boolean x_tiles;
	PF_Boolean y_tiles;
	PF_Boolean mirror;
} SqueezeInfo;

#endif