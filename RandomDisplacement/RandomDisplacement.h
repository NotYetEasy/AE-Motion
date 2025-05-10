#pragma once
#ifndef RandomDisplacement_H
#define RandomDisplacement_H

#include "RandomDisplacementKernel.cl.h"
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
#include "RandomDisplacement_Kernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023.\rRandom Displacement effect."

#define NAME			"Random Displacement"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	RANDOM_DISPLACEMENT_INPUT = 0,
	RANDOM_DISPLACEMENT_MAGNITUDE,
	RANDOM_DISPLACEMENT_EVOLUTION,
	RANDOM_DISPLACEMENT_SEED,
	RANDOM_DISPLACEMENT_SCATTER,
	TILES_GROUP,          
	X_TILES_DISK_ID,
	Y_TILES_DISK_ID,
	MIRROR_DISK_ID,
	TILES_GROUP_END,      
	RANDOM_DISPLACEMENT_NUM_PARAMS
};


#define	RANDOM_DISPLACEMENT_MAGNITUDE_MIN		0
#define	RANDOM_DISPLACEMENT_MAGNITUDE_MAX		2000
#define	RANDOM_DISPLACEMENT_MAGNITUDE_DFLT		50

#define	RANDOM_DISPLACEMENT_EVOLUTION_MIN		0
#define	RANDOM_DISPLACEMENT_EVOLUTION_MAX		2000
#define	RANDOM_DISPLACEMENT_EVOLUTION_DFLT		0

#define	RANDOM_DISPLACEMENT_SEED_MIN			0
#define	RANDOM_DISPLACEMENT_SEED_MAX			5
#define	RANDOM_DISPLACEMENT_SEED_DFLT			0

#define	RANDOM_DISPLACEMENT_SCATTER_MIN			0
#define	RANDOM_DISPLACEMENT_SCATTER_MAX			2
#define	RANDOM_DISPLACEMENT_SCATTER_DFLT		0.5


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

typedef struct DisplacementInfo {
	PF_FpLong	magnitude;
	PF_FpLong	evolution;
	PF_FpLong	seed;
	PF_FpLong	scatter;
	PF_Boolean  x_tiles;       
	PF_Boolean  y_tiles;       
	PF_Boolean  mirror;       
	A_long      width;        
	A_long      height;       
} DisplacementInfo, * DisplacementInfoP, ** DisplacementInfoH;


#endif
