#pragma once
#ifndef Swirl
#define Swirl

#include "SwirlKernel.cl.h"
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
#include "SwirlKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023 DKT Effects.\rSwirl effect with tiling options."

#define NAME			"Swirl"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	1
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

enum {
	SWIRL_INPUT = 0,
	SWIRL_CENTER,
	SWIRL_STRENGTH,
	SWIRL_RADIUS,
	SWIRL_TILES_GROUP,     
	SWIRL_X_TILES,        
	SWIRL_Y_TILES,        
	SWIRL_MIRROR,         
	SWIRL_TILES_GROUP_END,   
	SWIRL_NUM_PARAMS
};

enum {
	CENTER_DISK_ID = 1,
	STRENGTH_DISK_ID,
	RADIUS_DISK_ID,
	X_TILES_DISK_ID,       
	Y_TILES_DISK_ID,       
	MIRROR_DISK_ID,        
};

#define STR_NAME          "Swirl"
#define STR_DESCRIPTION   "A vortex-like deformation. Strength, radius, and center can be adjusted."
#define STR_CENTER_PARAM  "Center"
#define STR_STRENGTH_PARAM "Strength"
#define STR_RADIUS_PARAM  "Radius"
#define STR_X_TILES_PARAM  "X Tiles"
#define STR_Y_TILES_PARAM  "Y Tiles"
#define STR_MIRROR_PARAM   "Mirror"

#define SWIRL_STRENGTH_MIN   -0.5
#define SWIRL_STRENGTH_MAX   0.5
#define SWIRL_STRENGTH_DFLT  0.1

#define SWIRL_RADIUS_MIN     0.0
#define SWIRL_RADIUS_MAX     0.8
#define SWIRL_RADIUS_DFLT    0.3

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
	float centerX;
	float centerY;
	float strength;
	float radius;
	int x_tiles;
	int y_tiles;
	int mirror;
	int width;
	int height;
	PF_EffectWorld* inputP;  
} SwirlParams;

#endif