#pragma once
#ifndef LensBlur
#define LensBlur
#include "LensBlurKernel.cl.h"
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
#include "LensBlurKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define STR_CENTER_PARAM      "Center"
#define STR_STRENGTH_PARAM    "Strength"
#define STR_RADIUS_PARAM      "Radius"

#define CENTER_DISK_ID        1
#define STRENGTH_DISK_ID      2
#define RADIUS_DISK_ID        3

#define DESCRIPTION	"\nCopyright 2023 Adobe Inc.\rLens Blur effect that applies a directional blur away from center."

#define NAME			"Lens Blur"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	LENSBLUR_INPUT = 0,
	LENSBLUR_CENTER,
	LENSBLUR_STRENGTH,
	LENSBLUR_RADIUS,
	LENSBLUR_NUM_PARAMS
};


#define	STRENGTH_MIN_VALUE		0
#define	STRENGTH_MAX_VALUE		1
#define	STRENGTH_MIN_SLIDER		0
#define	STRENGTH_MAX_SLIDER		1
#define	STRENGTH_DFLT			0.15

#define	RADIUS_MIN_VALUE		0
#define	RADIUS_MAX_VALUE		1
#define	RADIUS_MIN_SLIDER		0
#define	RADIUS_MAX_SLIDER		1
#define	RADIUS_DFLT				0.15



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
	float strengthF;
	float radiusF;
	int width;       
	int height;      
} BlurInfo;

#endif