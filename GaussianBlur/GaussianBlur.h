#pragma once
#ifndef GaussianBlur
#define GaussianBlur

#include "GaussianBlurKernel.cl.h"
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
#include "GaussianBlurKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023 Adobe Inc.\rGaussian Blur effect."

#define NAME			"GaussianBlur"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	GAUSSIANBLUR_INPUT = 0,
	GAUSSIANBLUR_STRENGTH,
	GAUSSIANBLUR_NUM_PARAMS
};


#define	GAUSSIANBLUR_STRENGTH_MIN		0.0
#define	GAUSSIANBLUR_STRENGTH_MAX		2.0
#define	GAUSSIANBLUR_STRENGTH_MIN_SLIDER	0.0
#define	GAUSSIANBLUR_STRENGTH_MAX_SLIDER	2.0
#define	GAUSSIANBLUR_STRENGTH_DFLT		0.15


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
	float mStrength;
	float strengthF;
	PF_EffectWorld* inputImg;
	PF_EffectWorld* hblurImg;
	A_long comp_width;
	A_long comp_height;
	PF_RationalScale downsample_x;
	PF_RationalScale downsample_y;
} BlurInfo;

#endif