#pragma once
#ifndef ExposureGamma_H
#define ExposureGamma_H

#include "ExposureGammaKernel.cl.h"
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
#include "ExposureGammaKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023 Adobe Inc.\rExposure and Gamma Correction Effect."

#define NAME			"Exposure/Gamma"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	EXPOSUREGAMMA_INPUT = 0,
	EXPOSUREGAMMA_EXPOSURE,
	EXPOSUREGAMMA_GAMMA,
	EXPOSUREGAMMA_OFFSET,
	EXPOSUREGAMMA_NUM_PARAMS
};


#define	EXPOSURE_MIN_VALUE		-2.0
#define	EXPOSURE_MAX_VALUE		2.0
#define	EXPOSURE_MIN_SLIDER		-2.0
#define	EXPOSURE_MAX_SLIDER		2.0
#define	EXPOSURE_DFLT			0.0

#define	GAMMA_MIN_VALUE			0.01
#define	GAMMA_MAX_VALUE			9.99
#define	GAMMA_MIN_SLIDER		0.01
#define	GAMMA_MAX_SLIDER		9.99
#define	GAMMA_DFLT				1.0

#define	OFFSET_MIN_VALUE		-0.9
#define	OFFSET_MAX_VALUE		0.9
#define	OFFSET_MIN_SLIDER		-0.9
#define	OFFSET_MAX_SLIDER		0.9
#define	OFFSET_DFLT				0.0

#define EXPOSURE_DISK_ID        1
#define GAMMA_DISK_ID           2
#define OFFSET_DISK_ID          3

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



typedef struct ExposureGammaInfo {
	PF_FpLong	exposure;
	PF_FpLong	gamma;
	PF_FpLong	offset;
	PF_EffectWorld* input;
} ExposureGammaInfo, * ExposureGammaInfoP, ** ExposureGammaInfoH;

#endif
