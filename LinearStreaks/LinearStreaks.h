#pragma once
#ifndef LinearStreaks
#define LinearStreaks

#include "LinearStreaksKernel.cl.h"
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
#include "LinearStreaksKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2018-2023 Adobe Inc.\rLinear streaks effect."

#define NAME			"LinearStreaks"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	1
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
	LINEAR_STREAKS_INPUT = 0,
	LINEAR_STREAKS_STRENGTH,
	LINEAR_STREAKS_ANGLE,
	LINEAR_STREAKS_ALPHA,
	LINEAR_STREAKS_BIAS,
	LINEAR_STREAKS_R_MODE,
	LINEAR_STREAKS_G_MODE,
	LINEAR_STREAKS_B_MODE,
	LINEAR_STREAKS_A_MODE,
	LINEAR_STREAKS_NUM_PARAMS
};

#define	STREAKS_STRENGTH_MIN		0.0f
#define	STREAKS_STRENGTH_MAX		1.0f
#define	STREAKS_STRENGTH_DFLT		0.15f

#define STREAKS_ANGLE_DFLT          0.0f

#define STREAKS_ALPHA_MIN           0.0f
#define STREAKS_ALPHA_MAX           1.0f
#define STREAKS_ALPHA_DFLT          1.0f

#define STREAKS_BIAS_MIN            -1.0f
#define STREAKS_BIAS_MAX            1.0f
#define STREAKS_BIAS_DFLT           0.0f

#define STREAKS_MODE_MIN            0
#define STREAKS_MODE_MAX            1
#define STREAKS_MODE_AVG            2
#define STREAKS_MODE_OFF            3

enum {
	STRENGTH_DISK_ID = 1,
	ANGLE_DISK_ID,
	ALPHA_DISK_ID,
	BIAS_DISK_ID,
	R_MODE_DISK_ID,
	G_MODE_DISK_ID,
	B_MODE_DISK_ID,
	A_MODE_DISK_ID
};

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
	void* inputP;              
	A_long inputWidth;        
	A_long inputHeight;       
	A_long inputRowBytes;      
	A_long outputWidth;       
	A_long outputHeight;      
	float strength;
	float angle;
	float alpha;
	float bias;
	A_long rmode;
	A_long gmode;
	A_long bmode;
	A_long amode;
} LinearStreaksInfo, * LinearStreaksInfoP, ** LinearStreaksInfoH;

struct vec4 {
	float r, g, b, a;
};

template<typename PixelT>
inline PixelT* GetPixelAddress(void* baseAddr, int x, int y, int rowBytes) {
	return reinterpret_cast<PixelT*>(
		reinterpret_cast<char*>(baseAddr) + y * rowBytes + x * sizeof(PixelT)
		);
}

#endif  