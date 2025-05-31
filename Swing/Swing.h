#pragma once
#ifndef Swing_H
#define Swing_H

#include "SwingKernel.cl.h"
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
#define HAS_CUDA 1
#else
#include <OpenCL/cl.h>
#define HAS_HLSL 0
#define HAS_METAL 1
#define HAS_CUDA 1
#include <Metal/Metal.h>
#include "SwingKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023 DKT.\rSwing rotation effect."

#define NAME			"Swing"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

enum {
	SWING_INPUT = 0,
	SWING_NORMAL_CHECKBOX,
	SWING_FREQ,
	SWING_ANGLE1,
	SWING_ANGLE2,
	SWING_PHASE,
	SWING_WAVE_TYPE,
	SWING_TILES_GROUP,
	SWING_X_TILES,
	SWING_Y_TILES,
	SWING_MIRROR,
	SWING_TILES_GROUP_END,
	SWING_COMPATIBILITY_GROUP,
	SWING_COMPATIBILITY_CHECKBOX,
	SWING_COMPATIBILITY_FREQUENCY,
	SWING_COMPATIBILITY_ANGLE1,
	SWING_COMPATIBILITY_ANGLE2,
	SWING_COMPATIBILITY_PHASE,
	SWING_COMPATIBILITY_WAVE_TYPE,
	SWING_COMPATIBILITY_GROUP_END,
	SWING_NUM_PARAMS
};

enum {
	FREQ_DISK_ID = 1,
	ANGLE1_DISK_ID,
	ANGLE2_DISK_ID,
	PHASE_DISK_ID,
	WAVE_TYPE_DISK_ID,
	X_TILES_DISK_ID,
	Y_TILES_DISK_ID,
	MIRROR_DISK_ID,
	NORMAL_CHECKBOX_ID,
	COMPATIBILITY_CHECKBOX_ID,
	COMPATIBILITY_FREQUENCY_DISK_ID,
	COMPATIBILITY_ANGLE1_DISK_ID,
	COMPATIBILITY_ANGLE2_DISK_ID,
	COMPATIBILITY_PHASE_DISK_ID,
	COMPATIBILITY_WAVE_TYPE_DISK_ID
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

typedef struct {
	double frequency;
	double angle1;
	double angle2;
	double phase;
	A_long waveType;
	bool x_tiles;
	bool y_tiles;
	bool mirror;
	double current_time;
	double layer_start_seconds;
	double accumulated_phase;
	bool accumulated_phase_initialized;     
	bool normal_enabled;
	bool compatibility_enabled;
	double compat_frequency;
	double compat_angle1;
	double compat_angle2;
	double compat_phase;
	A_long compat_wave_type;
} SwingParams;

struct OpenCLGPUData
{
	cl_kernel swing_kernel;
};

#if HAS_HLSL
#include "DirectXUtils.h"
struct DirectXGPUData
{
	DXContextPtr mContext;
	ShaderObjectPtr mSwingShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState> swing_pipeline;
};
#endif

#endif  
