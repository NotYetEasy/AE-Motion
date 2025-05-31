#pragma once
#ifndef WaveWarp_H
#define WaveWarp_H

#include "WaveWarpKernel.cl.h"
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
#include "WaveWarpKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023.\rWave Warp effect."

#define NAME            "WaveWarp"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	1
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


enum {
    WAVEWARP_INPUT = 0,
    WAVEWARP_PHASE,
    WAVEWARP_ANGLE,
    WAVEWARP_SPACING,
    WAVEWARP_MAGNITUDE,
    WAVEWARP_WARPANGLE,
    WAVEWARP_DAMPING_GROUP_START,
    WAVEWARP_DAMPING,
    WAVEWARP_DAMPINGSPACE,
    WAVEWARP_DAMPINGORIGIN,
    WAVEWARP_SCREENSPACE,
    WAVEWARP_DAMPING_GROUP_END,
    WAVEWARP_TILES_GROUP_START,
    WAVEWARP_XTILES,
    WAVEWARP_YTILES,
    WAVEWARP_MIRROR,
    WAVEWARP_TILES_GROUP_END,
    WAVEWARP_NUM_PARAMS
};


#define WAVEWARP_PHASE_MIN       0.0
#define WAVEWARP_PHASE_MAX       500.0
#define WAVEWARP_PHASE_DFLT      0.0

#define WAVEWARP_ANGLE_DFLT      0.0

#define WAVEWARP_SPACING_MIN     0.0
#define WAVEWARP_SPACING_MAX     500.0
#define WAVEWARP_SPACING_DFLT    20.0

#define WAVEWARP_MAGNITUDE_MIN   0.0
#define WAVEWARP_MAGNITUDE_MAX   30.0
#define WAVEWARP_MAGNITUDE_DFLT  4.0

#define WAVEWARP_WARPANGLE_DFLT  90.0

#define WAVEWARP_DAMPING_MIN     -1.0
#define WAVEWARP_DAMPING_MAX     1.0
#define WAVEWARP_DAMPING_DFLT    0.0

#define WAVEWARP_DAMPINGSPACE_MIN    -1.0
#define WAVEWARP_DAMPINGSPACE_MAX    1.0
#define WAVEWARP_DAMPINGSPACE_DFLT   0.0

#define WAVEWARP_DAMPINGORIGIN_MIN   0.0
#define WAVEWARP_DAMPINGORIGIN_MAX   1.0
#define WAVEWARP_DAMPINGORIGIN_DFLT  0.5


extern "C" {

    DllExport
        PF_Err
        EffectMain(
            PF_Cmd          cmd,
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
    float phase;
    float direction;
    float spacing;
    float magnitude;
    float warpAngle;
    float damping;
    float dampingSpace;
    float dampingOrigin;
    int screenSpace;
    int xTiles;
    int yTiles;
    int mirror;
} WaveWarpParams;

typedef struct WaveWarpInfo {
    PF_FpLong   phase;
    PF_FpLong   direction;
    PF_FpLong   spacing;
    PF_FpLong   magnitude;
    PF_FpLong   offset;
    PF_FpLong   damping;
    PF_FpLong   dampingSpace;
    PF_FpLong   dampingOrigin;
    PF_Boolean  screenSpace;
    PF_Boolean  xTiles;
    PF_Boolean  yTiles;
    PF_Boolean  mirror;
    A_long      width;
    A_long      height;
    A_long      rowbytes;
    void* srcData;
    PF_InData* in_data;
    PF_FpLong   cos_a1;
    PF_FpLong   sin_a1;
    PF_FpLong   warp_angle_rad;
    PF_FpLong   original_direction;
    PF_FpLong   original_offset;
} WaveWarpInfo, * WaveWarpInfoP, ** WaveWarpInfoH;

#if HAS_CUDA
extern void WaveWarp_CUDA(
    float const* src,
    float* dst,
    unsigned int srcPitch,
    unsigned int dstPitch,
    int is16f,
    unsigned int width,
    unsigned int height,
    float phase,
    float direction,
    float spacing,
    float magnitude,
    float warpAngle,
    float damping,
    float dampingSpace,
    float dampingOrigin,
    int screenSpace,
    int xTiles,
    int yTiles,
    int mirror);
#endif

#endif  
