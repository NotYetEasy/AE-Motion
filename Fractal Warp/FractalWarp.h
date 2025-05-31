#pragma once
#ifndef FractalWarp_H
#define FractalWarp_H

#include "FractalWarpKernel.cl.h"
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
#include "FractalWarpKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023\rFractal Warp effect."

#define NAME			"Fractal Warp"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

enum {
    POSITION_DISK_ID = 1,
    PARALLAX_DISK_ID,
    MAGNITUDE_DISK_ID,
    DETAIL_DISK_ID,
    LACUNARITY_DISK_ID,
    SCREENSPACE_DISK_ID,
    OCTAVES_DISK_ID,
    X_TILES_DISK_ID,
    Y_TILES_DISK_ID,
    MIRROR_DISK_ID
};

enum {
    FRACTALWARP_INPUT = 0,
    FRACTALWARP_POSITION,
    FRACTALWARP_PARALLAX,
    FRACTALWARP_MAGNITUDE,
    FRACTALWARP_DETAIL,
    FRACTALWARP_LACUNARITY,
    FRACTALWARP_SCREENSPACE,
    FRACTALWARP_OCTAVES,
    FRACTALWARP_TILES_GROUP,
    FRACTALWARP_X_TILES,
    FRACTALWARP_Y_TILES,
    FRACTALWARP_MIRROR,
    FRACTALWARP_TILES_GROUP_END,
    FRACTALWARP_NUM_PARAMS
};

#define FRACTALWARP_MAGNITUDE_MIN        -5.0
#define FRACTALWARP_MAGNITUDE_MAX        5.0
#define FRACTALWARP_MAGNITUDE_DFLT       0.2

#define FRACTALWARP_DETAIL_MIN           0.1
#define FRACTALWARP_DETAIL_MAX           4.0
#define FRACTALWARP_DETAIL_DFLT          1.0

#define FRACTALWARP_LACUNARITY_MIN       0.25
#define FRACTALWARP_LACUNARITY_MAX       0.75
#define FRACTALWARP_LACUNARITY_DFLT      0.5

#define FRACTALWARP_OCTAVES_MIN          1
#define FRACTALWARP_OCTAVES_MAX          9
#define FRACTALWARP_OCTAVES_DFLT         6

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

typedef struct {
    float matrix[16];         
} TransformMatrix;

typedef struct {
    TransformMatrix textureToScreen;           
    TransformMatrix screenToLayer;              
    TransformMatrix layerToScreen;             
} CoordinateTransforms;

typedef struct
{
    float positionX;
    float positionY;
    float parallaxX;
    float parallaxY;
    float magnitude;
    float detail;
    float lacunarity;
    int screenSpace;
    int octaves;
    int x_tiles;
    int y_tiles;
    int mirror;
    A_long width;
    A_long height;
    void* inputP;
    A_long rowbytes;
    CoordinateTransforms transforms;
} FractalWarpParams;


#endif  
