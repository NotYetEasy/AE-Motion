#pragma once

#ifndef FRACTALWARP_H
#define FRACTALWARP_H

#ifndef PF_TABLE_BITS
#define PF_TABLE_BITS 12
#endif

#include "AEConfig.h"

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_Macros.h"
#include "Param_Utils.h"
#include "AE_EffectCBSuites.h"
#include "String_Utils.h"
#include "AE_GeneralPlug.h"
#include "AEFX_ChannelDepthTpl.h"
#include "AEGP_SuiteHandler.h"

/* Versioning information */
#define MAJOR_VERSION    1
#define MINOR_VERSION    0
#define BUG_VERSION      0
#define STAGE_VERSION    PF_Stage_DEVELOP
#define BUILD_VERSION    800E4

/* Parameter defaults */
#define FRACTALWARP_MAGNITUDE_MIN        -5.0
#define FRACTALWARP_MAGNITUDE_MAX        5.0
#define FRACTALWARP_MAGNITUDE_DFLT       0.2

#define FRACTALWARP_DETAIL_MIN           0.1
#define FRACTALWARP_DETAIL_MAX           4.0
#define FRACTALWARP_DETAIL_DFLT           1.0

#define FRACTALWARP_LACUNARITY_MIN        0.25
#define FRACTALWARP_LACUNARITY_MAX        0.75
#define FRACTALWARP_LACUNARITY_DFLT       0.5

#define FRACTALWARP_OCTAVES_MIN           1
#define FRACTALWARP_OCTAVES_MAX           9
#define FRACTALWARP_OCTAVES_DFLT          6

enum {
    FRACTALWARP_INPUT = 0,
    FRACTALWARP_POSITION,
    FRACTALWARP_PARALLAX,
    FRACTALWARP_MAGNITUDE,
    FRACTALWARP_DETAIL,
    FRACTALWARP_LACUNARITY,
    FRACTALWARP_SCREENSPACE,
    FRACTALWARP_OCTAVES,
    FRACTALWARP_NUM_PARAMS
};

enum {
    POSITION_DISK_ID = 1,
    PARALLAX_DISK_ID,
    MAGNITUDE_DISK_ID,
    DETAIL_DISK_ID,
    LACUNARITY_DISK_ID,
    SCREENSPACE_DISK_ID,
    OCTAVES_DISK_ID
};

#ifndef PF_WORLD_IS_FLOAT
#define PF_WORLD_IS_FLOAT(W) ((W)->world_flags & PF_WorldFlag_FLOAT)
#endif

typedef struct {
    A_FpLong    x;
    A_FpLong    y;
} PointF;

typedef struct FractalWarpInfo {
    PointF      position;
    PointF      parallax;
    A_FpLong    magnitude;
    A_FpLong    detail;
    A_FpLong    lacunarity;
    PF_Boolean  screenSpace;
    A_long      octaves;
    A_long      width;
    A_long      height;
    void* inputP;
    A_long      rowbytes;
    float       sceneSizeX;
    float       sceneSizeY;
    PF_WorldTransformSuite1* world_transformP; // added this
} FractalWarpInfo, * FractalWarpInfoP, ** FractalWarpInfoH;


static float Fract(float x);
static float Hash(float p);
static float Hash(float x, float y);
static float Noise(float x, float y);
static float FBM(float x, float y, float px, float py, int octaveCount, float intensity);

static PF_FixedPoint FloatToFixed(float value);
static float FixedToFloat(PF_FixedPoint value);

extern "C" {
    DllExport PF_Err EffectMain(
        PF_Cmd          cmd,
        PF_InData* in_data,
        PF_OutData* out_data,
        PF_ParamDef* params[],
        PF_LayerDef* output,
        void* extra);
}

#endif // FRACTALWARP_H
