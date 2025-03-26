/*
    InnerPinchBulge.h
*/

#pragma once

#ifndef INNER_PINCH_BULGE_H
#define INNER_PINCH_BULGE_H

typedef unsigned char        u_char;
typedef unsigned short        u_short;
typedef unsigned short        u_int16;
typedef unsigned long        u_long;
typedef short int            int16;
#define PF_TABLE_BITS    12
#define PF_TABLE_SZ_16    4096

#define PF_DEEP_COLOR_AWARE 1    // make sure we get 16bpc pixels; 
// AE_Effect.h checks for this.

#include "AEConfig.h"

#ifdef AE_OS_WIN
typedef unsigned short PixelType;
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
#define    MAJOR_VERSION    1
#define    MINOR_VERSION    0
#define    BUG_VERSION        0
#define    STAGE_VERSION    PF_Stage_DEVELOP
#define    BUILD_VERSION    1

/* Parameter defaults */
#define    INNER_PINCH_BULGE_CENTER_X_DFLT    0
#define    INNER_PINCH_BULGE_CENTER_Y_DFLT    0
#define    INNER_PINCH_BULGE_STRENGTH_MIN     -1.00
#define    INNER_PINCH_BULGE_STRENGTH_MAX     1.00
#define    INNER_PINCH_BULGE_STRENGTH_DFLT    0.5
#define    INNER_PINCH_BULGE_RADIUS_MIN       0.0
#define    INNER_PINCH_BULGE_RADIUS_MAX       2.5
#define    INNER_PINCH_BULGE_RADIUS_DFLT      0.3
#define    INNER_PINCH_BULGE_FEATHER_MIN      0.0
#define    INNER_PINCH_BULGE_FEATHER_MAX      1.0
#define    INNER_PINCH_BULGE_FEATHER_DFLT     0.1

/* Parameter indices */
enum {
    INNER_PINCH_BULGE_INPUT = 0,
    INNER_PINCH_BULGE_CENTER,
    INNER_PINCH_BULGE_STRENGTH,
    INNER_PINCH_BULGE_RADIUS,
    INNER_PINCH_BULGE_FEATHER,
    INNER_PINCH_BULGE_GAUSSIAN,
    INNER_PINCH_BULGE_NUM_PARAMS
};

/* Parameter disk IDs */
enum {
    CENTER_DISK_ID = 1,
    STRENGTH_DISK_ID,
    RADIUS_DISK_ID,
    FEATHER_DISK_ID,
    GAUSSIAN_DISK_ID
};

/* Custom data structure for effect */
typedef struct PinchBulgeInfo {
    PF_FpLong            strength;
    PF_FpLong            radius;
    PF_FpLong            feather;
    PF_Boolean           useGaussian;
    PF_Fixed             center_x;
    PF_Fixed             center_y;
    PF_EffectWorld* input;
} PinchBulgeInfo, * PinchBulgeInfoP, ** PinchBulgeInfoH;

// Function declarations
extern "C" {
    DllExport
        PF_Err
        EffectMain(
            PF_Cmd            cmd,
            PF_InData* in_data,
            PF_OutData* out_data,
            PF_ParamDef* params[],
            PF_LayerDef* output,
            void* extra);
}

#endif // INNER_PINCH_BULGE_H
