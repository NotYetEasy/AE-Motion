/*
    PinchBulge.h
*/

#pragma once

#ifndef PINCHBULGE_H
#define PINCHBULGE_H

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
#define    PINCH_CENTER_X_DFLT    0
#define    PINCH_CENTER_Y_DFLT    0
#define    PINCH_STRENGTH_MIN     -1.00
#define    PINCH_STRENGTH_MAX     1.00
#define    PINCH_STRENGTH_DFLT    0.0
#define    PINCH_RADIUS_MIN       0.0
#define    PINCH_RADIUS_MAX       2.5
#define    PINCH_RADIUS_DFLT      0.3

/* Parameter indices */
enum {
    PINCH_INPUT = 0,
    PINCH_CENTER,
    PINCH_STRENGTH,
    PINCH_RADIUS,
    PINCH_NUM_PARAMS
};

/* Parameter disk IDs */
enum {
    CENTER_DISK_ID = 1,
    STRENGTH_DISK_ID,
    RADIUS_DISK_ID
};

/* String IDs */
enum {
    StrID_NONE,
    StrID_Name,
    StrID_Description,
    StrID_Center_Param_Name,
    StrID_Strength_Param_Name,
    StrID_Radius_Param_Name,
    StrID_NUMTYPES
};

/* String table */
#define STR(id) (GetStringPtr(id))

/* Custom data structure for effect */
typedef struct PinchInfo {
    PF_FpLong            strength;
    PF_FpLong            radius;
    PF_Fixed             center_x;
    PF_Fixed             center_y;
    PF_EffectWorld* input;
} PinchInfo, * PinchInfoP, ** PinchInfoH;

// Add this to PinchBulge.h
typedef struct {
    PF_EffectWorld* input;
    PF_EffectWorld* output;
    float strength;
    float radius;
    float centerX;
    float centerY;
    int startY;
    int endY;
} ThreadData;

// Function declarations
void ProcessFloatPixelsThreaded(void* refcon, A_long thread_index, A_long thread_count, A_long y_start, A_long y_stop);


/* Function prototypes */
char* GetStringPtr(int strNum);

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

#endif // PINCHBULGE_H
