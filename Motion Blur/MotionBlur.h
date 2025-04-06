/*
    MotionBlur.h
*/

#pragma once

#ifndef MOTIONBLUR_H
#define MOTIONBLUR_H

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
#include "SimplexNoise.h"

/* Versioning information */

#define    MAJOR_VERSION    1
#define    MINOR_VERSION    0
#define    BUG_VERSION        0
#define    STAGE_VERSION    PF_Stage_DEVELOP
#define    BUILD_VERSION    1

/* String definitions */
#define STR_NONE            ""
#define STR_NAME            "Motion Blur"
#define STR_DESCRIPTION     "A motion blur effect.\rCopyright 2025 DKT."
#define STR_TUNE_NAME       "Tune"
#define STR_POSITION_NAME   "Position"
#define STR_SCALE_NAME      "Scale"
#define STR_ANGLE_NAME      "Angle"

/* Parameter defaults */

#define    TUNE_MIN        0
#define    TUNE_MAX        4
#define    TUNE_DFLT        1

enum {
    MOTIONBLUR_INPUT = 0,
    MOTIONBLUR_TUNE,
    MOTIONBLUR_POSITION,
    MOTIONBLUR_SCALE,
    MOTIONBLUR_ANGLE,
    MOTIONBLUR_NUM_PARAMS
};

enum {
    TUNE_DISK_ID = 1,
    POSITION_DISK_ID,
    SCALE_DISK_ID,
    ANGLE_DISK_ID,
};

typedef struct MotionBlurInfo{
    PF_FpLong    tuneF;
    PF_Boolean   position;
    PF_Boolean   scale;
    PF_Boolean   angle;
    PF_EffectWorld *input;  // Added to store the input layer
    PF_Handle    sequence_data; // Added to store sequence data
    PF_InData    *in_data;  // Added to access in_data in pixel functions
} MotionBlurInfo, *MotionBlurInfoP, **MotionBlurInfoH;


extern "C" {

    DllExport
    PF_Err
    EffectMain(
        PF_Cmd            cmd,
        PF_InData        *in_data,
        PF_OutData        *out_data,
        PF_ParamDef        *params[],
        PF_LayerDef        *output,
        void            *extra);

}

#endif // MOTIONBLUR_H
