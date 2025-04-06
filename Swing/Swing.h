#pragma once

#ifndef SWING_H
#define SWING_H

typedef unsigned char        u_char;
typedef unsigned short       u_short;
typedef unsigned short       u_int16;
typedef unsigned long        u_long;
typedef short int            int16;
#define PF_TABLE_BITS    12
#define PF_TABLE_SZ_16   4096

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
#include "AE_EffectSuitesHelper.h"

/* Versioning information */
#define MAJOR_VERSION    1
#define MINOR_VERSION    0
#define BUG_VERSION      0
#define STAGE_VERSION    PF_Stage_DEVELOP
#define BUILD_VERSION    1

/* Parameter defaults */
enum {
    SWING_INPUT = 0,
    SWING_FREQ,
    SWING_ANGLE1,
    SWING_ANGLE2,
    SWING_PHASE,
    SWING_WAVE_TYPE,
    SWING_NUM_PARAMS
};

enum {
    FREQ_DISK_ID = 1,
    ANGLE1_DISK_ID,
    ANGLE2_DISK_ID,
    PHASE_DISK_ID,
    WAVE_TYPE_DISK_ID
};

// String IDs
enum {
    StrID_NONE,
    StrID_Name,
    StrID_Description,
    StrID_Freq_Param_Name,
    StrID_Angle1_Param_Name,
    StrID_Angle2_Param_Name,
    StrID_Phase_Param_Name,
    StrID_Wave_Param_Name,
    StrID_NUMTYPES
};

// String definitions
#define STR(x) #x
#define STRINGIFY(x) STR(x)

#define STR_NONE ""
#define STR_Name "Swing"
#define STR_Description "A swing rotation effect with SmartFX support.\rCopyright 2023 DKT."
#define STR_Freq_Param_Name "Frequency"
#define STR_Angle1_Param_Name "Angle 1"
#define STR_Angle2_Param_Name "Angle 2"
#define STR_Phase_Param_Name "Phase"
#define STR_Wave_Param_Name "Wave"

// Structure to hold effect parameters
struct {
    double frequency;
    double angle1;
    double angle2;
    double phase;
    A_long waveType;
    PF_Rect expanded_rect;
    PF_Point anchor_point;
    A_Matrix4 transform;
    float pre_effect_source_origin_x;
    float pre_effect_source_origin_y;
    float downsample_x;
    float downsample_y;
    bool has_frequency_keyframes;
    A_long time_shift;
    double current_time;
} mix_data;


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

#endif // SWING_H
