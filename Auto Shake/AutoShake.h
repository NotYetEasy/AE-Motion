#pragma once
#ifndef AUTOSHAKE_H
#define AUTOSHAKE_H

typedef unsigned char        u_char;
typedef unsigned short       u_short;
typedef unsigned short       u_int16;
typedef unsigned long        u_long;
typedef short int            int16;
#define PF_TABLE_BITS    12
#define PF_TABLE_SZ_16   4096

#define PF_DEEP_COLOR_AWARE 1   // make sure we get 16bpc pixels; 
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

#define MAJOR_VERSION    1
#define MINOR_VERSION    0
#define BUG_VERSION      0
#define STAGE_VERSION    PF_Stage_DEVELOP
#define BUILD_VERSION    1


/* Parameter defaults */

#define AUTOSHAKE_MAGNITUDE_MIN      0
#define AUTOSHAKE_MAGNITUDE_MAX      2000
#define AUTOSHAKE_MAGNITUDE_DFLT     50

#define AUTOSHAKE_FREQUENCY_MIN      0
#define AUTOSHAKE_FREQUENCY_MAX      16
#define AUTOSHAKE_FREQUENCY_DFLT     2.0

#define AUTOSHAKE_EVOLUTION_MIN      0
#define AUTOSHAKE_EVOLUTION_MAX      2000
#define AUTOSHAKE_EVOLUTION_DFLT     0

#define AUTOSHAKE_SEED_MIN           0
#define AUTOSHAKE_SEED_MAX           5
#define AUTOSHAKE_SEED_DFLT          0

#define AUTOSHAKE_ANGLE_DFLT         45.0

#define AUTOSHAKE_SLACK_MIN          0
#define AUTOSHAKE_SLACK_MAX          100
#define AUTOSHAKE_SLACK_DFLT         25

#define AUTOSHAKE_ZSHAKE_MIN         0
#define AUTOSHAKE_ZSHAKE_MAX         2000
#define AUTOSHAKE_ZSHAKE_DFLT        0

enum {
    AUTOSHAKE_INPUT = 0,
    AUTOSHAKE_MAGNITUDE,
    AUTOSHAKE_FREQUENCY,
    AUTOSHAKE_EVOLUTION,
    AUTOSHAKE_SEED,
    AUTOSHAKE_ANGLE,
    AUTOSHAKE_SLACK,
    AUTOSHAKE_ZSHAKE,
    AUTOSHAKE_NUM_PARAMS
};

enum {
    MAGNITUDE_DISK_ID = 1,
    FREQUENCY_DISK_ID,
    EVOLUTION_DISK_ID,
    SEED_DISK_ID,
    ANGLE_DISK_ID,
    SLACK_DISK_ID,
    ZSHAKE_DISK_ID,
};

typedef struct ShakeInfo {
    PF_FpLong magnitude;
    PF_FpLong frequency;
    PF_FpLong evolution;
    PF_FpLong seed;
    PF_FpLong angle;
    PF_FpLong slack;
    PF_FpLong zshake;
} ShakeInfo, * ShakeInfoP, ** ShakeInfoH;


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

#endif // AUTOSHAKE_H
