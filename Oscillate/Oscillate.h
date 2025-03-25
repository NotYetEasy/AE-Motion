#pragma once
#ifndef RANDOMMOVE_H
#define RANDOMMOVE_H

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
#define ANGLE_MIN       -3600.0
#define ANGLE_MAX       3600.0
#define ANGLE_DFLT      45.0

#define FREQUENCY_MIN   0.0
#define FREQUENCY_MAX   16.0
#define FREQUENCY_DFLT  2.0

#define MAGNITUDE_MIN   0
#define MAGNITUDE_MAX   4000
#define MAGNITUDE_DFLT  25

#define PHASE_MIN       0.0
#define PHASE_MAX       1000.0
#define PHASE_DFLT      0.0

enum {
    RANDOMMOVE_INPUT = 0,
    DIRECTION_SLIDER,
    ANGLE_SLIDER,
    FREQUENCY_SLIDER,
    MAGNITUDE_SLIDER,
    WAVE_TYPE_SLIDER,
    PHASE_SLIDER,
    RANDOMMOVE_NUM_PARAMS
};

typedef struct RandomMoveInfo {
    A_long    direction;
    PF_FpLong angle;
    PF_FpLong frequency;
    PF_FpLong magnitude;
    A_long    wave_type;
    PF_FpLong phase;
} RandomMoveInfo;


extern "C" {
    DllExport PF_Err EffectMain(
        PF_Cmd          cmd,
        PF_InData       *in_data,
        PF_OutData      *out_data,
        PF_ParamDef     *params[],
        PF_LayerDef     *output,
        void            *extra);
}

#endif // RANDOMMOVE_H

