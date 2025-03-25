#pragma once

#ifndef WAVEWARP_H
#define WAVEWARP_H

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

/* Versioning information */
#define MAJOR_VERSION    1
#define MINOR_VERSION    1
#define BUG_VERSION      0
#define STAGE_VERSION    PF_Stage_DEVELOP
#define BUILD_VERSION    1

/* Parameter defaults */
#define WAVEWARP_PHASE_MIN       0.0
#define WAVEWARP_PHASE_MAX       500.0
#define WAVEWARP_PHASE_DFLT      0.0

#define WAVEWARP_ANGLE_DFLT  0.0

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

enum {
    WAVEWARP_INPUT = 0,
    WAVEWARP_PHASE,
    WAVEWARP_ANGLE,
    WAVEWARP_SPACING,
    WAVEWARP_MAGNITUDE,
    WAVEWARP_WARPANGLE,
    WAVEWARP_DAMPING,
    WAVEWARP_DAMPINGSPACE,
    WAVEWARP_DAMPINGORIGIN,
    WAVEWARP_SCREENSPACE,
    WAVEWARP_NUM_PARAMS
};

enum {
    PHASE_DISK_ID = 1,
    ANGLE_DISK_ID,
    SPACING_DISK_ID,
    MAGNITUDE_DISK_ID,
    WARPANGLE_DISK_ID,
    DAMPING_DISK_ID,
    DAMPINGSPACE_DISK_ID,
    DAMPINGORIGIN_DISK_ID,
    SCREENSPACE_DISK_ID
};

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
    A_long      width;
    A_long      height;
    A_long      rowbytes;
    void* srcData;    

  
    PF_FpLong   cos_a1;
    PF_FpLong   sin_a1;

   
    PF_FpLong   warp_angle_rad;

   
    PF_FpLong   original_direction;
    PF_FpLong   original_offset;
} WaveWarpInfo, * WaveWarpInfoP, ** WaveWarpInfoH;



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

#endif // WAVEWARP_H
