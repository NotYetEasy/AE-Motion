#pragma once
#ifndef MotionBlur
#define MotionBlur

#include "MotionBlurKernel.cl.h"
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
#include "SimplexNoise.h"


#if _WIN32
#include <CL/cl.h>
#define HAS_HLSL 1
#define HAS_METAL 0
#else
#include <OpenCL/cl.h>
#define HAS_HLSL 0
#define HAS_METAL 1
#include <Metal/Metal.h>
#include "MotionBlurKernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION "\nCopyright 2025 DKT Effects.\rMotion Blur effect."

#define NAME            "Motion Blur"
#define MAJOR_VERSION   1
#define MINOR_VERSION   0
#define BUG_VERSION     0
#define STAGE_VERSION   PF_Stage_DEVELOP
#define BUILD_VERSION   1

#define STR_NONE            ""
#define STR_NAME            "Motion Blur"
#define STR_DESCRIPTION     "A motion blur effect.\rCopyright 2025 DKT."
#define STR_TUNE_NAME       "Tune"
#define STR_POSITION_NAME   "Position"
#define STR_SCALE_NAME      "Scale"
#define STR_ANGLE_NAME      "Angle"

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

#define TUNE_MIN        0.0
#define TUNE_MAX        4.0
#define TUNE_DFLT       1.0


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
    bool has_motion_prev_curr;
    bool has_scale_change_prev_curr;
    bool has_rotation_prev_curr;
    bool has_motion_curr_next;
    bool has_scale_change_curr_next;
    bool has_rotation_curr_next;
    double motion_x_prev_curr;
    double motion_y_prev_curr;
    double motion_x_curr_next;
    double motion_y_curr_next;
    double scale_x_prev_curr;
    double scale_y_prev_curr;
    double scale_x_curr_next;
    double scale_y_curr_next;
    double rotation_prev_curr;
    double rotation_curr_next;
    bool position_enabled;
    bool scale_enabled;
    bool angle_enabled;
    double tune_value;
    PF_EffectWorld* input_world;
    float scale_velocity;
    AEGP_TwoDVal anchor_point;
    double effect_rotation;
    bool has_effect_rotation;
    PF_InData* in_data;      
} DetectionData;

#endif  
