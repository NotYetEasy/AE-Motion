#ifndef ZoomBlur_H
#define ZoomBlur_H

#include "ZoomBlurKernel.cl.h"
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
#include "ZoomBlur_Kernel.metal.h"
#endif
#include <math.h>

#ifdef AE_OS_WIN
#include <Windows.h>
#endif

#define DESCRIPTION	"\nCopyright 2023 Adobe Inc.\rZoom Blur effect."

#define NAME			"Zoom Blur"
#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1

enum {
    ZOOMBLUR_INPUT = 0,
    ZOOMBLUR_STRENGTH,
    ZOOMBLUR_CENTER,
    ZOOMBLUR_ADAPTIVE,
    ZOOMBLUR_NUM_PARAMS
};

enum {
    STRENGTH_DISK_ID = 1,
    CENTER_DISK_ID,
    ADAPTIVE_DISK_ID
};

#define	ZOOMBLUR_STRENGTH_MIN		0
#define	ZOOMBLUR_STRENGTH_MAX		1
#define	ZOOMBLUR_STRENGTH_DFLT		0.15

#define STR_NAME "Zoom Blur"
#define STR_DESCRIPTION "Applies a radial blur to the layer, as though zooming in or out."
#define STR_STRENGTH_PARAM_NAME "Strength"
#define STR_CENTER_PARAM_NAME "Center"
#define STR_ADAPTIVE_PARAM_NAME "Adaptive"

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define CLAMP(x, min, max) (MIN(MAX(x, min), max))

typedef struct {
    PF_Point    center;
    PF_FpLong   strength;
    PF_Boolean  adaptive;
    A_long      width;
    A_long      height;
    void* src_data;
    A_long      rowbytes;
} ZoomBlurInfo;

typedef struct {
    int         mSrcPitch;
    int         mDstPitch;
    int         m16f;
    int         mWidth;
    int         mHeight;
    float       mStrength;
    float       mCenterX;
    float       mCenterY;
    int         mAdaptive;
} ZoomBlurParams;

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

#endif  