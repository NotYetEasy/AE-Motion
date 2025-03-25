/*
	RandomDisplacement.h
*/

#pragma once

#ifndef RANDOM_DISPLACEMENT_H
#define RANDOM_DISPLACEMENT_H

#ifndef PF_TABLE_BITS
    #define PF_TABLE_BITS 12
#endif

#define PF_DEEP_COLOR_AWARE 1	
								

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

#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


/* Parameter defaults */

#define	RANDOM_DISPLACEMENT_MAGNITUDE_MIN		0
#define	RANDOM_DISPLACEMENT_MAGNITUDE_MAX		2000
#define	RANDOM_DISPLACEMENT_MAGNITUDE_DFLT		50

#define	RANDOM_DISPLACEMENT_EVOLUTION_MIN		0
#define	RANDOM_DISPLACEMENT_EVOLUTION_MAX		2000
#define	RANDOM_DISPLACEMENT_EVOLUTION_DFLT		0

#define	RANDOM_DISPLACEMENT_SEED_MIN			0
#define	RANDOM_DISPLACEMENT_SEED_MAX			5
#define	RANDOM_DISPLACEMENT_SEED_DFLT			0

#define	RANDOM_DISPLACEMENT_SCATTER_MIN			0
#define	RANDOM_DISPLACEMENT_SCATTER_MAX			2
#define	RANDOM_DISPLACEMENT_SCATTER_DFLT		0.5

enum {
	RANDOM_DISPLACEMENT_INPUT = 0,
	RANDOM_DISPLACEMENT_MAGNITUDE,
	RANDOM_DISPLACEMENT_EVOLUTION,
	RANDOM_DISPLACEMENT_SEED,
	RANDOM_DISPLACEMENT_SCATTER,
	RANDOM_DISPLACEMENT_NUM_PARAMS
};

enum {
	MAGNITUDE_DISK_ID = 1,
	EVOLUTION_DISK_ID,
	SEED_DISK_ID,
	SCATTER_DISK_ID
};

typedef struct DisplacementInfo {
	PF_FpLong	magnitude;
	PF_FpLong	evolution;
	PF_FpLong	seed;
	PF_FpLong	scatter;
} DisplacementInfo, *DisplacementInfoP, **DisplacementInfoH;

extern "C" {
    DllExport PF_Err EffectMain(
        PF_Cmd          cmd,
        PF_InData       *in_data,
        PF_OutData      *out_data,
        PF_ParamDef     *params[],
        PF_LayerDef     *output,
        void            *extra);
        
    DllExport PF_Err PluginDataEntryFunction(
        PF_PluginDataPtr inPtr,
        PF_PluginDataCB inPluginDataCallBackPtr,
        SPBasicSuite* inSPBasicSuitePtr,
        const char* inHostName,
        const char* inHostVersion);
}

#endif // RANDOM_DISPLACEMENT_H


