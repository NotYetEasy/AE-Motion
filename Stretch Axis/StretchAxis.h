#pragma once

#ifndef STRETCH_H
#define STRETCH_H

typedef unsigned char		u_char;
typedef unsigned short		u_short;
typedef unsigned short		u_int16;
typedef unsigned long		u_long;
typedef short int			int16;
#define PF_TABLE_BITS	12
#define PF_TABLE_SZ_16	4096

#define PF_DEEP_COLOR_AWARE 1	// make sure we get 16bpc pixels; 
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

#define	MAJOR_VERSION	1
#define	MINOR_VERSION	0
#define	BUG_VERSION		0
#define	STAGE_VERSION	PF_Stage_DEVELOP
#define	BUILD_VERSION	1


/* Parameter defaults */

#define STRETCH_SCALE_MIN        0.01
#define STRETCH_SCALE_MAX        50.0
#define STRETCH_SCALE_DFLT       1.0

#define STRETCH_ANGLE_MIN        -3600.0
#define STRETCH_ANGLE_MAX        3600.0
#define STRETCH_ANGLE_DFLT       0.0


enum {
	STRETCH_INPUT = 0,
	STRETCH_SCALE,
	STRETCH_ANGLE,
	STRETCH_CONTENT_ONLY,
	STRETCH_NUM_PARAMS
};

enum {
	SCALE_DISK_ID = 1,
	ANGLE_DISK_ID,
	CONTENT_ONLY_DISK_ID,
};

typedef struct StretchInfo {
	PF_FpLong	scale;
	PF_FpLong	angle;
	PF_Boolean	content_only;
	A_long		width;
	A_long		height;
	void* src;          
	A_long		rowbytes;

	// Additional fields for SmartRender support
	PF_EffectWorld* input;    // For SmartRender
} StretchInfo, * StretchInfoP, ** StretchInfoH;


extern "C" {

	DllExport
		PF_Err
		EffectMain(
			PF_Cmd			cmd,
			PF_InData* in_data,
			PF_OutData* out_data,
			PF_ParamDef* params[],
			PF_LayerDef* output,
			void* extra);

}

#endif // STRETCH_H
