#pragma once

#ifndef SQUEEZE_H
#define SQUEEZE_H

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

#define	SQUEEZE_STRENGTH_MIN		-2.0
#define	SQUEEZE_STRENGTH_MAX		2.0
#define	SQUEEZE_STRENGTH_DFLT		0.0

enum {
	SQUEEZE_INPUT = 0,
	SQUEEZE_STRENGTH,
	SQUEEZE_NUM_PARAMS
};

enum {
	STRENGTH_DISK_ID = 1,
};

// Define helper functions for type safety
inline A_u_char getRedVal(const PF_Pixel8* pixel) { return pixel->red; }
inline A_u_char getGreenVal(const PF_Pixel8* pixel) { return pixel->green; }
inline A_u_char getBlueVal(const PF_Pixel8* pixel) { return pixel->blue; }
inline A_u_char getAlphaVal(const PF_Pixel8* pixel) { return pixel->alpha; }

inline A_u_short getRedVal(const PF_Pixel16* pixel) { return pixel->red; }
inline A_u_short getGreenVal(const PF_Pixel16* pixel) { return pixel->green; }
inline A_u_short getBlueVal(const PF_Pixel16* pixel) { return pixel->blue; }
inline A_u_short getAlphaVal(const PF_Pixel16* pixel) { return pixel->alpha; }

inline PF_FpShort getRedVal(const PF_PixelFloat* pixel) { return pixel->red; }
inline PF_FpShort getGreenVal(const PF_PixelFloat* pixel) { return pixel->green; }
inline PF_FpShort getBlueVal(const PF_PixelFloat* pixel) { return pixel->blue; }
inline PF_FpShort getAlphaVal(const PF_PixelFloat* pixel) { return pixel->alpha; }

typedef struct SqueezeInfo {
	PF_FpLong	strength;
	A_long		width;
	A_long		height;
	A_long      rowbytes;
	PF_Pixel8* src8P;
	PF_Pixel16* src16P;
	PF_PixelFloat* srcFloatP;
	PF_EffectWorld* input;
} SqueezeInfo, * SqueezeInfoP, ** SqueezeInfoH;

// String IDs
#define STR(x) #x
#define StrID_Name				"Squeeze"
#define StrID_Description		"Distorts an image by squeezing it horizontally or vertically."
#define StrID_Strength_Param_Name	"Strength"

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

#endif // SQUEEZE_H
