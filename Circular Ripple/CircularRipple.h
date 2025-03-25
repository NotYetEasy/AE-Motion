#pragma once

#ifndef CIRCULAR_RIPPLE_H
#define CIRCULAR_RIPPLE_H

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

// FREQUENCY
#define	CIRCULAR_RIPPLE_FREQ_MIN		0
#define	CIRCULAR_RIPPLE_FREQ_MAX		100
#define	CIRCULAR_RIPPLE_FREQ_DFLT		20

// STRENGTH
#define	CIRCULAR_RIPPLE_STRENGTH_MIN	-1
#define	CIRCULAR_RIPPLE_STRENGTH_MAX	1
#define	CIRCULAR_RIPPLE_STRENGTH_DFLT	0.025

// PHASE
#define	CIRCULAR_RIPPLE_PHASE_MIN		-1000
#define	CIRCULAR_RIPPLE_PHASE_MAX		1000
#define	CIRCULAR_RIPPLE_PHASE_DFLT		0

// RADIUS
#define	CIRCULAR_RIPPLE_RADIUS_MIN		0
#define	CIRCULAR_RIPPLE_RADIUS_MAX		0.8
#define	CIRCULAR_RIPPLE_RADIUS_DFLT		0.3

// FEATHER
#define	CIRCULAR_RIPPLE_FEATHER_MIN		0.001
#define	CIRCULAR_RIPPLE_FEATHER_MAX		1.0
#define	CIRCULAR_RIPPLE_FEATHER_DFLT	0.1

enum {
	CIRCULAR_RIPPLE_INPUT = 0,
	CIRCULAR_RIPPLE_CENTER,
	CIRCULAR_RIPPLE_FREQUENCY,
	CIRCULAR_RIPPLE_STRENGTH,
	CIRCULAR_RIPPLE_PHASE,
	CIRCULAR_RIPPLE_RADIUS,
	CIRCULAR_RIPPLE_FEATHER,
	CIRCULAR_RIPPLE_NUM_PARAMS
};

enum {
	CENTER_DISK_ID = 1,
	FREQUENCY_DISK_ID,
	STRENGTH_DISK_ID,
	PHASE_DISK_ID,
	RADIUS_DISK_ID,
	FEATHER_DISK_ID
};

typedef struct {
	PF_Point        center;
	PF_FpLong       frequency;
	PF_FpLong       strength;
	PF_FpLong       phase;
	PF_FpLong       radius;
	PF_FpLong       feather;
	A_long          width;
	A_long          height;
	void* src;
	A_long          rowbytes;
} RippleInfo;


extern "C" {

	DllExport
	PF_Err
	EffectMain(
		PF_Cmd			cmd,
		PF_InData		*in_data,
		PF_OutData		*out_data,
		PF_ParamDef		*params[],
		PF_LayerDef		*output,
		void			*extra);

}

#endif // CIRCULAR_RIPPLE_H
