#include "Squeeze.h"
#include <stdio.h>
#include <math.h>

static PF_Err
About(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
		"%s v%d.%d\r%s",
		"Squeeze",
		MAJOR_VERSION,
		MINOR_VERSION,
		"Distorts an image by squeezing it horizontally or vertically.");
	return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	out_data->my_version = PF_VERSION(MAJOR_VERSION,
		MINOR_VERSION,
		BUG_VERSION,
		STAGE_VERSION,
		BUILD_VERSION);

	out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
		PF_OutFlag_PIX_INDEPENDENT |
		PF_OutFlag_I_EXPAND_BUFFER;

	out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
		PF_OutFlag2_FLOAT_COLOR_AWARE |
		PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

	return PF_Err_NONE;
}

static PF_Err
ParamsSetup(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err		err = PF_Err_NONE;
	PF_ParamDef	def;

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX("Strength",
		SQUEEZE_STRENGTH_MIN,
		SQUEEZE_STRENGTH_MAX,
		SQUEEZE_STRENGTH_MIN,
		SQUEEZE_STRENGTH_MAX,
		SQUEEZE_STRENGTH_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		STRENGTH_DISK_ID);

	out_data->num_params = SQUEEZE_NUM_PARAMS;

	return err;
}

// Helper function to implement mix operation
static float mix(float a, float b, float t) {
	return a * (1.0f - t) + b * t;
}

// Helper function to implement clamp operation
static float clamp(float value, float min_val, float max_val) {
	if (value < min_val) return min_val;
	if (value > max_val) return max_val;
	return value;
}

// Function to sample pixel with bilinear interpolation
template <typename PixelType>
static void
SampleBilinear(
	void* src_data,
	A_long rowbytes,
	A_long width,
	A_long height,
	float x,
	float y,
	PixelType* outP)
{
	// Check if x is out of bounds - if so, make transparent
	if (x < 0 || x >= width) {
		outP->alpha = 0;
		outP->red = 0;
		outP->green = 0;
		outP->blue = 0;
		return;
	}

	// Get integer and fractional parts
	int x1 = static_cast<int>(x);
	int y1 = static_cast<int>(y);
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	float fx = x - x1;
	float fy = y - y1;

	// Clamp y coordinates to valid range (but not x, which should be transparent)
	y1 = MIN(MAX(y1, 0), height - 1);
	y2 = MIN(MAX(y2, 0), height - 1);

	// Clamp x only for accessing the buffer (not for transparency check)
	int x1_clamped = MIN(MAX(x1, 0), width - 1);
	int x2_clamped = MIN(MAX(x2, 0), width - 1);

	// Get pointers to pixels
	PixelType* p11, * p12, * p21, * p22;

	// Get base pointer
	PixelType* base = reinterpret_cast<PixelType*>(src_data);

	// Calculate pointers to the four pixels for bilinear interpolation
	p11 = reinterpret_cast<PixelType*>((char*)base + (y1 * rowbytes)) + x1_clamped;
	p12 = reinterpret_cast<PixelType*>((char*)base + (y2 * rowbytes)) + x1_clamped;
	p21 = reinterpret_cast<PixelType*>((char*)base + (y1 * rowbytes)) + x2_clamped;
	p22 = reinterpret_cast<PixelType*>((char*)base + (y2 * rowbytes)) + x2_clamped;

	// Check edge conditions for x
	bool x1_valid = (x1 >= 0 && x1 < width);
	bool x2_valid = (x2 >= 0 && x2 < width);

	// Interpolate with zero alpha for out-of-bounds x values
	float s1, s2;

	// Alpha
	s1 = x1_valid ? (1 - fx) * p11->alpha : 0;
	s1 += x2_valid ? fx * p21->alpha : 0;

	s2 = x1_valid ? (1 - fx) * p12->alpha : 0;
	s2 += x2_valid ? fx * p22->alpha : 0;

	outP->alpha = (1 - fy) * s1 + fy * s2;

	// Only process color if we have any alpha
	if (outP->alpha > 0) {
		// Red
		s1 = x1_valid ? (1 - fx) * p11->red : 0;
		s1 += x2_valid ? fx * p21->red : 0;

		s2 = x1_valid ? (1 - fx) * p12->red : 0;
		s2 += x2_valid ? fx * p22->red : 0;

		outP->red = (1 - fy) * s1 + fy * s2;

		// Green
		s1 = x1_valid ? (1 - fx) * p11->green : 0;
		s1 += x2_valid ? fx * p21->green : 0;

		s2 = x1_valid ? (1 - fx) * p12->green : 0;
		s2 += x2_valid ? fx * p22->green : 0;

		outP->green = (1 - fy) * s1 + fy * s2;

		// Blue
		s1 = x1_valid ? (1 - fx) * p11->blue : 0;
		s1 += x2_valid ? fx * p21->blue : 0;

		s2 = x1_valid ? (1 - fx) * p12->blue : 0;
		s2 += x2_valid ? fx * p22->blue : 0;

		outP->blue = (1 - fy) * s1 + fy * s2;
	}
	else {
		// Zero out color channels if alpha is zero
		outP->red = 0;
		outP->green = 0;
		outP->blue = 0;
	}
}

static PF_Err
SqueezeFunc8(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8* inP,
	PF_Pixel8* outP)
{
	PF_Err			err = PF_Err_NONE;
	SqueezeInfo* siP = reinterpret_cast<SqueezeInfo*>(refcon);

	if (siP) {
		// Get normalized coordinates (0 to 1)
		float width = static_cast<float>(siP->width);
		float height = static_cast<float>(siP->height);

		float normX = static_cast<float>(xL) / width;
		float normY = static_cast<float>(yL) / height;

		// Transform to -1 to 1 range
		float stX = 2.0f * normX - 1.0f;
		float stY = 2.0f * normY - 1.0f;

		// Apply squeeze transformation
		float str = siP->strength / 2.0f;

		float absY = fabsf(stY);
		float absX = fabsf(stX);

		// Add a small epsilon to prevent division by very small numbers
		const float epsilon = 0.0001f;

		// Calculate divisors with a smoother transition
		float xdiv = 1.0f + (1.0f - (absY * absY)) * -str;
		float ydiv = 1.0f + (1.0f - (absX * absX)) * str;

		// Clamp the divisors with a slightly larger minimum to avoid division issues
		xdiv = clamp(xdiv, epsilon, 2.0f);
		ydiv = clamp(ydiv, epsilon, 2.0f);

		// Apply the transformation
		stX /= xdiv;
		stY /= ydiv;

		// Transform back to 0 to 1 range
		float newNormX = stX / 2.0f + 0.5f;
		float newNormY = stY / 2.0f + 0.5f;

		// Transform to pixel coordinates
		float sourceX = newNormX * width;
		float sourceY = newNormY * height;

		// Sample the source image
		SampleBilinear<PF_Pixel8>(siP->src8P, siP->rowbytes, siP->width, siP->height, sourceX, sourceY, outP);
	}
	else {
		*outP = *inP;
	}

	return err;
}

static PF_Err
SqueezeFunc16(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel16* inP,
	PF_Pixel16* outP)
{
	PF_Err			err = PF_Err_NONE;
	SqueezeInfo* siP = reinterpret_cast<SqueezeInfo*>(refcon);

	if (siP) {
		// Get normalized coordinates (0 to 1)
		float width = static_cast<float>(siP->width);
		float height = static_cast<float>(siP->height);

		float normX = static_cast<float>(xL) / width;
		float normY = static_cast<float>(yL) / height;

		// Transform to -1 to 1 range
		float stX = 2.0f * normX - 1.0f;
		float stY = 2.0f * normY - 1.0f;

		// Apply squeeze transformation
		float str = siP->strength / 2.0f;

		float absY = fabsf(stY);
		float absX = fabsf(stX);

		// Add a small epsilon to prevent division by very small numbers
		const float epsilon = 0.0001f;

		// Calculate divisors with a smoother transition
		float xdiv = 1.0f + (1.0f - (absY * absY)) * -str;
		float ydiv = 1.0f + (1.0f - (absX * absX)) * str;

		// Clamp the divisors with a slightly larger minimum to avoid division issues
		xdiv = clamp(xdiv, epsilon, 2.0f);
		ydiv = clamp(ydiv, epsilon, 2.0f);

		// Apply the transformation
		stX /= xdiv;
		stY /= ydiv;

		// Transform back to 0 to 1 range
		float newNormX = stX / 2.0f + 0.5f;
		float newNormY = stY / 2.0f + 0.5f;

		// Transform to pixel coordinates
		float sourceX = newNormX * width;
		float sourceY = newNormY * height;

		// Sample the source image
		SampleBilinear<PF_Pixel16>(siP->src16P, siP->rowbytes, siP->width, siP->height, sourceX, sourceY, outP);
	}
	else {
		*outP = *inP;
	}

	return err;
}

static PF_Err
SqueezeFuncFloat(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_PixelFloat* inP,
	PF_PixelFloat* outP)
{
	PF_Err			err = PF_Err_NONE;
	SqueezeInfo* siP = reinterpret_cast<SqueezeInfo*>(refcon);

	if (siP) {
		// Get normalized coordinates (0 to 1)
		float width = static_cast<float>(siP->width);
		float height = static_cast<float>(siP->height);

		float normX = static_cast<float>(xL) / width;
		float normY = static_cast<float>(yL) / height;

		// Transform to -1 to 1 range
		float stX = 2.0f * normX - 1.0f;
		float stY = 2.0f * normY - 1.0f;

		// Apply squeeze transformation
		float str = siP->strength / 2.0f;

		float absY = fabsf(stY);
		float absX = fabsf(stX);

		// Add a small epsilon to prevent division by very small numbers
		const float epsilon = 0.0001f;

		// Calculate divisors with a smoother transition
		float xdiv = 1.0f + (1.0f - (absY * absY)) * -str;
		float ydiv = 1.0f + (1.0f - (absX * absX)) * str;

		// Clamp the divisors with a slightly larger minimum to avoid division issues
		xdiv = clamp(xdiv, epsilon, 2.0f);
		ydiv = clamp(ydiv, epsilon, 2.0f);

		// Apply the transformation
		stX /= xdiv;
		stY /= ydiv;

		// Transform back to 0 to 1 range
		float newNormX = stX / 2.0f + 0.5f;
		float newNormY = stY / 2.0f + 0.5f;

		// Transform to pixel coordinates
		float sourceX = newNormX * width;
		float sourceY = newNormY * height;

		// Sample the source image
		SampleBilinear<PF_PixelFloat>(siP->srcFloatP, siP->rowbytes, siP->width, siP->height, sourceX, sourceY, outP);
	}
	else {
		*outP = *inP;
	}

	return err;
}

static PF_Err
SmartPreRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PreRenderExtra* extra)
{
	PF_Err err = PF_Err_NONE;

	// Initialize effect info structure
	SqueezeInfo info;
	AEFX_CLR_STRUCT(info);

	PF_ParamDef param_copy;
	AEFX_CLR_STRUCT(param_copy);

	// Initialize max_result_rect to the current output request rect
	extra->output->max_result_rect = extra->input->output_request.rect;

	// Get strength parameter
	ERR(PF_CHECKOUT_PARAM(in_data, SQUEEZE_STRENGTH, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.strength = param_copy.u.fs_d.value;

	// Calculate buffer expansion based on strength
	A_long expansion = 0;
	expansion = static_cast<A_long>(ceil(fabs(info.strength) * 100.0f));

	// Set up render request with expanded area
	PF_RenderRequest req = extra->input->output_request;

	if (expansion > 0) {
		req.rect.left -= expansion;
		req.rect.top -= expansion;
		req.rect.right += expansion;
		req.rect.bottom += expansion;
	}
	req.preserve_rgb_of_zero_alpha = TRUE;

	// Checkout the input layer with our expanded request
	PF_CheckoutResult checkout;
	ERR(extra->cb->checkout_layer(in_data->effect_ref,
		SQUEEZE_INPUT,
		SQUEEZE_INPUT,
		&req,
		in_data->current_time,
		in_data->time_step,
		in_data->time_scale,
		&checkout));

	if (!err) {
		// Update max_result_rect based on checkout result
		extra->output->max_result_rect = checkout.max_result_rect;

		// Set result_rect to match max_result_rect
		extra->output->result_rect = extra->output->max_result_rect;

		// Set output flags
		extra->output->solid = FALSE;
		extra->output->pre_render_data = NULL;
		extra->output->flags = PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;
	}

	return err;
}

static PF_Err
SmartRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_SmartRenderExtra* extra)
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	// Checkout input layer pixels
	PF_EffectWorld* input_worldP = NULL;
	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, SQUEEZE_INPUT, &input_worldP));

	if (!err && input_worldP) {
		// Checkout output buffer
		PF_EffectWorld* output_worldP = NULL;
		ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

		if (!err && output_worldP) {
			PF_ParamDef param_copy;
			AEFX_CLR_STRUCT(param_copy);

			SqueezeInfo info;
			AEFX_CLR_STRUCT(info);

			// Get strength parameter
			ERR(PF_CHECKOUT_PARAM(in_data, SQUEEZE_STRENGTH, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) info.strength = param_copy.u.fs_d.value;

			// Set up info
			info.width = input_worldP->width;
			info.height = input_worldP->height;
			info.rowbytes = input_worldP->rowbytes;
			info.input = input_worldP;

			// Clear output buffer
			PF_Pixel empty_pixel = { 0, 0, 0, 0 };
			ERR(suites.FillMatteSuite2()->fill(
				in_data->effect_ref,
				&empty_pixel,
				NULL,
				output_worldP));

			if (!err) {
				// Determine pixel depth by examining the world
				double bytesPerPixel = static_cast<double>(input_worldP->rowbytes) / static_cast<double>(input_worldP->width);
				bool is16bit = false;
				bool is32bit = false;

				if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
					is32bit = true;
					info.srcFloatP = reinterpret_cast<PF_PixelFloat*>(input_worldP->data);

					// Use the IterateFloatSuite for 32-bit processing
					ERR(suites.IterateFloatSuite1()->iterate(
						in_data,
						0,                // progress base
						output_worldP->height,  // progress final
						input_worldP,     // src
						NULL,             // area - null for all pixels
						static_cast<void*>(&info),     // refcon - custom data
						SqueezeFuncFloat,   // pixel function pointer
						output_worldP));
				}
				else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
					is16bit = true;
					info.src16P = reinterpret_cast<PF_Pixel16*>(input_worldP->data);

					// Process with 16-bit iterate suite
					ERR(suites.Iterate16Suite2()->iterate(
						in_data,
						0,                // progress base
						output_worldP->height,  // progress final
						input_worldP,     // src
						NULL,             // area - null for all pixels
						static_cast<void*>(&info),     // refcon - custom data
						SqueezeFunc16,      // pixel function pointer
						output_worldP));
				}
				else {
					info.src8P = reinterpret_cast<PF_Pixel8*>(input_worldP->data);

					// Process with 8-bit iterate suite
					ERR(suites.Iterate8Suite2()->iterate(
						in_data,
						0,                // progress base
						output_worldP->height,  // progress final
						input_worldP,     // src
						NULL,             // area - null for all pixels
						static_cast<void*>(&info),     // refcon - custom data
						SqueezeFunc8,       // pixel function pointer
						output_worldP));
				}
			}
		}
	}

	// Check in the input layer pixels
	if (input_worldP) {
		ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, SQUEEZE_INPUT));
	}

	return err;
}

static PF_Err
Render(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err				err = PF_Err_NONE;
	AEGP_SuiteHandler	suites(in_data->pica_basicP);

	SqueezeInfo			siP;
	AEFX_CLR_STRUCT(siP);

	// Get input parameters
	siP.strength = params[SQUEEZE_STRENGTH]->u.fs_d.value;

	// Store dimensions for the sampling functions
	siP.width = output->width;
	siP.height = output->height;
	siP.rowbytes = params[SQUEEZE_INPUT]->u.ld.rowbytes;
	siP.src8P = reinterpret_cast<PF_Pixel8*>(params[SQUEEZE_INPUT]->u.ld.data);

	// Determine pixel depth by examining the world
	double bytesPerPixel = static_cast<double>(siP.rowbytes) / static_cast<double>(siP.width);
	bool is16bit = false;
	bool is32bit = false;

	if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
		is32bit = true;
	}
	else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
		is16bit = true;
	}

	A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

	// Process the image based on bit depth
	if (is32bit) {
		// 32-bit floating point
		siP.srcFloatP = reinterpret_cast<PF_PixelFloat*>(params[SQUEEZE_INPUT]->u.ld.data);

		ERR(suites.IterateFloatSuite1()->iterate(
			in_data,
			0,								// progress base
			linesL,							// progress final
			&params[SQUEEZE_INPUT]->u.ld,	// src 
			NULL,							// area - null for all pixels
			static_cast<void*>(&siP),		// refcon - your custom data pointer
			SqueezeFuncFloat,				// pixel function pointer
			output));
	}
	else if (is16bit) {
		// 16-bit
		siP.src16P = reinterpret_cast<PF_Pixel16*>(params[SQUEEZE_INPUT]->u.ld.data);

		ERR(suites.Iterate16Suite2()->iterate(
			in_data,
			0,								// progress base
			linesL,							// progress final
			&params[SQUEEZE_INPUT]->u.ld,	// src 
			NULL,							// area - null for all pixels
			static_cast<void*>(&siP),		// refcon - your custom data pointer
			SqueezeFunc16,					// pixel function pointer
			output));
	}
	else {
		// 8-bit
		ERR(suites.Iterate8Suite2()->iterate(
			in_data,
			0,								// progress base
			linesL,							// progress final
			&params[SQUEEZE_INPUT]->u.ld,	// src 
			NULL,							// area - null for all pixels
			static_cast<void*>(&siP),		// refcon - your custom data pointer
			SqueezeFunc8,					// pixel function pointer
			output));
	}

	return err;
}


extern "C" DllExport
PF_Err PluginDataEntryFunction2(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB2 inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	result = PF_REGISTER_EFFECT_EXT2(
		inPtr,
		inPluginDataCallBackPtr,
		"Squeeze", // Name
		"DKT Squeeze", // Match Name
		"DKT Effects", // Category
		AE_RESERVED_INFO, // Reserved Info
		"EffectMain",	// Entry point
		"");	// support URL removed

	return result;
}


PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output,
	void* extra)
{
	PF_Err		err = PF_Err_NONE;

	try {
		switch (cmd) {
		case PF_Cmd_ABOUT:
			err = About(in_data,
				out_data,
				params,
				output);
			break;

		case PF_Cmd_GLOBAL_SETUP:
			err = GlobalSetup(in_data,
				out_data,
				params,
				output);
			break;

		case PF_Cmd_PARAMS_SETUP:
			err = ParamsSetup(in_data,
				out_data,
				params,
				output);
			break;

		case PF_Cmd_RENDER:
			err = Render(in_data,
				out_data,
				params,
				output);
			break;

		case PF_Cmd_SMART_PRE_RENDER:
			err = SmartPreRender(in_data,
				out_data,
				(PF_PreRenderExtra*)extra);
			break;

		case PF_Cmd_SMART_RENDER:
			err = SmartRender(in_data,
				out_data,
				(PF_SmartRenderExtra*)extra);
			break;

			// Handle other commands silently
		default:
			break;
		}
	}
	catch (PF_Err& thrown_err) {
		err = thrown_err;
	}
	catch (std::exception& e) {
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
	catch (...) {
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	return err;
}
