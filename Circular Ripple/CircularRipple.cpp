#include "CircularRipple.h"
#include <stdio.h> 
#include <stdlib.h>

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

static PF_Err
About(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
		"Circular Ripple\r"
		"Created by DKT with Unknown's help.\r"
		"Under development!!\r"
		"Discord: dkt0 and unknown1234\r"
		"Contact us if you want to contribute or report bugs!");
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

	// CENTER POINT
	PF_ADD_POINT("Center",
		0,
		0,
		0,
		CENTER_DISK_ID);

	// FREQUENCY
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Frequency",
		0,			// Min
		100,		// Max
		0,			// Valid min
		20,			// Valid max
		20,			// Default
		PF_Precision_HUNDREDTHS,
		0,
		0,
		FREQUENCY_DISK_ID);

	// STRENGTH
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Strength",
		-1,			// Min
		1,			// Max
		-1,			// Valid min
		1,			// Valid max
		0.025,		// Default
		PF_Precision_THOUSANDTHS,
		0,
		0,
		STRENGTH_DISK_ID);

	// PHASE
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Phase",
		-1000,		// Min
		1000,		// Max
		-1,		// Valid min
		1,		// Valid max
		0,			// Default
		PF_Precision_HUNDREDTHS,
		0,
		0,
		PHASE_DISK_ID);

	// RADIUS
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Radius",
		0,			// Min
		0.8,		// Max
		0,			// Valid min
		0.8,		// Valid max
		0.3,		// Default
		PF_Precision_HUNDREDTHS,
		0,
		0,
		RADIUS_DISK_ID);

	// FEATHER
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Feather",
		0.001,		// Min
		1.0,		// Max
		0.001,		// Valid min
		1.0,		// Valid max
		0.1,		// Default
		PF_Precision_THOUSANDTHS,
		0,
		0,
		FEATHER_DISK_ID);

	out_data->num_params = CIRCULAR_RIPPLE_NUM_PARAMS;

	return err;
}

// Helper function to implement mix operation
static float mix(float a, float b, float t) {
	return a * (1.0f - t) + b * t;
}

// Helper function to implement smoothstep operation
static float smoothstep(float edge0, float edge1, float x) {
	float t = MAX(0.0f, MIN(1.0f, (x - edge0) / (edge1 - edge0)));
	return t * t * (3.0f - 2.0f * t);
}

/**
 * Function to sample pixel with bilinear interpolation
 */
template <typename PixelType>
static void
SampleBilinear(
	PF_EffectWorld* input,
	float x,
	float y,
	PixelType* outP)
{
	// Check if x is out of bounds - if so, make transparent
	if (x < 0 || x >= input->width) {
		outP->alpha = 0;
		outP->red = 0;
		outP->green = 0;
		outP->blue = 0;
		return;
	}

	// Get integer and fractional parts
	int x1 = (int)x;
	int y1 = (int)y;
	int x2 = x1 + 1;
	int y2 = y1 + 1;

	float fx = x - x1;
	float fy = y - y1;

	// Clamp y coordinates to valid range (but not x, which should be transparent)
	y1 = MIN(MAX(y1, 0), input->height - 1);
	y2 = MIN(MAX(y2, 0), input->height - 1);

	// Clamp x only for accessing the buffer (not for transparency check)
	int x1_clamped = MIN(MAX(x1, 0), input->width - 1);
	int x2_clamped = MIN(MAX(x2, 0), input->width - 1);

	// Get pointers to pixels
	PixelType* p11, * p12, * p21, * p22;

	// Determine pixel depth by examining the world
	double bytesPerPixel = (double)input->rowbytes / (double)input->width;

	if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
		PF_PixelFloat* base = reinterpret_cast<PF_PixelFloat*>(input->data);
		p11 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_PixelFloat) + x1_clamped]);
		p12 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_PixelFloat) + x1_clamped]);
		p21 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_PixelFloat) + x2_clamped]);
		p22 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_PixelFloat) + x2_clamped]);
	}
	else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
		PF_Pixel16* base = reinterpret_cast<PF_Pixel16*>(input->data);
		p11 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_Pixel16) + x1_clamped]);
		p12 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_Pixel16) + x1_clamped]);
		p21 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_Pixel16) + x2_clamped]);
		p22 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_Pixel16) + x2_clamped]);
	}
	else { // 8-bit (default)
		PF_Pixel8* base = reinterpret_cast<PF_Pixel8*>(input->data);
		p11 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_Pixel8) + x1_clamped]);
		p12 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_Pixel8) + x1_clamped]);
		p21 = reinterpret_cast<PixelType*>(&base[y1 * input->rowbytes / sizeof(PF_Pixel8) + x2_clamped]);
		p22 = reinterpret_cast<PixelType*>(&base[y2 * input->rowbytes / sizeof(PF_Pixel8) + x2_clamped]);
	}

	// Check edge conditions for x
	bool x1_valid = (x1 >= 0 && x1 < input->width);
	bool x2_valid = (x2 >= 0 && x2 < input->width);

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

/**
 * Pixel processing function for 8-bit color depth
 */
static PF_Err
RippleFunc8(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel8* inP,
	PF_Pixel8* outP)
{
	PF_Err			err = PF_Err_NONE;
	RippleInfo* rippleP = reinterpret_cast<RippleInfo*>(refcon);

	if (!rippleP) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	// Use the input pixel initially
	*outP = *inP;

	// Get the dimensions and parameters
	PF_FpLong width = (PF_FpLong)rippleP->width;
	PF_FpLong height = (PF_FpLong)rippleP->height;

	// Calculate center position based on user parameter
	PF_FpLong centerX = width / 2.0 + (PF_FpLong)rippleP->center.x / 65536.0;
	PF_FpLong centerY = height / 2.0 - (PF_FpLong)rippleP->center.y / 65536.0;

	// Calculate normalized UV coordinates
	PF_FpLong uvX = (PF_FpLong)xL / width;
	PF_FpLong uvY = (PF_FpLong)yL / height;

	// Calculate center in normalized coordinates
	PF_FpLong centerNormX = centerX / width;
	PF_FpLong centerNormY = centerY / height;

	// Offset UV by center
	PF_FpLong offsetX = uvX - centerNormX;
	PF_FpLong offsetY = uvY - centerNormY;

	// Adjust for aspect ratio (multiply Y by height/width)
	offsetY *= (height / width);

	// Calculate distance from center
	PF_FpLong dist = sqrt(offsetX * offsetX + offsetY * offsetY);

	// Calculate feather size and radii
	PF_FpLong featherSize = rippleP->radius * 0.5 * rippleP->feather;
	PF_FpLong innerRadius = MAX(0.0, rippleP->radius - featherSize);
	PF_FpLong outerRadius = MAX(innerRadius + 0.00001, rippleP->radius + featherSize);

	// Calculate damping factor
	PF_FpLong damping;
	if (dist >= outerRadius) {
		damping = 0.0;
	}
	else if (dist <= innerRadius) {
		damping = 1.0;
	}
	else {
		// Smoothstep implementation
		PF_FpLong t = (dist - innerRadius) / (outerRadius - innerRadius);
		t = 1.0 - t; // Invert t
		damping = t * t * (3.0 - 2.0 * t);
	}

	// Calculate ripple offset
	const PF_FpLong PI = 3.14159265358979323846;
	PF_FpLong angle = (dist * rippleP->frequency * PI * 2.0) + (rippleP->phase * PI * 2.0);
	PF_FpLong sinVal = sin(angle);

	// Calculate normalized direction vector
	PF_FpLong len = sqrt(offsetX * offsetX + offsetY * offsetY);
	PF_FpLong normX = 0, normY = 0;
	if (len > 0.0001) {
		normX = offsetX / len;
		normY = offsetY / len;
	}

	// Apply offset
	PF_FpLong strength_factor = rippleP->strength / 2.0;
	PF_FpLong offsetFactorX = sinVal * strength_factor * normX * damping;
	PF_FpLong offsetFactorY = sinVal * strength_factor * normY * damping;

	offsetX += offsetFactorX;
	offsetY += offsetFactorY;

	// Undo aspect ratio adjustment
	offsetY /= (height / width);

	// Calculate final sampling coordinates
	PF_FpLong sampleX = (offsetX + centerNormX) * width;
	PF_FpLong sampleY = (offsetY + centerNormY) * height;

	// Check if horizontal coordinate is out of bounds
	if (sampleX < 0 || sampleX >= width) {
		// Make pixel transparent/black for horizontal out-of-bounds
		outP->alpha = 0;
		outP->red = 0;
		outP->green = 0;
		outP->blue = 0;
		return err;
	}

	// For vertical coordinates, we'll clamp to the image boundaries
	A_long sourceX = (A_long)(sampleX + 0.5);
	A_long sourceY = (A_long)(sampleY + 0.5);

	// Clamp only Y coordinates (vertical)
	sourceX = MIN(MAX(sourceX, 0), rippleP->width - 1);
	sourceY = MIN(MAX(sourceY, 0), rippleP->height - 1);

	// Sample the source pixel
	PF_Pixel8* srcP = (PF_Pixel8*)((char*)rippleP->src + (sourceY * rippleP->rowbytes)) + sourceX;

	// Copy the sampled pixel to output
	*outP = *srcP;

	return err;
}

/**
 * Pixel processing function for 16-bit color depth
 */
static PF_Err
RippleFunc16(
	void* refcon,
	A_long		xL,
	A_long		yL,
	PF_Pixel16* inP,
	PF_Pixel16* outP)
{
	PF_Err			err = PF_Err_NONE;
	RippleInfo* rippleP = reinterpret_cast<RippleInfo*>(refcon);

	if (!rippleP) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	// Use the input pixel initially
	*outP = *inP;

	// Get the dimensions and parameters
	PF_FpLong width = (PF_FpLong)rippleP->width;
	PF_FpLong height = (PF_FpLong)rippleP->height;

	// Calculate center position based on user parameter
	PF_FpLong centerX = width / 2.0 + (PF_FpLong)rippleP->center.x / 65536.0;
	PF_FpLong centerY = height / 2.0 - (PF_FpLong)rippleP->center.y / 65536.0;

	// Calculate normalized UV coordinates
	PF_FpLong uvX = (PF_FpLong)xL / width;
	PF_FpLong uvY = (PF_FpLong)yL / height;

	// Calculate center in normalized coordinates
	PF_FpLong centerNormX = centerX / width;
	PF_FpLong centerNormY = centerY / height;

	// Offset UV by center
	PF_FpLong offsetX = uvX - centerNormX;
	PF_FpLong offsetY = uvY - centerNormY;

	// Adjust for aspect ratio (multiply Y by height/width)
	offsetY *= (height / width);

	// Calculate distance from center
	PF_FpLong dist = sqrt(offsetX * offsetX + offsetY * offsetY);

	// Calculate feather size and radii
	PF_FpLong featherSize = rippleP->radius * 0.5 * rippleP->feather;
	PF_FpLong innerRadius = MAX(0.0, rippleP->radius - featherSize);
	PF_FpLong outerRadius = MAX(innerRadius + 0.00001, rippleP->radius + featherSize);

	// Calculate damping factor
	PF_FpLong damping;
	if (dist >= outerRadius) {
		damping = 0.0;
	}
	else if (dist <= innerRadius) {
		damping = 1.0;
	}
	else {
		// Smoothstep implementation
		PF_FpLong t = (dist - innerRadius) / (outerRadius - innerRadius);
		t = 1.0 - t; // Invert t
		damping = t * t * (3.0 - 2.0 * t);
	}

	// Calculate ripple offset
	const PF_FpLong PI = 3.14159265358979323846;
	PF_FpLong angle = (dist * rippleP->frequency * PI * 2.0) + (rippleP->phase * PI * 2.0);
	PF_FpLong sinVal = sin(angle);

	// Calculate normalized direction vector
	PF_FpLong len = sqrt(offsetX * offsetX + offsetY * offsetY);
	PF_FpLong normX = 0, normY = 0;
	if (len > 0.0001) {
		normX = offsetX / len;
		normY = offsetY / len;
	}

	// Apply offset
	PF_FpLong strength_factor = rippleP->strength / 2.0;
	PF_FpLong offsetFactorX = sinVal * strength_factor * normX * damping;
	PF_FpLong offsetFactorY = sinVal * strength_factor * normY * damping;

	offsetX += offsetFactorX;
	offsetY += offsetFactorY;

	// Undo aspect ratio adjustment
	offsetY /= (height / width);

	// Calculate final sampling coordinates
	PF_FpLong sampleX = (offsetX + centerNormX) * width;
	PF_FpLong sampleY = (offsetY + centerNormY) * height;

	// Check if horizontal coordinate is out of bounds
	if (sampleX < 0 || sampleX >= width) {
		// Make pixel transparent/black for horizontal out-of-bounds
		outP->alpha = 0;
		outP->red = 0;
		outP->green = 0;
		outP->blue = 0;
		return err;
	}

	// For vertical coordinates, we'll clamp to the image boundaries
	A_long sourceX = (A_long)(sampleX + 0.5);
	A_long sourceY = (A_long)(sampleY + 0.5);

	// Clamp only Y coordinates (vertical)
	sourceX = MIN(MAX(sourceX, 0), rippleP->width - 1);
	sourceY = MIN(MAX(sourceY, 0), rippleP->height - 1);

	// Sample the source pixel
	PF_Pixel16* srcP = (PF_Pixel16*)((char*)rippleP->src + (sourceY * rippleP->rowbytes)) + sourceX;

	// Copy the sampled pixel to output
	*outP = *srcP;

	return err;
}

/**
 * Pixel processing function for 32-bit float color depth
 */
static PF_Err
RippleFuncFloat(
	void* refcon,
	A_long        xL,
	A_long        yL,
	PF_PixelFloat* inP,
	PF_PixelFloat* outP)
{
	PF_Err            err = PF_Err_NONE;
	RippleInfo* rippleP = reinterpret_cast<RippleInfo*>(refcon);

	if (!rippleP) {
		return PF_Err_BAD_CALLBACK_PARAM;
	}

	// Use the input pixel initially
	*outP = *inP;

	// Get the dimensions and parameters
	PF_FpLong width = (PF_FpLong)rippleP->width;
	PF_FpLong height = (PF_FpLong)rippleP->height;

	// Calculate center position based on user parameter
	PF_FpLong centerX = width / 2.0 + (PF_FpLong)rippleP->center.x / 65536.0;
	PF_FpLong centerY = height / 2.0 - (PF_FpLong)rippleP->center.y / 65536.0;

	// Calculate normalized UV coordinates
	PF_FpLong uvX = (PF_FpLong)xL / width;
	PF_FpLong uvY = (PF_FpLong)yL / height;

	// Calculate center in normalized coordinates
	PF_FpLong centerNormX = centerX / width;
	PF_FpLong centerNormY = centerY / height;

	// Offset UV by center
	PF_FpLong offsetX = uvX - centerNormX;
	PF_FpLong offsetY = uvY - centerNormY;

	// Adjust for aspect ratio (multiply Y by height/width)
	offsetY *= (height / width);

	// Calculate distance from center
	PF_FpLong dist = sqrt(offsetX * offsetX + offsetY * offsetY);

	// Calculate feather size and radii
	PF_FpLong featherSize = rippleP->radius * 0.5 * rippleP->feather;
	PF_FpLong innerRadius = MAX(0.0, rippleP->radius - featherSize);
	PF_FpLong outerRadius = MAX(innerRadius + 0.00001, rippleP->radius + featherSize);

	// Calculate damping factor
	PF_FpLong damping;
	if (dist >= outerRadius) {
		damping = 0.0;
	}
	else if (dist <= innerRadius) {
		damping = 1.0;
	}
	else {
		// Smoothstep implementation
		PF_FpLong t = (dist - innerRadius) / (outerRadius - innerRadius);
		t = 1.0 - t; // Invert t
		damping = t * t * (3.0 - 2.0 * t);
	}

	// Calculate ripple offset
	const PF_FpLong PI = 3.14159265358979323846;
	PF_FpLong angle = (dist * rippleP->frequency * PI * 2.0) + (rippleP->phase * PI * 2.0);
	PF_FpLong sinVal = sin(angle);

	// Calculate normalized direction vector
	PF_FpLong len = sqrt(offsetX * offsetX + offsetY * offsetY);
	PF_FpLong normX = 0, normY = 0;
	if (len > 0.0001) {
		normX = offsetX / len;
		normY = offsetY / len;
	}

	// Apply offset
	PF_FpLong strength_factor = rippleP->strength / 2.0;
	PF_FpLong offsetFactorX = sinVal * strength_factor * normX * damping;
	PF_FpLong offsetFactorY = sinVal * strength_factor * normY * damping;

	offsetX += offsetFactorX;
	offsetY += offsetFactorY;

	// Undo aspect ratio adjustment
	offsetY /= (height / width);

	// Calculate final sampling coordinates
	PF_FpLong sampleX = (offsetX + centerNormX) * width;
	PF_FpLong sampleY = (offsetY + centerNormY) * height;

	// Check if horizontal coordinate is out of bounds
	if (sampleX < 0 || sampleX >= width) {
		// Make pixel transparent/black for horizontal out-of-bounds
		outP->alpha = 0;
		outP->red = 0;
		outP->green = 0;
		outP->blue = 0;
		return err;
	}

	// For vertical coordinates, we'll clamp to the image boundaries
	A_long sourceX = (A_long)(sampleX + 0.5);
	A_long sourceY = (A_long)(sampleY + 0.5);

	// Clamp only Y coordinates (vertical)
	sourceX = MIN(MAX(sourceX, 0), rippleP->width - 1);
	sourceY = MIN(MAX(sourceY, 0), rippleP->height - 1);

	// Sample the source pixel
	PF_PixelFloat* srcP = (PF_PixelFloat*)((char*)rippleP->src + (sourceY * rippleP->rowbytes)) + sourceX;

	// Copy the sampled pixel to output
	*outP = *srcP;

	return err;
}

/**
 * Smart PreRender function - prepares for rendering
 */
static PF_Err
SmartPreRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PreRenderExtra* extra)
{
	PF_Err err = PF_Err_NONE;

	// Initialize ripple info structure
	RippleInfo ripple;
	AEFX_CLR_STRUCT(ripple);

	PF_ParamDef param_copy;
	AEFX_CLR_STRUCT(param_copy);

	// Initialize max_result_rect to the current output request rect
	extra->output->max_result_rect = extra->input->output_request.rect;

	// Get center parameter
	ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_CENTER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) {
		ripple.center.x = param_copy.u.td.x_value;
		ripple.center.y = param_copy.u.td.y_value;
	}

	// Get other parameters
	ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_FREQUENCY, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) ripple.frequency = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_STRENGTH, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) ripple.strength = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_PHASE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) ripple.phase = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_RADIUS, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) ripple.radius = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_FEATHER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) ripple.feather = param_copy.u.fs_d.value;

	// Calculate buffer expansion based on radius and strength
	A_long expansion = 0;
	expansion = ceil(fabs(ripple.strength) * ripple.radius * 100.0);

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
		CIRCULAR_RIPPLE_INPUT,
		CIRCULAR_RIPPLE_INPUT,
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

/**
 * Smart Render function - performs the actual effect rendering
 */
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
	ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, CIRCULAR_RIPPLE_INPUT, &input_worldP));

	if (!err && input_worldP) {
		// Checkout output buffer
		PF_EffectWorld* output_worldP = NULL;
		ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

		if (!err && output_worldP) {
			PF_ParamDef param_copy;
			AEFX_CLR_STRUCT(param_copy);

			RippleInfo ripple;
			AEFX_CLR_STRUCT(ripple);

			// Get center parameter
			ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_CENTER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) {
				ripple.center.x = param_copy.u.td.x_value;
				ripple.center.y = param_copy.u.td.y_value;
			}

			// Get other parameters
			ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_FREQUENCY, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) ripple.frequency = param_copy.u.fs_d.value;

			ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_STRENGTH, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) ripple.strength = param_copy.u.fs_d.value;

			ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_PHASE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) ripple.phase = param_copy.u.fs_d.value;

			ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_RADIUS, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) ripple.radius = param_copy.u.fs_d.value;

			ERR(PF_CHECKOUT_PARAM(in_data, CIRCULAR_RIPPLE_FEATHER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) ripple.feather = param_copy.u.fs_d.value;

			// Set up ripple info
			ripple.width = input_worldP->width;
			ripple.height = input_worldP->height;
			ripple.src = input_worldP->data;
			ripple.rowbytes = input_worldP->rowbytes;

			// Clear output buffer
			PF_Pixel empty_pixel = { 0, 0, 0, 0 };
			ERR(suites.FillMatteSuite2()->fill(
				in_data->effect_ref,
				&empty_pixel,
				NULL,
				output_worldP));

			if (!err) {
				// Determine pixel depth by examining the world
				double bytesPerPixel = (double)input_worldP->rowbytes / (double)input_worldP->width;
				bool is16bit = false;
				bool is32bit = false;

				if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
					is32bit = true;
				}
				else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
					is16bit = true;
				}

				if (is32bit) {
					// Use the IterateFloatSuite for 32-bit processing
					ERR(suites.IterateFloatSuite1()->iterate(
						in_data,
						0,                // progress base
						output_worldP->height,  // progress final
						input_worldP,     // src
						NULL,             // area - null for all pixels
						(void*)&ripple,   // refcon - custom data
						RippleFuncFloat,  // pixel function pointer
						output_worldP));
				}
				else if (is16bit) {
					// Process with 16-bit iterate suite
					ERR(suites.Iterate16Suite2()->iterate(
						in_data,
						0,                // progress base
						output_worldP->height,  // progress final
						input_worldP,     // src
						NULL,             // area - null for all pixels
						(void*)&ripple,   // refcon - custom data
						RippleFunc16,     // pixel function pointer
						output_worldP));
				}
				else {
					// Process with 8-bit iterate suite
					ERR(suites.Iterate8Suite2()->iterate(
						in_data,
						0,                // progress base
						output_worldP->height,  // progress final
						input_worldP,     // src
						NULL,             // area - null for all pixels
						(void*)&ripple,   // refcon - custom data
						RippleFunc8,      // pixel function pointer
						output_worldP));
				}
			}
		}
	}

	// Check in the input layer pixels
	if (input_worldP) {
		ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, CIRCULAR_RIPPLE_INPUT));
	}

	return err;
}

/**
 * Legacy rendering function for older AE versions
 */
static PF_Err
Render(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err                err = PF_Err_NONE;
	AEGP_SuiteHandler    suites(in_data->pica_basicP);

	// Set up ripple info structure
	RippleInfo            ripple;
	AEFX_CLR_STRUCT(ripple);

	// Get parameter values
	ripple.center.x = params[CIRCULAR_RIPPLE_CENTER]->u.td.x_value;
	ripple.center.y = params[CIRCULAR_RIPPLE_CENTER]->u.td.y_value;
	ripple.frequency = params[CIRCULAR_RIPPLE_FREQUENCY]->u.fs_d.value;
	ripple.strength = params[CIRCULAR_RIPPLE_STRENGTH]->u.fs_d.value;
	ripple.phase = params[CIRCULAR_RIPPLE_PHASE]->u.fs_d.value;
	ripple.radius = params[CIRCULAR_RIPPLE_RADIUS]->u.fs_d.value;
	ripple.feather = params[CIRCULAR_RIPPLE_FEATHER]->u.fs_d.value;

	// Get input layer dimensions and data
	ripple.width = params[CIRCULAR_RIPPLE_INPUT]->u.ld.width;
	ripple.height = params[CIRCULAR_RIPPLE_INPUT]->u.ld.height;
	ripple.src = params[CIRCULAR_RIPPLE_INPUT]->u.ld.data;
	ripple.rowbytes = params[CIRCULAR_RIPPLE_INPUT]->u.ld.rowbytes;

	// Determine pixel depth by examining the world
	double bytesPerPixel = (double)ripple.rowbytes / (double)ripple.width;
	bool is16bit = false;
	bool is32bit = false;

	if (bytesPerPixel >= 16.0) { // 32-bit float (4 channels * 4 bytes)
		is32bit = true;
	}
	else if (bytesPerPixel >= 8.0) { // 16-bit (4 channels * 2 bytes)
		is16bit = true;
	}

	A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

	if (is32bit) {
		// Use the IterateFloatSuite for 32-bit processing
		ERR(suites.IterateFloatSuite1()->iterate(
			in_data,
			0,                                // progress base
			linesL,                           // progress final
			&params[CIRCULAR_RIPPLE_INPUT]->u.ld, // src 
			NULL,                            // area - null for all pixels
			(void*)&ripple,                  // refcon - custom data
			RippleFuncFloat,                 // pixel function pointer
			output));
	}
	else if (is16bit) {
		// Process with 16-bit iterate suite
		ERR(suites.Iterate16Suite2()->iterate(
			in_data,
			0,                                // progress base
			linesL,                           // progress final
			&params[CIRCULAR_RIPPLE_INPUT]->u.ld, // src 
			NULL,                            // area - null for all pixels
			(void*)&ripple,                  // refcon - custom data
			RippleFunc16,                    // pixel function pointer
			output));
	}
	else {
		// Process with 8-bit iterate suite
		ERR(suites.Iterate8Suite2()->iterate(
			in_data,
			0,                                // progress base
			linesL,                           // progress final
			&params[CIRCULAR_RIPPLE_INPUT]->u.ld, // src 
			NULL,                            // area - null for all pixels
			(void*)&ripple,                  // refcon - custom data
			RippleFunc8,                     // pixel function pointer
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
		"Circular Ripple", // Name
		"DKT Circular Ripple", // Match Name
		"DKT Effects", // Category
		AE_RESERVED_INFO, // Reserved Info
		"EffectMain",	// Entry point
		"https://www.adobe.com");	// support URL

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
		}
	}
	catch (PF_Err& thrown_err) {
		err = thrown_err;
	}
	return err;
}

