#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "Oscillate.h"
#include <iostream>

// brings in M_PI on Windows
#define _USE_MATH_DEFINES
#include <math.h>

inline PF_Err CL2Err(cl_int cl_result) {
	if (cl_result == CL_SUCCESS) {
		return PF_Err_NONE;
	}
	else {
		// set a breakpoint here to pick up OpenCL errors.
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))


// CUDA kernel; see Oscillate.cu
extern void Oscillate_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float angle,
	float frequency,
	float magnitude,
	int direction,
	int waveType,
	float phase,
	float currentTime,
	int xTiles,
	int yTiles,
	int mirror);

static PF_Err
About(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
		"Oscillate v%d.%d\r"
		"Created by DKT with Unknown's help.\r"
		"Under development!!\r"
		"Discord: dkt0 and unknown1234\r"
		"Contact us if you want to contribute or report bugs!",
		MAJOR_VERSION,
		MINOR_VERSION);
	return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
	PF_InData* in_dataP,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err	err = PF_Err_NONE;

	// Set version information
	out_data->my_version = PF_VERSION(MAJOR_VERSION,
		MINOR_VERSION,
		BUG_VERSION,
		STAGE_VERSION,
		BUILD_VERSION);

	// Set plugin flags
	out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
		PF_OutFlag_NON_PARAM_VARY;

	out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
		PF_OutFlag2_FLOAT_COLOR_AWARE |
		PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
		PF_OutFlag2_I_MIX_GUID_DEPENDENCIES |
		PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 |
		PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;

	return err;
}

static PF_Err UpdateParameterUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[])
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	// Get the direction value (subtract 1 to convert from 1-based to 0-based)
	A_long direction = params[DIRECTION_SLIDER]->u.pd.value - 1;

	// Create a copy of the angle parameter to modify
	PF_ParamDef param_copy = *params[ANGLE_SLIDER];

	// Disable angle parameter when "Depth" (direction = 1) is selected
	if (direction == 1) {  // Depth selected
		param_copy.ui_flags |= PF_PUI_DISABLED;
	}
	else {
		param_copy.ui_flags &= ~PF_PUI_DISABLED;
	}

	// Update the UI
	ERR(suites.ParamUtilsSuite3()->PF_UpdateParamUI(in_data->effect_ref,
		ANGLE_SLIDER,
		&param_copy));

	return err;
}

static PF_Err
ParamsSetup(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err err = PF_Err_NONE;
	PF_ParamDef def;

	// Clear parameter definition structure
	AEFX_CLR_STRUCT(def);

	// Direction parameter (Angle, Depth, or Orbit)
	PF_ADD_POPUP("Direction",
		3,                        // Number of choices
		1,                        // Default choice (1-based)
		"Angle|Depth|Orbit",      // Choices
		PF_ParamFlag_SUPERVISE,   // Enable parameter supervision
		DIRECTION_SLIDER);

	AEFX_CLR_STRUCT(def);

	// Angle parameter (in degrees)
	PF_ADD_FLOAT_SLIDERX("Angle",
		-3600.0,           // Min value
		3600.0,           // Max value
		0.0,              // Valid min
		360.0,            // Valid max
		45.0,             // Default value
		PF_Precision_TENTHS,
		PF_ParamFlag_SUPERVISE,
		0,
		ANGLE_SLIDER);

	AEFX_CLR_STRUCT(def);

	// Frequency parameter (oscillation speed)
	PF_ADD_FLOAT_SLIDERX("Frequency",
		FREQUENCY_MIN,
		FREQUENCY_MAX,
		FREQUENCY_MIN,
		FREQUENCY_MAX,
		FREQUENCY_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		FREQUENCY_SLIDER);

	AEFX_CLR_STRUCT(def);

	// Magnitude parameter (oscillation amount)
	PF_ADD_FLOAT_SLIDERX("Magnitude",
		MAGNITUDE_MIN,
		MAGNITUDE_MAX,
		MAGNITUDE_MIN,
		MAGNITUDE_MAX,
		MAGNITUDE_DFLT,
		PF_Precision_INTEGER,
		0,
		0,
		MAGNITUDE_SLIDER);

	AEFX_CLR_STRUCT(def);

	// Wave type parameter (Sine or Triangle)
	PF_ADD_POPUP("Wave",
		2,                     // Number of choices
		1,                     // Default choice (1-based)
		"Sine|Triangle",       // Choices
		0,                     // Flags
		WAVE_TYPE_SLIDER);

	AEFX_CLR_STRUCT(def);

	// Phase parameter (wave offset)
	PF_ADD_FLOAT_SLIDERX("Phase",
		0.0,
		1000.0,
		0.0,
		1.0,
		0.0,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		PHASE_SLIDER);

	// Add Tiles group start
	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Tiles");
	def.flags = PF_ParamFlag_START_COLLAPSED;  // Start with group collapsed
	PF_ADD_PARAM(in_data, -1, &def);

	// Add X Tiles parameter as a checkbox
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("X Tiles",
		"",
		FALSE,
		0,
		X_TILES_DISK_ID);

	// Add Y Tiles parameter as a checkbox
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Y Tiles",
		"",
		FALSE,
		0,
		Y_TILES_DISK_ID);

	// Add Mirror checkbox
	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Mirror",
		"",
		FALSE,
		0,
		MIRROR_DISK_ID);

	// Add Tiles group end
	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_END;
	PF_ADD_PARAM(in_data, -1, &def);

	// Set total number of parameters
	out_data->num_params = RANDOMMOVE_NUM_PARAMS;

	return err;
}

// Check if there are any keyframes on the frequency parameter
bool HasAnyFrequencyKeyframes(PF_InData* in_data)
{
	PF_Err err = PF_Err_NONE;
	bool has_keyframes = false;

	AEGP_SuiteHandler suites(in_data->pica_basicP);

	// Get the effect reference
	AEGP_EffectRefH effect_ref = NULL;
	AEGP_StreamRefH stream_ref = NULL;
	A_long num_keyframes = 0;

	// Get the effect reference
	if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
		AEGP_EffectRefH aegp_effect_ref = NULL;
		err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(NULL, in_data->effect_ref, &aegp_effect_ref);

		if (!err && aegp_effect_ref) {
			// Get the stream for the frequency parameter
			err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL,
				aegp_effect_ref,
				FREQUENCY_SLIDER,
				&stream_ref);

			if (!err && stream_ref) {
				// Check how many keyframes are on this stream
				err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(stream_ref, &num_keyframes);

				// If there are any keyframes, set the flag
				if (!err && num_keyframes > 0) {
					has_keyframes = true;
				}

				// Dispose of the stream reference
				suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
			}

			// Dispose of the effect reference
			suites.EffectSuite4()->AEGP_DisposeEffect(aegp_effect_ref);
		}
	}

	return has_keyframes;
}

// GPU data initialized at GPU setup and used during render.
struct OpenCLGPUData
{
	cl_kernel oscillate_kernel;
};

#if HAS_HLSL
inline PF_Err DXErr(bool inSuccess) {
	if (inSuccess) { return PF_Err_NONE; }
	else { return PF_Err_INTERNAL_STRUCT_DAMAGED; }
}
#define DX_ERR(FUNC) ERR(DXErr(FUNC))
#include "DirectXUtils.h"
struct DirectXGPUData
{
	DXContextPtr mContext;
	ShaderObjectPtr mOscillateShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState>oscillate_pipeline;
};

PF_Err NSError2PFErr(NSError* inError)
{
	if (inError)
	{
		return PF_Err_INTERNAL_STRUCT_DAMAGED;  // For debugging, uncomment above line and set breakpoint here
	}
	return PF_Err_NONE;
}
#endif // HAS_METAL

static PF_Err
GPUDeviceSetup(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_GPUDeviceSetupExtra* extraP)
{
	PF_Err err = PF_Err_NONE;

	PF_GPUDeviceInfo device_info;
	AEFX_CLR_STRUCT(device_info);

	AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
		kPFHandleSuite,
		kPFHandleSuiteVersion1,
		out_dataP);

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpuDeviceSuite =
		AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP,
			kPFGPUDeviceSuite,
			kPFGPUDeviceSuiteVersion1,
			out_dataP);

	gpuDeviceSuite->GetDeviceInfo(in_dataP->effect_ref,
		extraP->input->device_index,
		&device_info);

	// Load and compile the kernel - a real plugin would cache binaries to disk

	if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
		// Nothing to do here. CUDA Kernel statically linked
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
	else if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
		PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(OpenCLGPUData));
		OpenCLGPUData* cl_gpu_data = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_int result = CL_SUCCESS;

		char const* k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";

		size_t sizes[] = { strlen(k16fString), strlen(kOscillateKernel_OpenCLString) };
		char const* strings[] = { k16fString, kOscillateKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->oscillate_kernel = clCreateKernel(program, "OscillateKernel", &result);
			CL_ERR(result);
		}

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#if HAS_HLSL
	else if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
	{
		PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(DirectXGPUData));
		DirectXGPUData* dx_gpu_data = reinterpret_cast<DirectXGPUData*>(*gpu_dataH);
		memset(dx_gpu_data, 0, sizeof(DirectXGPUData));

		// Create objects
		dx_gpu_data->mContext = std::make_shared<DXContext>();
		dx_gpu_data->mOscillateShader = std::make_shared<ShaderObject>();

		// Create the DXContext
		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"OscillateKernel", csoPath, sigPath));

		// Load the shader
		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mOscillateShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		// Create a library from source
		NSString* source = [NSString stringWithCString : kOscillateKernel_MetalString encoding : NSUTF8StringEncoding];
		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

		NSError* error = nil;
		id<MTLLibrary> library = [[device newLibraryWithSource : source options : nil error : &error]autorelease];

		// An error code is set for Metal compile warnings, so use nil library as the error signal
		if (!err && !library) {
			err = NSError2PFErr(error);
		}

		// For debugging only. This will contain Metal compile warnings and errors.
		NSString* getError = error.localizedDescription;

		PF_Handle metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
		MetalGPUData* metal_data = reinterpret_cast<MetalGPUData*>(*metal_handle);

		// Create pipeline state from function extracted from library
		if (err == PF_Err_NONE)
		{
			id<MTLFunction> oscillate_function = nil;
			NSString* oscillate_name = [NSString stringWithCString : "OscillateKernel" encoding : NSUTF8StringEncoding];

			oscillate_function = [[library newFunctionWithName : oscillate_name]autorelease];

			if (!oscillate_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->oscillate_pipeline = [device newComputePipelineStateWithFunction : oscillate_function error : &error];
				err = NSError2PFErr(error);
			}

			if (!err) {
				extraP->output->gpu_data = metal_handle;
				out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
			}
		}
	}
#endif
	return err;
}

static PF_Err
GPUDeviceSetdown(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_GPUDeviceSetdownExtra* extraP)
{
	PF_Err err = PF_Err_NONE;

	if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		(void)clReleaseKernel(cl_gpu_dataP->oscillate_kernel);

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#if HAS_HLSL
	else if (extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		DirectXGPUData* dx_gpu_dataP = reinterpret_cast<DirectXGPUData*>(*gpu_dataH);

		// Note: DirectX: If deferred execution is implemented, a GPU sync is
		// necessary before the plugin shutdown.
		dx_gpu_dataP->mContext.reset();
		dx_gpu_dataP->mOscillateShader.reset();

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		PF_Handle metal_handle = (PF_Handle)extraP->input->gpu_data;

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(metal_handle);
	}
#endif
	return err;
}

// Triangle wave function
static PF_FpLong TriangleWave(PF_FpLong t)
{
	// Shift phase by 0.75 and normalize to [0,1]
	t = fmod(t + 0.75, 1.0);

	// Handle negative values
	if (t < 0)
		t += 1.0;

	// Transform to triangle wave [-1,1]
	return (fabs(t - 0.5) - 0.25) * 4.0;
}

// Calculate wave value (sine or triangle)
static PF_FpLong CalculateWaveValue(int wave_type, PF_FpLong frequency, PF_FpLong current_time, PF_FpLong phase)
{
	PF_FpLong X, m;

	if (wave_type == 0) {
		// Sine wave
		X = (frequency * 2.0 * current_time) + (phase * 2.0);
		m = sin(X * M_PI);
	}
	else {
		// Triangle wave
		X = ((frequency * 2.0 * current_time) + (phase * 2.0)) / 2.0 + phase;
		m = TriangleWave(X);
	}

	return m;
}

static PF_Err
SmartPreRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PreRenderExtra* extra)
{
	PF_Err err = PF_Err_NONE;

	// Initialize effect info structure
	RandomMoveInfo info;
	AEFX_CLR_STRUCT(info);

	PF_ParamDef param_copy;
	AEFX_CLR_STRUCT(param_copy);

	// Get parameters (we still need these for rendering)
	ERR(PF_CHECKOUT_PARAM(in_data, DIRECTION_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.direction = param_copy.u.pd.value - 1;

	ERR(PF_CHECKOUT_PARAM(in_data, ANGLE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.angle = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, FREQUENCY_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.frequency = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, MAGNITUDE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.magnitude = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, WAVE_TYPE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.wave_type = param_copy.u.pd.value - 1;

	ERR(PF_CHECKOUT_PARAM(in_data, PHASE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.phase = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, X_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.x_tiles = param_copy.u.bd.value;

	ERR(PF_CHECKOUT_PARAM(in_data, Y_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.y_tiles = param_copy.u.bd.value;

	ERR(PF_CHECKOUT_PARAM(in_data, MIRROR_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.mirror = param_copy.u.bd.value;

	// Check if there are any frequency keyframes
	bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

	// Calculate time offset for animation
	A_long time_offset = 0;
	if (has_frequency_keyframes) {
		time_offset = in_data->time_step / 2; // 0.5 frames in time units
	}

	// Set up render request
	PF_RenderRequest req = extra->input->output_request;
	req.preserve_rgb_of_zero_alpha = TRUE;

	// Checkout the input layer
	PF_CheckoutResult checkout;
	ERR(extra->cb->checkout_layer(in_data->effect_ref,
		RANDOMMOVE_INPUT,
		RANDOMMOVE_INPUT,
		&req,
		in_data->current_time,
		in_data->time_step,
		in_data->time_scale,
		&checkout));

	if (!err) {
		// Create detection data structure for GUID mixing
		struct {
			A_u_char has_frequency_keyframes;
			A_long time_offset;
			RandomMoveInfo info;
		} detection_data;

		detection_data.has_frequency_keyframes = has_frequency_keyframes ? 1 : 0;
		detection_data.time_offset = time_offset;
		detection_data.info = info;

		// Mix in the data for caching
		ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(detection_data), &detection_data));

		// Update output parameters
		extra->output->max_result_rect = checkout.max_result_rect;
		extra->output->result_rect = checkout.result_rect;
		extra->output->solid = FALSE;
		extra->output->pre_render_data = NULL;
		extra->output->flags = PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
	}

	return err;
}

template <typename PixelType>
static PixelType
SampleBilinear(PixelType* src, PF_FpLong x, PF_FpLong y, A_long width, A_long height, A_long rowbytes,
	bool x_tiles, bool y_tiles, bool mirror) {

	// Initialize variables to track if we're sampling outside the image
	bool outsideBounds = false;

	// Handle tiling based on X and Y Tiles parameters
	if (x_tiles) {
		// X tiling is enabled
		if (mirror) {
			// Mirror tiling: create ping-pong pattern
			float intPart;
			float fracPart = modff(fabsf(x / width), &intPart);
			int isOdd = (int)intPart & 1;
			x = isOdd ? (1.0f - fracPart) * width : fracPart * width;
		}
		else {
			// Regular repeat tiling
			x = fmod(fmod(x, width) + width, width);
		}
	}
	else {
		// X tiling is disabled - check if outside bounds
		if (x < 0 || x >= width) {
			outsideBounds = true;
		}
	}

	// Apply Y tiling
	if (y_tiles) {
		// Y tiling is enabled
		if (mirror) {
			// Mirror tiling: create ping-pong pattern
			float intPart;
			float fracPart = modff(fabsf(y / height), &intPart);
			int isOdd = (int)intPart & 1;
			y = isOdd ? (1.0f - fracPart) * height : fracPart * height;
		}
		else {
			// Regular repeat tiling
			y = fmod(fmod(y, height) + height, height);
		}
	}
	else {
		// Y tiling is disabled - check if outside bounds
		if (y < 0 || y >= height) {
			outsideBounds = true;
		}
	}

	// If we're outside bounds and tiling is disabled, return transparent pixel
	if (outsideBounds) {
		PixelType transparent;
		if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
			transparent.alpha = 0.0f;
			transparent.red = 0.0f;
			transparent.green = 0.0f;
			transparent.blue = 0.0f;
		}
		else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
			transparent.alpha = 0;
			transparent.red = 0;
			transparent.green = 0;
			transparent.blue = 0;
		}
		else {
			transparent.alpha = 0;
			transparent.red = 0;
			transparent.green = 0;
			transparent.blue = 0;
		}
		return transparent;
	}

	// At this point, we're guaranteed to be within bounds or using tiling
	// Clamp coordinates to valid range to avoid any out-of-bounds access
	x = MAX(0, MIN(width - 1.001f, x));
	y = MAX(0, MIN(height - 1.001f, y));

	// Get integer and fractional parts
	A_long x0 = static_cast<A_long>(x);
	A_long y0 = static_cast<A_long>(y);
	A_long x1 = MIN(x0 + 1, width - 1);
	A_long y1 = MIN(y0 + 1, height - 1);

	PF_FpLong fx = x - x0;
	PF_FpLong fy = y - y0;

	// Get the four surrounding pixels
	PixelType* p00 = (PixelType*)((char*)src + y0 * rowbytes) + x0;
	PixelType* p01 = (PixelType*)((char*)src + y0 * rowbytes) + x1;
	PixelType* p10 = (PixelType*)((char*)src + y1 * rowbytes) + x0;
	PixelType* p11 = (PixelType*)((char*)src + y1 * rowbytes) + x1;

	// Bilinear interpolation for each channel
	PixelType result;
	if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
		result.alpha = (1.0f - fx) * (1.0f - fy) * p00->alpha +
			fx * (1.0f - fy) * p01->alpha +
			(1.0f - fx) * fy * p10->alpha +
			fx * fy * p11->alpha;

		result.red = (1.0f - fx) * (1.0f - fy) * p00->red +
			fx * (1.0f - fy) * p01->red +
			(1.0f - fx) * fy * p10->red +
			fx * fy * p11->red;

		result.green = (1.0f - fx) * (1.0f - fy) * p00->green +
			fx * (1.0f - fy) * p01->green +
			(1.0f - fx) * fy * p10->green +
			fx * fy * p11->green;

		result.blue = (1.0f - fx) * (1.0f - fy) * p00->blue +
			fx * (1.0f - fy) * p01->blue +
			(1.0f - fx) * fy * p10->blue +
			fx * fy * p11->blue;
	}
	else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
		result.alpha = static_cast<A_u_short>(
			(1.0f - fx) * (1.0f - fy) * p00->alpha +
			fx * (1.0f - fy) * p01->alpha +
			(1.0f - fx) * fy * p10->alpha +
			fx * fy * p11->alpha + 0.5f);

		result.red = static_cast<A_u_short>(
			(1.0f - fx) * (1.0f - fy) * p00->red +
			fx * (1.0f - fy) * p01->red +
			(1.0f - fx) * fy * p10->red +
			fx * fy * p11->red + 0.5f);

		result.green = static_cast<A_u_short>(
			(1.0f - fx) * (1.0f - fy) * p00->green +
			fx * (1.0f - fy) * p01->green +
			(1.0f - fx) * fy * p10->green +
			fx * fy * p11->green + 0.5f);

		result.blue = static_cast<A_u_short>(
			(1.0f - fx) * (1.0f - fy) * p00->blue +
			fx * (1.0f - fy) * p01->blue +
			(1.0f - fx) * fy * p10->blue +
			fx * fy * p11->blue + 0.5f);
	}
	else {
		result.alpha = static_cast<A_u_char>(
			(1.0f - fx) * (1.0f - fy) * p00->alpha +
			fx * (1.0f - fy) * p01->alpha +
			(1.0f - fx) * fy * p10->alpha +
			fx * fy * p11->alpha + 0.5f);

		result.red = static_cast<A_u_char>(
			(1.0f - fx) * (1.0f - fy) * p00->red +
			fx * (1.0f - fy) * p01->red +
			(1.0f - fx) * fy * p10->red +
			fx * fy * p11->red + 0.5f);

		result.green = static_cast<A_u_char>(
			(1.0f - fx) * (1.0f - fy) * p00->green +
			fx * (1.0f - fy) * p01->green +
			(1.0f - fx) * fy * p10->green +
			fx * fy * p11->green + 0.5f);

		result.blue = static_cast<A_u_char>(
			(1.0f - fx) * (1.0f - fy) * p00->blue +
			fx * (1.0f - fy) * p01->blue +
			(1.0f - fx) * fy * p10->blue +
			fx * fy * p11->blue + 0.5f);
	}

	return result;
}

static PF_Err
SmartRenderGPU(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	RandomMoveInfo* infoP,
	PF_FpLong current_time)
{
	PF_Err err = PF_Err_NONE;

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite = AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP,
		kPFGPUDeviceSuite,
		kPFGPUDeviceSuiteVersion1,
		out_dataP);

	if (pixel_format != PF_PixelFormat_GPU_BGRA128) {
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
		return err;
	}

	A_long bytes_per_pixel = 16;

	PF_GPUDeviceInfo device_info;
	ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref, extraP->input->device_index, &device_info));

	void* src_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

	void* dst_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

	// Set up parameters for the kernel
	typedef struct {
		int srcPitch;
		int dstPitch;
		int is16f;
		int width;
		int height;
		float angle;
		float frequency;
		float magnitude;
		int direction;
		int waveType;
		float phase;
		float currentTime;
		int xTiles;
		int yTiles;
		int mirror;
	} OscillateParams;

	OscillateParams params;

	// Initialize parameters
	params.width = input_worldP->width;
	params.height = input_worldP->height;
	params.is16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
	params.srcPitch = input_worldP->rowbytes / bytes_per_pixel;
	params.dstPitch = output_worldP->rowbytes / bytes_per_pixel;
	params.angle = infoP->angle;
	params.frequency = infoP->frequency;
	params.magnitude = infoP->magnitude;
	params.direction = infoP->direction;
	params.waveType = infoP->wave_type;
	params.phase = infoP->phase;
	params.currentTime = current_time;
	params.xTiles = infoP->x_tiles;
	params.yTiles = infoP->y_tiles;
	params.mirror = infoP->mirror;

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

		// Set the arguments
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.srcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.dstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.is16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.width));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.height));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.angle));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.frequency));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.magnitude));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.direction));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.waveType));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.phase));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.currentTime));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.xTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.yTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.mirror));

		// Launch the kernel
		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = {
			((params.width + threadBlock[0] - 1) / threadBlock[0]) * threadBlock[0],
			((params.height + threadBlock[1] - 1) / threadBlock[1]) * threadBlock[1]
		};

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->oscillate_kernel,
			2,
			0,
			grid,
			threadBlock,
			0,
			0,
			0));
	}
#if HAS_CUDA
	else if (!err && extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
		Oscillate_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			params.srcPitch,
			params.dstPitch,
			params.is16f,
			params.width,
			params.height,
			params.angle,
			params.frequency,
			params.magnitude,
			params.direction,
			params.waveType,
			params.phase,
			params.currentTime,
			params.xTiles,
			params.yTiles,
			params.mirror);

		if (cudaPeekAtLastError() != cudaSuccess) {
			err = PF_Err_INTERNAL_STRUCT_DAMAGED;
		}
	}
#endif
#if HAS_HLSL
	else if (!err && extraP->input->what_gpu == PF_GPU_Framework_DIRECTX)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		DirectXGPUData* dx_gpu_data = reinterpret_cast<DirectXGPUData*>(*gpu_dataH);

		// Execute Oscillate
		DXShaderExecution shaderExecution(
			dx_gpu_data->mContext,
			dx_gpu_data->mOscillateShader,
			3);

		// Note: The order of elements in the param structure should be identical to the order expected by the shader
		DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(OscillateParams)));
		DX_ERR(shaderExecution.SetUnorderedAccessView(
			(ID3D12Resource*)dst_mem,
			params.height * output_worldP->rowbytes));
		DX_ERR(shaderExecution.SetShaderResourceView(
			(ID3D12Resource*)src_mem,
			params.height * input_worldP->rowbytes));
		DX_ERR(shaderExecution.Execute(
			(UINT)((params.width + 15) / 16),
			(UINT)((params.height + 15) / 16)));
	}
#endif
#if HAS_METAL
	else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		PF_Handle metal_handle = (PF_Handle)extraP->input->gpu_data;
		MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

		// Set the arguments
		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &params
			length : sizeof(OscillateParams)
			options : MTLResourceStorageModeManaged]autorelease];

		// Launch the command
		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->oscillate_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = {
			((params.width + threadsPerGroup.width - 1) / threadsPerGroup.width),
			((params.height + threadsPerGroup.height - 1) / threadsPerGroup.height),
			1
		};

		[computeEncoder setComputePipelineState : metal_dataP->oscillate_pipeline] ;
		[computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
		[computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
		[computeEncoder setBuffer : param_buffer offset : 0 atIndex : 2] ;
		[computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;
		[computeEncoder endEncoding] ;
		[commandBuffer commit] ;

		err = NSError2PFErr([commandBuffer error]);
	}
#endif // HAS_METAL

	return err;
}

static PF_Err
SmartRenderCPU(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	RandomMoveInfo* infoP,
	PF_FpLong current_time)
{
	PF_Err err = PF_Err_NONE;

	// Calculate angle in radians for direction vector
	PF_FpLong angleRad = infoP->angle * M_PI / 180.0;
	PF_FpLong dx = cos(angleRad);
	PF_FpLong dy = sin(angleRad);

	// Calculate wave value
	PF_FpLong m = CalculateWaveValue(infoP->wave_type,
		infoP->frequency,
		current_time,
		infoP->phase);

	// Initialize transformation values
	PF_FpLong offsetX = 0, offsetY = 0;
	PF_FpLong scale = 100.0;

	// Apply effect based on direction mode
	switch (infoP->direction) {
	case 0: // Angle mode - position offset only
		offsetX = dx * infoP->magnitude * m;
		offsetY = dy * infoP->magnitude * m;
		break;

	case 1: // Depth mode - scale only
		scale = 100.0 - (infoP->magnitude * m * 0.1);
		break;

	case 2: { // Orbit mode - position offset and scale
		offsetX = dx * infoP->magnitude * m;
		offsetY = dy * infoP->magnitude * m;

		// Calculate second wave with phase shift for scale
		PF_FpLong phaseShift = infoP->wave_type == 0 ? 0.25 : 0.125;
		m = CalculateWaveValue(infoP->wave_type,
			infoP->frequency,
			current_time,
			infoP->phase + phaseShift);

		scale = 100.0 - (infoP->magnitude * m * 0.1);
		break;
	}
	}

	// Iterate through all pixels in the output buffer
	A_long width = output_worldP->width;
	A_long height = output_worldP->height;
	A_long input_width = input_worldP->width;
	A_long input_height = input_worldP->height;
	A_long rowbytes = input_worldP->rowbytes;

	// Calculate center point
	PF_FpLong centerX = (width / 2.0);
	PF_FpLong centerY = (height / 2.0);
	PF_FpLong inputCenterX = (input_width / 2.0);
	PF_FpLong inputCenterY = (input_height / 2.0);

	// Calculate scale factor
	PF_FpLong scaleFactorX = 100.0 / scale;
	PF_FpLong scaleFactorY = 100.0 / scale;

	// Process based on pixel format
	switch (pixel_format) {
	case PF_PixelFormat_ARGB128: {
		// 32-bit float processing
		for (A_long y = 0; y < height; y++) {
			PF_PixelFloat* outP = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				// Calculate source coordinates with transformation
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

				// Sample from the input with tiling
				*outP = SampleBilinear<PF_PixelFloat>(
					(PF_PixelFloat*)input_worldP->data,
					srcX, srcY,
					input_width, input_height,
					rowbytes,
					infoP->x_tiles,
					infoP->y_tiles,
					infoP->mirror
				);
			}
		}
		break;
	}
	case PF_PixelFormat_ARGB64: {
		// 16-bit processing
		for (A_long y = 0; y < height; y++) {
			PF_Pixel16* outP = (PF_Pixel16*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				// Calculate source coordinates with transformation
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

				// Sample from the input with tiling
				*outP = SampleBilinear<PF_Pixel16>(
					(PF_Pixel16*)input_worldP->data,
					srcX, srcY,
					input_width, input_height,
					rowbytes,
					infoP->x_tiles,
					infoP->y_tiles,
					infoP->mirror
				);
			}
		}
		break;
	}
	case PF_PixelFormat_ARGB32:
	default: {
		// 8-bit processing
		for (A_long y = 0; y < height; y++) {
			PF_Pixel8* outP = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				// Calculate source coordinates with transformation
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

				// Sample from the input with tiling
				*outP = SampleBilinear<PF_Pixel8>(
					(PF_Pixel8*)input_worldP->data,
					srcX, srcY,
					input_width, input_height,
					rowbytes,
					infoP->x_tiles,
					infoP->y_tiles,
					infoP->mirror
				);
			}
		}
		break;
	}
	}

	return err;
}

static PF_Err
SmartRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_SmartRenderExtra* extraP,
	bool isGPU)
{
	PF_Err err = PF_Err_NONE;
	PF_Err err2 = PF_Err_NONE;

	PF_EffectWorld* input_worldP = NULL;
	PF_EffectWorld* output_worldP = NULL;

	// Get parameters for rendering
	RandomMoveInfo info;
	AEFX_CLR_STRUCT(info);

	PF_ParamDef param_copy;
	AEFX_CLR_STRUCT(param_copy);

	// Get all effect parameters
	ERR(PF_CHECKOUT_PARAM(in_data, DIRECTION_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.direction = param_copy.u.pd.value - 1;

	ERR(PF_CHECKOUT_PARAM(in_data, ANGLE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.angle = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, FREQUENCY_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.frequency = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, MAGNITUDE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.magnitude = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, WAVE_TYPE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.wave_type = param_copy.u.pd.value - 1;

	ERR(PF_CHECKOUT_PARAM(in_data, PHASE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.phase = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, X_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.x_tiles = param_copy.u.bd.value;

	ERR(PF_CHECKOUT_PARAM(in_data, Y_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.y_tiles = param_copy.u.bd.value;

	ERR(PF_CHECKOUT_PARAM(in_data, MIRROR_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.mirror = param_copy.u.bd.value;

	// Check if there are any frequency keyframes
	bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

	// Convert current time to seconds for wave calculations
	PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;

	// If there are any frequency keyframes, always advance time by half a frame
	if (has_frequency_keyframes) {
		// Shift by half a frame in time units
		A_long time_shift = in_data->time_step / 2;

		// Create a new time value with the shift applied
		A_Time shifted_time;
		shifted_time.value = in_data->current_time + time_shift;
		shifted_time.scale = in_data->time_scale;

		// Convert to seconds for the calculation
		current_time = (PF_FpLong)shifted_time.value / (PF_FpLong)shifted_time.scale;
	}

	if (!err) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, RANDOMMOVE_INPUT, &input_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		if (!err && input_worldP && output_worldP) {
			AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
				kPFWorldSuite,
				kPFWorldSuiteVersion2,
				out_data);

			PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
			ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

			if (isGPU) {
				ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, &info, current_time));
			}
			else {
				ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, &info, current_time));
			}
		}
	}

	// Check in the input layer pixels
	if (input_worldP) {
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, RANDOMMOVE_INPUT));
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
	PF_Err err = PF_Err_NONE;

	// This is a fallback for older versions that don't support Smart Render
	// Get parameters for rendering
	RandomMoveInfo info;
	AEFX_CLR_STRUCT(info);

	info.direction = params[DIRECTION_SLIDER]->u.pd.value - 1;
	info.angle = params[ANGLE_SLIDER]->u.fs_d.value;
	info.frequency = params[FREQUENCY_SLIDER]->u.fs_d.value;
	info.magnitude = params[MAGNITUDE_SLIDER]->u.fs_d.value;
	info.wave_type = params[WAVE_TYPE_SLIDER]->u.pd.value - 1;
	info.phase = params[PHASE_SLIDER]->u.fs_d.value;

	// Get tiling parameters
	info.x_tiles = params[X_TILES_DISK_ID]->u.bd.value;
	info.y_tiles = params[Y_TILES_DISK_ID]->u.bd.value;
	info.mirror = params[MIRROR_DISK_ID]->u.bd.value;

	// Check if there are any frequency keyframes
	bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

	// Convert current time to seconds for wave calculations
	PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;

	// If there are any frequency keyframes, always advance time by half a frame
	if (has_frequency_keyframes) {
		// Shift by half a frame in time units
		A_long time_shift = in_data->time_step / 2;

		// Create a new time value with the shift applied
		A_Time shifted_time;
		shifted_time.value = in_data->current_time + time_shift;
		shifted_time.scale = in_data->time_scale;

		// Convert to seconds for the calculation
		current_time = (PF_FpLong)shifted_time.value / (PF_FpLong)shifted_time.scale;
	}

	// Calculate angle in radians for direction vector
	PF_FpLong angleRad = info.angle * M_PI / 180.0;
	PF_FpLong dx = cos(angleRad);
	PF_FpLong dy = sin(angleRad);

	// Calculate wave value based on time, frequency and phase
	PF_FpLong X;
	PF_FpLong m;

	// Calculate wave value (sine or triangle)
	if (info.wave_type == 0) {
		// Sine wave
		X = (info.frequency * 2.0 * current_time) + (info.phase * 2.0);
		m = sin(X * M_PI);
	}
	else {
		// Triangle wave
		X = ((info.frequency * 2.0 * current_time) + (info.phase * 2.0)) / 2.0 + info.phase;
		m = TriangleWave(X);
	}

	// Initialize transformation values
	PF_FpLong offsetX = 0, offsetY = 0;
	PF_FpLong scale = 100.0;

	// Apply effect based on direction mode
	switch (info.direction) {
	case 0: // Angle mode - position offset only
		offsetX = dx * info.magnitude * m;
		offsetY = dy * info.magnitude * m;
		break;

	case 1: // Depth mode - scale only
		scale = 100.0 - (info.magnitude * m * 0.1);
		break;

	case 2: { // Orbit mode - position offset and scale
		offsetX = dx * info.magnitude * m;
		offsetY = dy * info.magnitude * m;

		// Calculate second wave with phase shift for scale
		PF_FpLong phaseShift = info.wave_type == 0 ? 0.25 : 0.125;
		PF_FpLong X2;

		if (info.wave_type == 0) {
			X2 = (info.frequency * 2.0 * current_time) + ((info.phase + phaseShift) * 2.0);
			m = sin(X2 * M_PI);
		}
		else {
			X2 = ((info.frequency * 2.0 * current_time) + ((info.phase + phaseShift) * 2.0)) / 2.0 + (info.phase + phaseShift);
			m = TriangleWave(X2);
		}
		scale = 100.0 - (info.magnitude * m * 0.1);
		break;
	}
	}

	// Get pixel format using PF_WorldSuite
	PF_PixelFormat pixelFormat;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	PF_WorldSuite2* wsP = NULL;
	ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
	if (!err) {
		ERR(wsP->PF_GetPixelFormat(output, &pixelFormat));
		suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
	}

	// Iterate through all pixels in the output buffer
	A_long width = output->width;
	A_long height = output->height;
	A_long input_width = params[RANDOMMOVE_INPUT]->u.ld.width;
	A_long input_height = params[RANDOMMOVE_INPUT]->u.ld.height;
	A_long rowbytes = params[RANDOMMOVE_INPUT]->u.ld.rowbytes;

	// Calculate center point
	PF_FpLong centerX = (width / 2.0);
	PF_FpLong centerY = (height / 2.0);
	PF_FpLong inputCenterX = (input_width / 2.0);
	PF_FpLong inputCenterY = (input_height / 2.0);

	// Calculate scale factor
	PF_FpLong scaleFactorX = 100.0 / scale;
	PF_FpLong scaleFactorY = 100.0 / scale;

	// Process based on pixel format
	switch (pixelFormat) {
	case PF_PixelFormat_ARGB128: {
		// 32-bit float processing
		for (A_long y = 0; y < height; y++) {
			PF_PixelFloat* outP = (PF_PixelFloat*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				// Calculate source coordinates with transformation
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

				// Sample from the input with tiling
				*outP = SampleBilinear<PF_PixelFloat>(
					(PF_PixelFloat*)params[RANDOMMOVE_INPUT]->u.ld.data,
					srcX, srcY,
					input_width, input_height,
					rowbytes,
					info.x_tiles,
					info.y_tiles,
					info.mirror
				);
			}
		}
		break;
	}
	case PF_PixelFormat_ARGB64: {
		// 16-bit processing
		for (A_long y = 0; y < height; y++) {
			PF_Pixel16* outP = (PF_Pixel16*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				// Calculate source coordinates with transformation
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

				// Sample from the input with tiling
				*outP = SampleBilinear<PF_Pixel16>(
					(PF_Pixel16*)params[RANDOMMOVE_INPUT]->u.ld.data,
					srcX, srcY,
					input_width, input_height,
					rowbytes,
					info.x_tiles,
					info.y_tiles,
					info.mirror
				);
			}
		}
		break;
	}
	case PF_PixelFormat_ARGB32:
	default: {
		// 8-bit processing
		for (A_long y = 0; y < height; y++) {
			PF_Pixel8* outP = (PF_Pixel8*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				// Calculate source coordinates with transformation
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

				// Sample from the input with tiling
				*outP = SampleBilinear<PF_Pixel8>(
					(PF_Pixel8*)params[RANDOMMOVE_INPUT]->u.ld.data,
					srcX, srcY,
					input_width, input_height,
					rowbytes,
					info.x_tiles,
					info.y_tiles,
					info.mirror
				);
			}
		}
		break;
	}
	}

	return err;
}

extern "C" DllExport
PF_Err PluginDataEntryFunction(
	PF_PluginDataPtr inPtr,
	PF_PluginDataCB inPluginDataCallBackPtr,
	SPBasicSuite* inSPBasicSuitePtr,
	const char* inHostName,
	const char* inHostVersion)
{
	PF_Err result = PF_Err_INVALID_CALLBACK;

	// Register the effect with After Effects
	result = PF_REGISTER_EFFECT(
		inPtr,
		inPluginDataCallBackPtr,
		"Oscillate",          // Effect name
		"DKT Oscillate",      // Match name - make sure this is unique
		"DKT Effects",    // Category
		AE_RESERVED_INFO
	);

	return result;
}

PF_Err EffectMain(
	PF_Cmd cmd,
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output,
	void* extra)
{
	PF_Err err = PF_Err_NONE;

	try {
		switch (cmd) {
		case PF_Cmd_ABOUT:
			err = About(in_data, out_data, params, output);
			break;
		case PF_Cmd_GLOBAL_SETUP:
			err = GlobalSetup(in_data, out_data, params, output);
			break;
		case PF_Cmd_PARAMS_SETUP:
			err = ParamsSetup(in_data, out_data, params, output);
			break;
		case PF_Cmd_SMART_PRE_RENDER:
			err = SmartPreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
			break;
		case PF_Cmd_SMART_RENDER:
			err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, false);
			break;
		case PF_Cmd_SMART_RENDER_GPU:
			err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, true);
			break;
		case PF_Cmd_RENDER:
			// Fallback for older versions that don't support Smart Render
			err = Render(in_data, out_data, params, output);
			break;
		case PF_Cmd_USER_CHANGED_PARAM:
		case PF_Cmd_UPDATE_PARAMS_UI:
			err = UpdateParameterUI(in_data, out_data, params);
			break;
		case PF_Cmd_GPU_DEVICE_SETUP:
			err = GPUDeviceSetup(in_data, out_data, (PF_GPUDeviceSetupExtra*)extra);
			break;
		case PF_Cmd_GPU_DEVICE_SETDOWN:
			err = GPUDeviceSetdown(in_data, out_data, (PF_GPUDeviceSetdownExtra*)extra);
			break;
		}
	}
	catch (PF_Err& thrown_err) {
		err = thrown_err;
	}
	catch (...) {
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	return err;
}


