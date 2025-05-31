#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "PulseSize.h"
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

inline PF_Err CL2Err(cl_int cl_result) {
	if (cl_result == CL_SUCCESS) {
		return PF_Err_NONE;
	}
	else {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))

extern void PulseSize_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float frequency,
	float shrink,
	float grow,
	int waveType,
	float phase,
	float currentTime,
	int xTiles,
	int yTiles,
	int mirror,
	int normalEnabled,
	int compatibilityEnabled,
	float compatFrequency,
	float compatShrink,
	float compatGrow,
	float compatPhase,
	int compatWaveType,
	float accumulatedPhase,
	int hasFrequencyKeyframes);

static PF_Err
About(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
		"Pulse Size v%d.%d\r"
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

	out_data->my_version = PF_VERSION(MAJOR_VERSION,
		MINOR_VERSION,
		BUG_VERSION,
		STAGE_VERSION,
		BUILD_VERSION);

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

static PF_Err
ParamsSetup(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err err = PF_Err_NONE;
	PF_ParamDef def;

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Normal",
		"",
		TRUE,
		0,
		NORMAL_CHECKBOX_ID);

	AEFX_CLR_STRUCT(def);
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
	PF_ADD_FLOAT_SLIDERX("Shrink",
		SHRINK_MIN,
		SHRINK_MAX,
		SHRINK_MIN,
		SHRINK_MAX,
		SHRINK_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		SHRINK_SLIDER);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Grow",
		GROW_MIN,
		GROW_MAX,
		GROW_MIN,
		GROW_MAX,
		GROW_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		GROW_SLIDER);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Phase",
		PHASE_MIN,
		PHASE_MAX,
		PHASE_MIN,
		PHASE_MAX,
		PHASE_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		PHASE_SLIDER);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Wave",
		2,
		1,
		"Sine|Triangle",
		0,
		WAVE_TYPE_SLIDER);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Tiles");
	def.flags = PF_ParamFlag_START_COLLAPSED;
	PF_ADD_PARAM(in_data, -1, &def);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("X Tiles",
		"",
		FALSE,
		0,
		X_TILES_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Y Tiles",
		"",
		FALSE,
		0,
		Y_TILES_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Mirror",
		"",
		FALSE,
		0,
		MIRROR_DISK_ID);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_END;
	PF_ADD_PARAM(in_data, -1, &def);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Compatibility");
	def.flags = PF_ParamFlag_START_COLLAPSED;
	PF_ADD_PARAM(in_data, -1, &def);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Compatibility",
		"",
		FALSE,
		0,
		COMPATIBILITY_CHECKBOX_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Frequency",
		0.1f,
		16.0f,
		0.1f,
		16.0f,
		2.0f,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_FREQUENCY_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Shrink",
		0.0f,
		1.0f,
		0.0f,
		1.0f,
		0.9f,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_SHRINK_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Grow",
		1.0f,
		2.0f,
		1.0f,
		2.0f,
		1.1f,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_GROW_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Phase",
		0.0f,
		2.0f,
		0.0f,
		2.0f,
		0.0f,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_PHASE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Wave",
		2,
		1,
		"Sine|Triangle",
		0,
		COMPATIBILITY_WAVE_TYPE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_END;
	PF_ADD_PARAM(in_data, -1, &def);

	out_data->num_params = PULSESIZE_NUM_PARAMS;

	return err;
}

bool HasAnyFrequencyKeyframes(PF_InData* in_data)
{
	PF_Err err = PF_Err_NONE;
	bool has_keyframes = false;

	AEGP_SuiteHandler suites(in_data->pica_basicP);

	AEGP_EffectRefH effect_ref = NULL;
	AEGP_StreamRefH stream_ref = NULL;
	A_long num_keyframes = 0;

	if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
		AEGP_EffectRefH aegp_effect_ref = NULL;
		err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(NULL, in_data->effect_ref, &aegp_effect_ref);

		if (!err && aegp_effect_ref) {
			err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL,
				aegp_effect_ref,
				FREQUENCY_SLIDER,
				&stream_ref);

			if (!err && stream_ref) {
				err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(stream_ref, &num_keyframes);

				if (!err && num_keyframes > 0) {
					has_keyframes = true;
				}

				suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
			}

			suites.EffectSuite4()->AEGP_DisposeEffect(aegp_effect_ref);
		}
	}

	return has_keyframes;
}

PF_Err valueAtTime(
	PF_InData* in_data,
	int stream_index,
	float time_secs,
	PF_FpLong* value_out)
{
	PF_Err err = PF_Err_NONE;

	AEGP_SuiteHandler suites(in_data->pica_basicP);

	AEGP_EffectRefH aegp_effect_ref = NULL;
	AEGP_StreamRefH stream_ref = NULL;

	A_Time time;
	time.value = (A_long)(time_secs * in_data->time_scale);
	time.scale = in_data->time_scale;

	if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
		err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(
			NULL,
			in_data->effect_ref,
			&aegp_effect_ref);

		if (!err && aegp_effect_ref) {
			err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(
				NULL,
				aegp_effect_ref,
				stream_index,
				&stream_ref);

			if (!err && stream_ref) {
				AEGP_StreamValue2 stream_value;
				err = suites.StreamSuite5()->AEGP_GetNewStreamValue(
					NULL,
					stream_ref,
					AEGP_LTimeMode_LayerTime,
					&time,
					FALSE,
					&stream_value);

				if (!err) {
					AEGP_StreamType stream_type;
					err = suites.StreamSuite5()->AEGP_GetStreamType(stream_ref, &stream_type);

					if (!err) {
						switch (stream_type) {
						case AEGP_StreamType_OneD:
							*value_out = stream_value.val.one_d;
							break;

						case AEGP_StreamType_TwoD:
						case AEGP_StreamType_TwoD_SPATIAL:
							*value_out = stream_value.val.two_d.x;
							break;

						case AEGP_StreamType_ThreeD:
						case AEGP_StreamType_ThreeD_SPATIAL:
							*value_out = stream_value.val.three_d.x;
							break;

						case AEGP_StreamType_COLOR:
							*value_out = stream_value.val.color.redF;
							break;

						default:
							err = PF_Err_BAD_CALLBACK_PARAM;
							break;
						}
					}

					suites.StreamSuite5()->AEGP_DisposeStreamValue(&stream_value);
				}

				suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
			}

			suites.EffectSuite4()->AEGP_DisposeEffect(aegp_effect_ref);
		}
	}

	return err;
}

static PF_Err
valueAtTimeHz(
	PF_InData* in_data,
	int stream_index,
	float time_secs,
	float duration,
	ThreadRenderData* renderData,
	PF_FpLong* value_out)
{
	PF_Err err = PF_Err_NONE;

	err = valueAtTime(in_data, stream_index, time_secs, value_out);
	if (err) return err;

	if (stream_index == FREQUENCY_SLIDER) {
		bool isKeyed = HasAnyFrequencyKeyframes(in_data);

		if (isKeyed) {
			float fps = 120.0f;
			int totalSteps = (int)roundf(duration * fps);
			int curSteps = (int)roundf(fps * time_secs);

			renderData->accumulated_phase = 0.0f;
			for (int i = 0; i <= curSteps; i++) {
				PF_FpLong stepValue;
				float adjusted_time = (i / fps) + renderData->layer_start_seconds;
				err = valueAtTime(in_data, stream_index, adjusted_time, &stepValue);
				if (err) return err;

				renderData->accumulated_phase += stepValue / fps;
			}

			*value_out = renderData->accumulated_phase;
		}
	}

	return err;
}


struct OpenCLGPUData
{
	cl_kernel pulsesize_kernel;
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
	ShaderObjectPtr mPulseSizeShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState>pulsesize_pipeline;
};

PF_Err NSError2PFErr(NSError* inError)
{
	if (inError)
	{
		return PF_Err_INTERNAL_STRUCT_DAMAGED;           
	}
	return PF_Err_NONE;
}
#endif  

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

	if (extraP->input->what_gpu == PF_GPU_Framework_CUDA) {
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
	else if (extraP->input->what_gpu == PF_GPU_Framework_OPENCL) {
		PF_Handle gpu_dataH = handle_suite->host_new_handle(sizeof(OpenCLGPUData));
		OpenCLGPUData* cl_gpu_data = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_int result = CL_SUCCESS;

		char const* k16fString = "#define GF_OPENCL_SUPPORTS_16F 0\n";

		size_t sizes[] = { strlen(k16fString), strlen(kPulseSizeKernel_OpenCLString) };
		char const* strings[] = { k16fString, kPulseSizeKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->pulsesize_kernel = clCreateKernel(program, "PulseSizeKernel", &result);
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

		dx_gpu_data->mContext = std::make_shared<DXContext>();
		dx_gpu_data->mPulseSizeShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"PulseSizeKernel", csoPath, sigPath));

		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mPulseSizeShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		NSString* source = [NSString stringWithCString : kPulseSizeKernel_MetalString encoding : NSUTF8StringEncoding];
		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

		NSError* error = nil;
		id<MTLLibrary> library = [[device newLibraryWithSource : source options : nil error : &error]autorelease];

		if (!err && !library) {
			err = NSError2PFErr(error);
		}

		NSString* getError = error.localizedDescription;

		PF_Handle metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
		MetalGPUData* metal_data = reinterpret_cast<MetalGPUData*>(*metal_handle);

		if (err == PF_Err_NONE)
		{
			id<MTLFunction> pulsesize_function = nil;
			NSString* pulsesize_name = [NSString stringWithCString : "PulseSizeKernel" encoding : NSUTF8StringEncoding];

			pulsesize_function = [[library newFunctionWithName : pulsesize_name]autorelease];

			if (!pulsesize_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->pulsesize_pipeline = [device newComputePipelineStateWithFunction : pulsesize_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->pulsesize_kernel);

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

		dx_gpu_dataP->mContext.reset();
		dx_gpu_dataP->mPulseSizeShader.reset();

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

static PF_FpLong TriangleWave(PF_FpLong t)
{
	t = fmod(t + 0.75, 1.0);

	if (t < 0)
		t += 1.0;

	return (fabs(t - 0.5) - 0.25) * 4.0;
}

static PF_FpLong CalculateWaveValue(int wave_type, PF_FpLong frequency, PF_FpLong current_time, PF_FpLong phase)
{
	PF_FpLong X, m;

	if (wave_type == 0) {
		X = (frequency * current_time) + phase;
		m = sin(X * M_PI);
	}
	else {
		X = ((frequency * current_time) + phase) / 2.0 + phase;
		m = TriangleWave(X);
	}

	return m;
}

static void
DisposePrerenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		ThreadRenderData* renderData = reinterpret_cast<ThreadRenderData*>(pre_render_dataPV);
		free(renderData);
	}
}

static PF_Err
SmartPreRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PreRenderExtra* extra)
{
	PF_Err err = PF_Err_NONE;

	AEGP_LayerIDVal layer_id = 0;
	A_Ratio stretch_factor = { 1, 1 };
	PF_FpLong layer_time_offset = 0;
	bool has_frequency_keyframes = false;

	PulseSizeInfo info;
	AEFX_CLR_STRUCT(info);

	PF_ParamDef param_copy;
	AEFX_CLR_STRUCT(param_copy);

	ERR(PF_CHECKOUT_PARAM(in_data, NORMAL_CHECKBOX_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.normal_enabled = param_copy.u.bd.value;

	ERR(PF_CHECKOUT_PARAM(in_data, FREQUENCY_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.frequency = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, SHRINK_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.shrink = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, GROW_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.grow = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, PHASE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.phase = param_copy.u.fs_d.value;

	ERR(PF_CHECKOUT_PARAM(in_data, WAVE_TYPE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.wave_type = param_copy.u.pd.value - 1;

	ERR(PF_CHECKOUT_PARAM(in_data, X_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.x_tiles = param_copy.u.bd.value;

	ERR(PF_CHECKOUT_PARAM(in_data, Y_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.y_tiles = param_copy.u.bd.value;

	ERR(PF_CHECKOUT_PARAM(in_data, MIRROR_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.mirror = param_copy.u.bd.value;

	ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_CHECKBOX_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
	if (!err) info.compatibility_enabled = param_copy.u.bd.value;

	if (info.compatibility_enabled) {
		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_FREQUENCY_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) info.compat_frequency = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_SHRINK_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) info.compat_shrink = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_GROW_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) info.compat_grow = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_PHASE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) info.compat_phase = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_WAVE_TYPE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) info.compat_wave_type = param_copy.u.pd.value - 1;
	}

	has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

	ThreadRenderData* renderData = reinterpret_cast<ThreadRenderData*>(malloc(sizeof(ThreadRenderData)));
	AEFX_CLR_STRUCT(*renderData);

	if (renderData) {
		renderData->info = info;

		PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;
		PF_FpLong duration = current_time;

		AEGP_SuiteHandler suites(in_data->pica_basicP);

		if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
			AEGP_LayerH layer = NULL;
			err = suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_data->effect_ref, &layer);

			if (!err && layer) {
				err = suites.LayerSuite7()->AEGP_GetLayerID(layer, &layer_id);

				A_Time in_point;
				err = suites.LayerSuite7()->AEGP_GetLayerInPoint(layer, AEGP_LTimeMode_LayerTime, &in_point);

				if (!err) {
					layer_time_offset = (PF_FpLong)in_point.value / (PF_FpLong)in_point.scale;
					renderData->layer_start_seconds = layer_time_offset;      
				}

				err = suites.LayerSuite7()->AEGP_GetLayerStretch(layer, &stretch_factor);
			}
		}

		PF_FpLong stretch_ratio = (PF_FpLong)stretch_factor.num / (PF_FpLong)stretch_factor.den;

		current_time -= layer_time_offset;
		duration -= layer_time_offset;

		current_time *= stretch_ratio;
		duration *= stretch_ratio;

		renderData->current_time = current_time;
		renderData->duration = duration;
		renderData->accumulated_phase = 0.0;
		renderData->accumulated_phase_initialized = false;       

		if (has_frequency_keyframes && info.frequency > 0 && info.normal_enabled) {
			PF_FpLong value_out;
			err = valueAtTimeHz(in_data, FREQUENCY_SLIDER, current_time, duration, renderData, &value_out);
			renderData->accumulated_phase_initialized = true;       
		}

		extra->output->pre_render_data = renderData;
		extra->output->delete_pre_render_data_func = DisposePrerenderData;
	}
	else {
		err = PF_Err_OUT_OF_MEMORY;
	}

	PF_RenderRequest req = extra->input->output_request;
	req.preserve_rgb_of_zero_alpha = TRUE;

	PF_CheckoutResult checkout;
	ERR(extra->cb->checkout_layer(in_data->effect_ref,
		PULSESIZE_INPUT,
		PULSESIZE_INPUT,
		&req,
		in_data->current_time,
		in_data->time_step,
		in_data->time_scale,
		&checkout));

	if (!err) {
		struct {
			A_u_char has_frequency_keyframes;
			A_long time_offset;
			AEGP_LayerIDVal layer_id;
			A_Ratio stretch_factor;
			PulseSizeInfo info;
		} detection_data;

		detection_data.has_frequency_keyframes = has_frequency_keyframes ? 1 : 0;
		detection_data.time_offset = 0;
		detection_data.layer_id = layer_id;
		detection_data.stretch_factor = stretch_factor;
		detection_data.info = info;

		ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(detection_data), &detection_data));

		extra->output->max_result_rect = checkout.max_result_rect;
		extra->output->result_rect = checkout.result_rect;
		extra->output->solid = FALSE;
		extra->output->flags = PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
	}

	return err;
}


template <typename PixelType>
static PixelType
SampleBilinear(PixelType* src, PF_FpLong x, PF_FpLong y, A_long width, A_long height, A_long rowbytes,
	bool x_tiles, bool y_tiles, bool mirror) {

	bool outsideBounds = false;

	if (x_tiles) {
		if (mirror) {
			float intPart;
			float fracPart = modff(fabsf(x / width), &intPart);
			int isOdd = (int)intPart & 1;
			x = isOdd ? (1.0f - fracPart) * width : fracPart * width;
		}
		else {
			x = fmod(fmod(x, width) + width, width);
		}
	}
	else {
		if (x < 0 || x >= width) {
			outsideBounds = true;
		}
	}

	if (y_tiles) {
		if (mirror) {
			float intPart;
			float fracPart = modff(fabsf(y / height), &intPart);
			int isOdd = (int)intPart & 1;
			y = isOdd ? (1.0f - fracPart) * height : fracPart * height;
		}
		else {
			y = fmod(fmod(y, height) + height, height);
		}
	}
	else {
		if (y < 0 || y >= height) {
			outsideBounds = true;
		}
	}

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

	x = MAX(0, MIN(width - 1.001f, x));
	y = MAX(0, MIN(height - 1.001f, y));

	A_long x0 = static_cast<A_long>(x);
	A_long y0 = static_cast<A_long>(y);
	A_long x1 = MIN(x0 + 1, width - 1);
	A_long y1 = MIN(y0 + 1, height - 1);

	PF_FpLong fx = x - x0;
	PF_FpLong fy = y - y0;

	PixelType* p00 = (PixelType*)((char*)src + y0 * rowbytes) + x0;
	PixelType* p01 = (PixelType*)((char*)src + y0 * rowbytes) + x1;
	PixelType* p10 = (PixelType*)((char*)src + y1 * rowbytes) + x0;
	PixelType* p11 = (PixelType*)((char*)src + y1 * rowbytes) + x1;

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
	PulseSizeInfo* infoP,
	PF_FpLong current_time,
	PF_FpLong accumulated_phase)
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

	bool normal_enabled = infoP->normal_enabled;
	bool compatibility_enabled = infoP->compatibility_enabled;

	typedef struct {
		int srcPitch;
		int dstPitch;
		int is16f;
		int width;
		int height;
		float frequency;
		float shrink;
		float grow;
		int waveType;
		float phase;
		float currentTime;
		int xTiles;
		int yTiles;
		int mirror;
		int normalEnabled;
		int compatibilityEnabled;
		float compatFrequency;
		float compatShrink;
		float compatGrow;
		float compatPhase;
		int compatWaveType;
		float accumulatedPhase;
		int hasFrequencyKeyframes;
	} PulseSizeParams;

	PulseSizeParams params;

	params.width = input_worldP->width;
	params.height = input_worldP->height;
	params.is16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
	params.srcPitch = input_worldP->rowbytes / bytes_per_pixel;
	params.dstPitch = output_worldP->rowbytes / bytes_per_pixel;
	params.frequency = infoP->frequency;
	params.shrink = infoP->shrink;
	params.grow = infoP->grow;
	params.waveType = infoP->wave_type;
	params.phase = infoP->phase;
	params.currentTime = current_time;
	params.xTiles = infoP->x_tiles;
	params.yTiles = infoP->y_tiles;
	params.mirror = infoP->mirror;
	params.normalEnabled = normal_enabled;
	params.compatibilityEnabled = compatibility_enabled;
	params.compatFrequency = infoP->compat_frequency;
	params.compatShrink = infoP->compat_shrink;
	params.compatGrow = infoP->compat_grow;
	params.compatPhase = infoP->compat_phase;
	params.compatWaveType = infoP->compat_wave_type;
	params.accumulatedPhase = accumulated_phase;
	params.hasFrequencyKeyframes = HasAnyFrequencyKeyframes(in_dataP);

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.srcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.dstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.is16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.width));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.height));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.frequency));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.shrink));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.grow));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.waveType));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.phase));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.currentTime));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.xTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.yTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.mirror));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.normalEnabled));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.compatibilityEnabled));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.compatFrequency));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.compatShrink));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.compatGrow));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.compatPhase));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.compatWaveType));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(float), &params.accumulatedPhase));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pulsesize_kernel, param_index++, sizeof(int), &params.hasFrequencyKeyframes));

		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = {
			((params.width + threadBlock[0] - 1) / threadBlock[0]) * threadBlock[0],
			((params.height + threadBlock[1] - 1) / threadBlock[1]) * threadBlock[1]
		};

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->pulsesize_kernel,
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
		PulseSize_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			params.srcPitch,
			params.dstPitch,
			params.is16f,
			params.width,
			params.height,
			params.frequency,
			params.shrink,
			params.grow,
			params.waveType,
			params.phase,
			params.currentTime,
			params.xTiles,
			params.yTiles,
			params.mirror,
			params.normalEnabled,
			params.compatibilityEnabled,
			params.compatFrequency,
			params.compatShrink,
			params.compatGrow,
			params.compatPhase,
			params.compatWaveType,
			params.accumulatedPhase,
			params.hasFrequencyKeyframes);

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

		DXShaderExecution shaderExecution(
			dx_gpu_data->mContext,
			dx_gpu_data->mPulseSizeShader,
			3);

		DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(PulseSizeParams)));
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

		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &params
			length : sizeof(PulseSizeParams)
			options : MTLResourceStorageModeManaged]autorelease];

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->pulsesize_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = {
			((params.width + threadsPerGroup.width - 1) / threadsPerGroup.width),
			((params.height + threadsPerGroup.height - 1) / threadsPerGroup.height),
			1
		};

		[computeEncoder setComputePipelineState : metal_dataP->pulsesize_pipeline] ;
		[computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
		[computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
		[computeEncoder setBuffer : param_buffer offset : 0 atIndex : 2] ;
		[computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;
		[computeEncoder endEncoding] ;
		[commandBuffer commit] ;

		err = NSError2PFErr([commandBuffer error]);
	}
#endif  

	return err;
}


static PF_Err
SmartRenderCPU(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	ThreadRenderData* renderData)
{
	PF_Err err = PF_Err_NONE;
	PulseSizeInfo* infoP = &renderData->info;
	PF_FpLong current_time = renderData->current_time;

	if ((infoP->normal_enabled && infoP->compatibility_enabled) ||
		(!infoP->normal_enabled && !infoP->compatibility_enabled)) {

		for (A_long y = 0; y < output_worldP->height; y++) {
			void* srcRow = (void*)((char*)input_worldP->data + y * input_worldP->rowbytes);
			void* dstRow = (void*)((char*)output_worldP->data + y * output_worldP->rowbytes);
			memcpy(dstRow, srcRow, output_worldP->width * (pixel_format == PF_PixelFormat_ARGB128 ? 16 : (pixel_format == PF_PixelFormat_ARGB64 ? 8 : 4)));
		}
		return err;
	}

	PF_FpLong m;
	PF_FpLong ds;

	if (infoP->normal_enabled) {
		bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

		if (has_frequency_keyframes && renderData->accumulated_phase > 0.0) {
			PF_FpLong effectivePhase = infoP->phase + renderData->accumulated_phase;

			if (infoP->wave_type == 0) {  
				m = sin(effectivePhase * M_PI);
			}
			else {  
				m = TriangleWave(effectivePhase / 2.0);
			}
		}
		else {
			PF_FpLong effectivePhase = infoP->phase + (current_time * infoP->frequency);

			if (infoP->wave_type == 0) {  
				m = sin(effectivePhase * M_PI);
			}
			else {  
				m = TriangleWave(effectivePhase / 2.0);
			}
		}

		PF_FpLong range = infoP->grow - infoP->shrink;
		ds = (range * ((m + 1.0) / 2.0)) + infoP->shrink;
	}
	else {
		if (infoP->compat_wave_type == 0) {  
			m = sin(((current_time * infoP->compat_frequency) + infoP->compat_phase) * M_PI);
		}
		else {  
			m = TriangleWave(((current_time * infoP->compat_frequency) + infoP->compat_phase) / 2.0);
		}

		PF_FpLong range = infoP->compat_grow - infoP->compat_shrink;
		ds = (range * ((m + 1.0) / 2.0)) + infoP->compat_shrink;
	}

	A_long width = output_worldP->width;
	A_long height = output_worldP->height;
	A_long input_width = input_worldP->width;
	A_long input_height = input_worldP->height;
	A_long rowbytes = input_worldP->rowbytes;

	PF_FpLong centerX = (width / 2.0);
	PF_FpLong centerY = (height / 2.0);
	PF_FpLong inputCenterX = (input_width / 2.0);
	PF_FpLong inputCenterY = (input_height / 2.0);

	PF_FpLong scaleFactor = 1.0 / ds;

	switch (pixel_format) {
	case PF_PixelFormat_ARGB128: {
		for (A_long y = 0; y < height; y++) {
			PF_PixelFloat* outP = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactor + inputCenterX;
				PF_FpLong srcY = (y - centerY) * scaleFactor + inputCenterY;

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
		for (A_long y = 0; y < height; y++) {
			PF_Pixel16* outP = (PF_Pixel16*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactor + inputCenterX;
				PF_FpLong srcY = (y - centerY) * scaleFactor + inputCenterY;

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
		for (A_long y = 0; y < height; y++) {
			PF_Pixel8* outP = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactor + inputCenterX;
				PF_FpLong srcY = (y - centerY) * scaleFactor + inputCenterY;

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

	ThreadRenderData* renderData = reinterpret_cast<ThreadRenderData*>(extraP->input->pre_render_data);

	if (renderData) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, PULSESIZE_INPUT, &input_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		if (!err && input_worldP && output_worldP) {
			renderData->width = input_worldP->width;
			renderData->height = input_worldP->height;
			renderData->input_data = input_worldP->data;
			renderData->input_rowbytes = input_worldP->rowbytes;

			AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
				kPFWorldSuite,
				kPFWorldSuiteVersion2,
				out_data);

			PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
			ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

			if (isGPU) {
				ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP,
					&renderData->info, renderData->current_time, renderData->accumulated_phase));
			}
			else {
				ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, renderData));
			}
		}
	}
	else {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	if (input_worldP) {
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, PULSESIZE_INPUT));
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

	PulseSizeInfo info;
	AEFX_CLR_STRUCT(info);

	info.frequency = params[FREQUENCY_SLIDER]->u.fs_d.value;
	info.shrink = params[SHRINK_SLIDER]->u.fs_d.value;
	info.grow = params[GROW_SLIDER]->u.fs_d.value;
	info.phase = params[PHASE_SLIDER]->u.fs_d.value;
	info.wave_type = params[WAVE_TYPE_SLIDER]->u.pd.value - 1;

	info.x_tiles = params[X_TILES_DISK_ID]->u.bd.value;
	info.y_tiles = params[Y_TILES_DISK_ID]->u.bd.value;
	info.mirror = params[MIRROR_DISK_ID]->u.bd.value;

	bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

	PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;

	if (has_frequency_keyframes) {
		A_long time_shift = in_data->time_step / 2;

		A_Time shifted_time;
		shifted_time.value = in_data->current_time + time_shift;
		shifted_time.scale = in_data->time_scale;

		current_time = (PF_FpLong)shifted_time.value / (PF_FpLong)shifted_time.scale;
	}

	PF_FpLong m = CalculateWaveValue(info.wave_type,
		info.frequency,
		current_time,
		info.phase);

	PF_FpLong range = info.grow - info.shrink;
	PF_FpLong ds = (range * ((m + 1.0) / 2.0)) + info.shrink;

	PF_PixelFormat pixelFormat;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	PF_WorldSuite2* wsP = NULL;
	ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
	if (!err) {
		ERR(wsP->PF_GetPixelFormat(output, &pixelFormat));
		suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
	}

	A_long width = output->width;
	A_long height = output->height;
	A_long input_width = params[PULSESIZE_INPUT]->u.ld.width;
	A_long input_height = params[PULSESIZE_INPUT]->u.ld.height;
	A_long rowbytes = params[PULSESIZE_INPUT]->u.ld.rowbytes;

	PF_FpLong centerX = (width / 2.0);
	PF_FpLong centerY = (height / 2.0);
	PF_FpLong inputCenterX = (input_width / 2.0);
	PF_FpLong inputCenterY = (input_height / 2.0);

	PF_FpLong scaleFactor = 1.0 / ds;

	switch (pixelFormat) {
	case PF_PixelFormat_ARGB128: {
		for (A_long y = 0; y < height; y++) {
			PF_PixelFloat* outP = (PF_PixelFloat*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactor + inputCenterX;
				PF_FpLong srcY = (y - centerY) * scaleFactor + inputCenterY;

				*outP = SampleBilinear<PF_PixelFloat>(
					(PF_PixelFloat*)params[PULSESIZE_INPUT]->u.ld.data,
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
		for (A_long y = 0; y < height; y++) {
			PF_Pixel16* outP = (PF_Pixel16*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactor + inputCenterX;
				PF_FpLong srcY = (y - centerY) * scaleFactor + inputCenterY;

				*outP = SampleBilinear<PF_Pixel16>(
					(PF_Pixel16*)params[PULSESIZE_INPUT]->u.ld.data,
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
		for (A_long y = 0; y < height; y++) {
			PF_Pixel8* outP = (PF_Pixel8*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactor + inputCenterX;
				PF_FpLong srcY = (y - centerY) * scaleFactor + inputCenterY;

				*outP = SampleBilinear<PF_Pixel8>(
					(PF_Pixel8*)params[PULSESIZE_INPUT]->u.ld.data,
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

	result = PF_REGISTER_EFFECT(
		inPtr,
		inPluginDataCallBackPtr,
		"Pulse Size",            
		"DKT Pulse Size",              
		"DKT Effects",          
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
			err = Render(in_data, out_data, params, output);
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