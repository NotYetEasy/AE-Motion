#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "Oscillate.h"
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
	int mirror,
	float downsample_x,
	float downsample_y,
	int compatibilityEnabled,
	float compatAngle,
	float compatFrequency,
	float compatMagnitude,
	int compatWaveType,
	int normalEnabled,
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

	out_data->my_version = PF_VERSION(MAJOR_VERSION,
		MINOR_VERSION,
		BUG_VERSION,
		STAGE_VERSION,
		BUILD_VERSION);

	out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;

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

	def.flags = PF_ParamFlag_SUPERVISE;      
	PF_ADD_POPUP("Direction",
		3,                           
		1,                           
		"Angle|Depth|Orbit",       
		0,                                
		DIRECTION_SLIDER);

	AEFX_CLR_STRUCT(def);

	PF_ADD_ANGLE("Angle",
		45,
		ANGLE_SLIDER);

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

	PF_ADD_POPUP("Wave",
		2,                        
		1,                        
		"Sine|Triangle",        
		0,                      
		WAVE_TYPE_SLIDER);

	AEFX_CLR_STRUCT(def);

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
	PF_ADD_FLOAT_SLIDERX("Angle",
		0,
		180,
		0,
		180,
		0,
		PF_Precision_INTEGER,
		0,
		0,
		COMPATIBILITY_ANGLE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Frequency",
		0.1,
		16.0,
		0.1,
		16.0,
		0.1,
		PF_Precision_TENTHS,
		0,
		0,
		COMPATIBILITY_FREQUENCY_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Magnitude",
		1.00,
		499.00,
		1.00,
		499.00,
		1.00,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_MAGNITUDE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_POPUP("Wave Type",
		2,                        
		1,                        
		"Sine|Triangle",        
		0,                      
		COMPATIBILITY_WAVE_TYPE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_END;
	PF_ADD_PARAM(in_data, -1, &def);

	out_data->num_params = RANDOMMOVE_NUM_PARAMS;

	return err;
}


static PF_Err
UserChangedParam(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output,
	PF_UserChangedParamExtra* extra)
{
	PF_Err err = PF_Err_NONE;

	if (extra->param_index == DIRECTION_SLIDER) {
		A_long direction = params[DIRECTION_SLIDER]->u.pd.value;

		PF_ParamDef angle_param_copy;
		AEFX_CLR_STRUCT(angle_param_copy);

		angle_param_copy = *(params[ANGLE_SLIDER]);

		if (direction == 2) {     
			angle_param_copy.ui_flags |= PF_PUI_DISABLED;
		}
		else {
			angle_param_copy.ui_flags &= ~PF_PUI_DISABLED;
		}

		AEGP_SuiteHandler suites(in_data->pica_basicP);

		PF_ParamUtilsSuite3* paramUtils = suites.ParamUtilsSuite3();
		if (paramUtils) {
			err = paramUtils->PF_UpdateParamUI(
				in_data->effect_ref,
				ANGLE_SLIDER,
				&angle_param_copy);
		}
	}

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

		dx_gpu_data->mContext = std::make_shared<DXContext>();
		dx_gpu_data->mOscillateShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"OscillateKernel", csoPath, sigPath));

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

		NSString* source = [NSString stringWithCString : kOscillateKernel_MetalString encoding : NSUTF8StringEncoding];
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
	OscillateRenderData* renderData,
	PF_FpLong* value_out)
{
	PF_Err err = PF_Err_NONE;

	err = valueAtTime(in_data, stream_index, time_secs, value_out);
	if (err) return err;

	if (stream_index == FREQUENCY_SLIDER) {
		bool isKeyed = HasAnyFrequencyKeyframes(in_data);

		bool isHz = true;              

		if (isHz && isKeyed) {
			float fps = 120.0f;
			int totalSteps = (int)roundf(duration * fps);
			int curSteps = (int)roundf(fps * time_secs);

			if (!renderData->accumulated_phase_initialized) {
				renderData->accumulated_phase = 0.0f;
				renderData->accumulated_phase_initialized = true;
			}

			if (curSteps >= 0) {
				renderData->accumulated_phase = 0.0f;
				for (int i = 0; i <= curSteps; i++) {
					PF_FpLong stepValue;
					err = valueAtTime(in_data, stream_index, i / fps, &stepValue);
					if (err) return err;

					renderData->accumulated_phase += stepValue / fps;
				}

				*value_out = renderData->accumulated_phase;
			}
		}
	}

	return err;
}

static PF_FpLong TriangleWave(PF_FpLong t)
{
	t = fmod(t + 0.75, 1.0);

	if (t < 0)
		t += 1.0;

	return (fabs(t - 0.5) - 0.25) * 4.0;
}

static PF_FpLong CalculateWaveValue(int wave_type, PF_FpLong frequency, PF_FpLong current_time, PF_FpLong phase, PF_FpLong accumulated_phase = 0.0)
{
	PF_FpLong X, m;

	if (accumulated_phase > 0.0) {
		if (wave_type == 0) {  
			X = ((accumulated_phase * 2.0 + phase * 2.0) * 3.14159);
			m = sin(X);
		}
		else {  
			X = ((accumulated_phase * 2.0) + (phase * 2.0)) / 2.0 + phase;
			m = TriangleWave(X);
		}
	}
	else {
		if (wave_type == 0) {  
			X = (frequency * 2.0 * current_time) + (phase * 2.0);
			m = sin(X * M_PI);
		}
		else {  
			X = ((frequency * 2.0 * current_time) + (phase * 2.0)) / 2.0 + phase;
			m = TriangleWave(X);
		}
	}

	return m;
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

static void
DisposePreRenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		OscillateRenderData* renderData = reinterpret_cast<OscillateRenderData*>(pre_render_dataPV);
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
	PF_CheckoutResult in_result;
	PF_RenderRequest req = extra->input->output_request;

	extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

	OscillateRenderData* renderData = reinterpret_cast<OscillateRenderData*>(malloc(sizeof(OscillateRenderData)));
	AEFX_CLR_STRUCT(*renderData);

	if (renderData) {
		PF_ParamDef param_copy;
		AEFX_CLR_STRUCT(param_copy);

		ERR(PF_CHECKOUT_PARAM(in_data, NORMAL_CHECKBOX_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.normal = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_CHECKBOX_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.compatibility = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, DIRECTION_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.direction = param_copy.u.pd.value - 1;

		ERR(PF_CHECKOUT_PARAM(in_data, ANGLE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) {
			renderData->info.angle = (PF_FpLong)(param_copy.u.ad.value) / 65536.0;
		}

		bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);
		renderData->has_frequency_keyframes = has_frequency_keyframes;

		ERR(PF_CHECKOUT_PARAM(in_data, FREQUENCY_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) {
			renderData->info.frequency = param_copy.u.fs_d.value;
		}

		ERR(PF_CHECKOUT_PARAM(in_data, MAGNITUDE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.magnitude = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, WAVE_TYPE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.wave_type = param_copy.u.pd.value - 1;

		ERR(PF_CHECKOUT_PARAM(in_data, PHASE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.phase = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, X_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.x_tiles = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, Y_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.y_tiles = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, MIRROR_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.mirror = param_copy.u.bd.value;

		if (renderData->info.compatibility) {
			ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_ANGLE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) renderData->info.compat_angle = param_copy.u.fs_d.value;

			ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_FREQUENCY_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) renderData->info.compat_frequency = param_copy.u.fs_d.value;

			ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_MAGNITUDE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) renderData->info.compat_magnitude = param_copy.u.fs_d.value;

			ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_WAVE_TYPE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
			if (!err) renderData->info.compat_wave_type = param_copy.u.pd.value - 1;
		}

		AEGP_LayerIDVal layer_id = 0;
		AEGP_SuiteHandler suites(in_data->pica_basicP);

		PF_FpLong layer_time_offset = 0;
		A_Ratio stretch_factor = { 1, 1 };

		if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
			AEGP_LayerH layer = NULL;

			err = suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_data->effect_ref, &layer);

			if (!err && layer) {
				err = suites.LayerSuite9()->AEGP_GetLayerID(layer, &layer_id);

				A_Time in_point;
				err = suites.LayerSuite9()->AEGP_GetLayerInPoint(layer, AEGP_LTimeMode_LayerTime, &in_point);

				if (!err) {
					layer_time_offset = (PF_FpLong)in_point.value / (PF_FpLong)in_point.scale;
					renderData->layer_start_seconds = layer_time_offset;
				}

				err = suites.LayerSuite9()->AEGP_GetLayerStretch(layer, &stretch_factor);
			}
		}

		A_long time_offset = 0;

		PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;
		PF_FpLong prev_time = current_time - ((PF_FpLong)in_data->time_step / (PF_FpLong)in_data->time_scale);
		PF_FpLong duration = current_time;

		PF_FpLong stretch_ratio = (PF_FpLong)stretch_factor.num / (PF_FpLong)stretch_factor.den;

		current_time -= layer_time_offset;
		prev_time -= layer_time_offset;
		duration -= layer_time_offset;

		current_time *= stretch_ratio;
		prev_time *= stretch_ratio;
		duration *= stretch_ratio;

		renderData->prev_time = prev_time;
		renderData->current_time = current_time;
		renderData->duration = duration;

		renderData->accumulated_phase = 0.0f;
		renderData->accumulated_phase_initialized = false;

		if (has_frequency_keyframes && renderData->info.frequency > 0) {
			PF_FpLong accumulated_phase;
			err = valueAtTimeHz(in_data, FREQUENCY_SLIDER, current_time, duration, renderData, &accumulated_phase);
			if (!err) {
				renderData->accumulated_phase = accumulated_phase;
				renderData->accumulated_phase_initialized = true;
			}
		}

		extra->output->pre_render_data = renderData;
		extra->output->delete_pre_render_data_func = DisposePreRenderData;

		ERR(extra->cb->checkout_layer(in_data->effect_ref,
			RANDOMMOVE_INPUT,
			RANDOMMOVE_INPUT,
			&req,
			in_data->current_time,
			in_data->time_step,
			in_data->time_scale,
			&in_result));

		if (!err) {
			struct {
				A_u_char has_frequency_keyframes;
				A_long time_offset;
				AEGP_LayerIDVal layer_id;
				A_Ratio stretch_factor;
				RandomMoveInfo info;
			} detection_data;

			detection_data.has_frequency_keyframes = has_frequency_keyframes ? 1 : 0;
			detection_data.time_offset = time_offset;
			detection_data.layer_id = layer_id;
			detection_data.stretch_factor = stretch_factor;
			detection_data.info = renderData->info;

			ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(detection_data), &detection_data));

			extra->output->max_result_rect = in_result.max_result_rect;
			extra->output->result_rect = in_result.result_rect;
			extra->output->solid = FALSE;
		}
	}
	else {
		err = PF_Err_OUT_OF_MEMORY;
	}

	return err;
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

	PF_RationalScale downsample_x = in_dataP->downsample_x;
	PF_RationalScale downsample_y = in_dataP->downsample_y;

	PF_FpLong downsample_factor_x = (PF_FpLong)downsample_x.num / (PF_FpLong)downsample_x.den;
	PF_FpLong downsample_factor_y = (PF_FpLong)downsample_y.num / (PF_FpLong)downsample_y.den;

	bool normal_enabled = infoP->normal;
	bool compatibility_enabled = infoP->compatibility;

	if ((!normal_enabled && !compatibility_enabled) || (normal_enabled && compatibility_enabled)) {
		infoP->magnitude = 0;
		infoP->frequency = 0;
	}

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
		float downsample_x;
		float downsample_y;
		int compatibilityEnabled;
		float compatAngle;
		float compatFrequency;
		float compatMagnitude;
		int compatWaveType;
		int normalEnabled;
		float accumulatedPhase;
		int hasFrequencyKeyframes;
	} OscillateParams;

	OscillateParams params;

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
	params.downsample_x = downsample_factor_x;
	params.downsample_y = downsample_factor_y;

	params.normalEnabled = normal_enabled;
	params.compatibilityEnabled = compatibility_enabled;
	params.compatAngle = infoP->compat_angle;
	params.compatFrequency = infoP->compat_frequency;
	params.compatMagnitude = infoP->compat_magnitude;
	params.compatWaveType = infoP->compat_wave_type;

	OscillateRenderData* renderData = reinterpret_cast<OscillateRenderData*>(extraP->input->pre_render_data);

	params.accumulatedPhase = 0.0f;
	params.hasFrequencyKeyframes = 0;

	if (renderData) {
		params.accumulatedPhase = renderData->accumulated_phase;
		params.hasFrequencyKeyframes = renderData->has_frequency_keyframes &&
			renderData->accumulated_phase_initialized ? 1 : 0;
	}

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

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
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.downsample_x));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.downsample_y));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.compatibilityEnabled));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.compatAngle));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.compatFrequency));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.compatMagnitude));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.compatWaveType));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.normalEnabled));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(float), &params.accumulatedPhase));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->oscillate_kernel, param_index++, sizeof(int), &params.hasFrequencyKeyframes));

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
			params.mirror,
			params.downsample_x,
			params.downsample_y,
			params.compatibilityEnabled,
			params.compatAngle,
			params.compatFrequency,
			params.compatMagnitude,
			params.compatWaveType,
			params.normalEnabled,
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
			dx_gpu_data->mOscillateShader,
			3);

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

		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &params
			length : sizeof(OscillateParams)
			options : MTLResourceStorageModeManaged]autorelease];

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
	RandomMoveInfo* infoP,
	PF_FpLong current_time,
	OscillateRenderData* renderData)
{
	PF_Err err = PF_Err_NONE;

	PF_RationalScale downsample_x = in_data->downsample_x;
	PF_RationalScale downsample_y = in_data->downsample_y;

	PF_FpLong downsample_factor_x = (PF_FpLong)downsample_x.num / (PF_FpLong)downsample_x.den;
	PF_FpLong downsample_factor_y = (PF_FpLong)downsample_y.num / (PF_FpLong)downsample_y.den;

	bool normal_enabled = infoP->normal;
	bool compatibility_enabled = infoP->compatibility;

	if ((!normal_enabled && !compatibility_enabled) || (normal_enabled && compatibility_enabled)) {
		A_long height = output_worldP->height;
		A_long width = output_worldP->width;

		switch (pixel_format) {
		case PF_PixelFormat_ARGB128: {
			for (A_long y = 0; y < height; y++) {
				PF_PixelFloat* srcP = (PF_PixelFloat*)((char*)input_worldP->data + y * input_worldP->rowbytes);
				PF_PixelFloat* dstP = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);
				memcpy(dstP, srcP, width * sizeof(PF_PixelFloat));
			}
			break;
		}
		case PF_PixelFormat_ARGB64: {
			for (A_long y = 0; y < height; y++) {
				PF_Pixel16* srcP = (PF_Pixel16*)((char*)input_worldP->data + y * input_worldP->rowbytes);
				PF_Pixel16* dstP = (PF_Pixel16*)((char*)output_worldP->data + y * output_worldP->rowbytes);
				memcpy(dstP, srcP, width * sizeof(PF_Pixel16));
			}
			break;
		}
		case PF_PixelFormat_ARGB32:
		default: {
			for (A_long y = 0; y < height; y++) {
				PF_Pixel8* srcP = (PF_Pixel8*)((char*)input_worldP->data + y * input_worldP->rowbytes);
				PF_Pixel8* dstP = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);
				memcpy(dstP, srcP, width * sizeof(PF_Pixel8));
			}
			break;
		}
		}
		return err;
	}

	PF_FpLong offsetX = 0, offsetY = 0;
	PF_FpLong scale = 100.0;

	PF_FpLong angleRad;
	PF_FpLong dx, dy;
	PF_FpLong m;

	if (compatibility_enabled) {
		angleRad = infoP->compat_angle * M_PI / 180.0;
		dx = sin(angleRad);
		dy = cos(angleRad);

		PF_FpLong duration = 1.0;
		PF_FpLong t = current_time;

		if (infoP->compat_wave_type == 0) {
			m = sin(t * duration * infoP->compat_frequency * 3.14159);
		}
		else {
			PF_FpLong wavePhase = t * duration * infoP->compat_frequency / 2.0;
			m = TriangleWave(wavePhase);
		}

		offsetX = dx * infoP->compat_magnitude * m * downsample_factor_x;
		offsetY = dy * infoP->compat_magnitude * m * downsample_factor_y;
	}
	else if (normal_enabled) {
		angleRad = infoP->angle * M_PI / 180.0;
		dx = cos(angleRad);
		dy = sin(angleRad);

		if (renderData && renderData->has_frequency_keyframes && renderData->accumulated_phase_initialized) {
			m = CalculateWaveValue(infoP->wave_type,
				infoP->frequency,
				current_time,
				infoP->phase,
				renderData->accumulated_phase);
		}
		else {
			m = CalculateWaveValue(infoP->wave_type,
				infoP->frequency,
				current_time,
				infoP->phase);
		}

		switch (infoP->direction) {
		case 0:
			offsetX = dx * (infoP->magnitude * downsample_factor_x) * m;
			offsetY = dy * (infoP->magnitude * downsample_factor_y) * m;
			break;

		case 1:
			scale = 100.0 - (infoP->magnitude * m * 0.1);
			break;

		case 2: {
			offsetX = dx * (infoP->magnitude * downsample_factor_x) * m;
			offsetY = dy * (infoP->magnitude * downsample_factor_y) * m;

			PF_FpLong phaseShift = infoP->wave_type == 0 ? 0.25 : 0.125;

			if (renderData && renderData->has_frequency_keyframes && renderData->accumulated_phase_initialized) {
				m = CalculateWaveValue(infoP->wave_type,
					infoP->frequency,
					current_time,
					infoP->phase,
					renderData->accumulated_phase);
			}
			else {
				m = CalculateWaveValue(infoP->wave_type,
					infoP->frequency,
					current_time,
					infoP->phase + phaseShift);
			}

			scale = 100.0 - (infoP->magnitude * m * 0.1);
			break;
		}
		}
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

	PF_FpLong scaleFactorX = 100.0 / scale;
	PF_FpLong scaleFactorY = 100.0 / scale;


	switch (pixel_format) {
	case PF_PixelFormat_ARGB128: {
		for (A_long y = 0; y < height; y++) {
			PF_PixelFloat* outP = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

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
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

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
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

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
	PF_Err err = PF_Err_NONE,
		err2 = PF_Err_NONE;

	PF_EffectWorld* input_worldP = NULL,
		* output_worldP = NULL;

	OscillateRenderData* renderData = reinterpret_cast<OscillateRenderData*>(extraP->input->pre_render_data);

	if (renderData) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, RANDOMMOVE_INPUT, &input_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
			kPFWorldSuite,
			kPFWorldSuiteVersion2,
			out_data);
		PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		if (isGPU) {
			ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, &renderData->info, renderData->current_time));
		}
		else {
			ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, &renderData->info, renderData->current_time, renderData));
		}
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, RANDOMMOVE_INPUT));
	}
	else {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
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

	RandomMoveInfo info;
	AEFX_CLR_STRUCT(info);

	info.direction = params[DIRECTION_SLIDER]->u.pd.value - 1;
	info.angle = params[ANGLE_SLIDER]->u.fs_d.value;
	info.frequency = params[FREQUENCY_SLIDER]->u.fs_d.value;
	info.magnitude = params[MAGNITUDE_SLIDER]->u.fs_d.value;
	info.wave_type = params[WAVE_TYPE_SLIDER]->u.pd.value - 1;
	info.phase = params[PHASE_SLIDER]->u.fs_d.value;

	info.x_tiles = params[X_TILES_DISK_ID]->u.bd.value;
	info.y_tiles = params[Y_TILES_DISK_ID]->u.bd.value;
	info.mirror = params[MIRROR_DISK_ID]->u.bd.value;

	bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

	AEGP_LayerIDVal layer_id = 0;
	AEGP_SuiteHandler suites(in_data->pica_basicP);
	AEGP_LayerH layer = NULL;

	if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
		err = suites.PFInterfaceSuite1()->AEGP_GetEffectLayer(in_data->effect_ref, &layer);

		if (!err && layer) {
			err = suites.LayerSuite9()->AEGP_GetLayerID(layer, &layer_id);
		}
	}

	PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;

	PF_FpLong layer_time_offset = 0;
	if (layer_id != 0 && layer != NULL) {
		A_Time in_point;
		err = suites.LayerSuite9()->AEGP_GetLayerInPoint(layer, AEGP_LTimeMode_LayerTime, &in_point);

		if (!err) {
			layer_time_offset = (PF_FpLong)in_point.value / (PF_FpLong)in_point.scale;
		}

		current_time -= layer_time_offset;
	}

	if (has_frequency_keyframes) {
		A_long time_shift = in_data->time_step / 2;

		A_Time shifted_time;
		shifted_time.value = in_data->current_time + time_shift;
		shifted_time.scale = in_data->time_scale;

		current_time = (PF_FpLong)shifted_time.value / (PF_FpLong)shifted_time.scale;

		if (layer_id != 0) {
			current_time -= layer_time_offset;
		}
	}

	PF_FpLong angleRad = info.angle * M_PI / 180.0;
	PF_FpLong dx = cos(angleRad);
	PF_FpLong dy = sin(angleRad);

	PF_FpLong m = CalculateWaveValue(info.wave_type,
		info.frequency,
		current_time,
		info.phase);

	PF_FpLong offsetX = 0, offsetY = 0;
	PF_FpLong scale = 100.0;

	switch (info.direction) {
	case 0:       
		offsetX = dx * info.magnitude * m;
		offsetY = dy * info.magnitude * m;
		break;

	case 1:      
		scale = 100.0 - (info.magnitude * m * 0.1);
		break;

	case 2: {        
		offsetX = dx * info.magnitude * m;
		offsetY = dy * info.magnitude * m;

		PF_FpLong phaseShift = info.wave_type == 0 ? 0.25 : 0.125;
		m = CalculateWaveValue(info.wave_type,
			info.frequency,
			current_time,
			info.phase + phaseShift);

		scale = 100.0 - (info.magnitude * m * 0.1);
		break;
	}
	}

	PF_PixelFormat pixelFormat;
	PF_WorldSuite2* wsP = NULL;
	ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
	if (!err) {
		ERR(wsP->PF_GetPixelFormat(output, &pixelFormat));
		suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
	}

	A_long width = output->width;
	A_long height = output->height;
	A_long input_width = params[RANDOMMOVE_INPUT]->u.ld.width;
	A_long input_height = params[RANDOMMOVE_INPUT]->u.ld.height;
	A_long rowbytes = params[RANDOMMOVE_INPUT]->u.ld.rowbytes;

	PF_FpLong centerX = (width / 2.0);
	PF_FpLong centerY = (height / 2.0);
	PF_FpLong inputCenterX = (input_width / 2.0);
	PF_FpLong inputCenterY = (input_height / 2.0);

	PF_FpLong scaleFactorX = 100.0 / scale;
	PF_FpLong scaleFactorY = 100.0 / scale;

	switch (pixelFormat) {
	case PF_PixelFormat_ARGB128: {
		for (A_long y = 0; y < height; y++) {
			PF_PixelFloat* outP = (PF_PixelFloat*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

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
		for (A_long y = 0; y < height; y++) {
			PF_Pixel16* outP = (PF_Pixel16*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

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
		for (A_long y = 0; y < height; y++) {
			PF_Pixel8* outP = (PF_Pixel8*)((char*)output->data + y * output->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = (x - centerX) * scaleFactorX + inputCenterX - offsetX;
				PF_FpLong srcY = (y - centerY) * scaleFactorY + inputCenterY - offsetY;

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

	result = PF_REGISTER_EFFECT(
		inPtr,
		inPluginDataCallBackPtr,
		"Oscillate",            
		"DKT Oscillate",              
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
		case PF_Cmd_USER_CHANGED_PARAM:
			err = UserChangedParam(in_data, out_data, params, output, (PF_UserChangedParamExtra*)extra);
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
