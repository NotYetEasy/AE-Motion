#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "AutoShake.h"
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

extern void AutoShake_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float magnitude,
	float frequency,
	float evolution,
	float seed,
	float angle,
	float slack,
	float zshake,
	int x_tiles,
	int y_tiles,
	int mirror,
	float currentTime,
	float downsample_x,
	float downsample_y,
	int normal_mode,
	int compatibility_mode,
	float compatibility_magnitude,
	float compatibility_speed,
	float compatibility_evolution,
	float compatibility_seed,
	float compatibility_angle,
	float compatibility_slack,
	float accumulated_phase,
	int has_frequency_keyframes);


static PF_Err
About(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_SPRINTF(out_data->return_msg,
		"Auto-Shake v%d.%d\r"
		"Created by DKT with Unknown's help.\r"
		"Under development!!\r"
		"Discord: dkt0 ; unknown1234\r"
		"Contact me if you want to contribute or report bugs!",
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

	out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE |
		PF_OutFlag2_SUPPORTS_SMART_RENDER |
		PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
		PF_OutFlag2_I_MIX_GUID_DEPENDENCIES;

	if (in_dataP->appl_id == 'PrMr') {

		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_dataP,
				kPFPixelFormatSuite,
				kPFPixelFormatSuiteVersion1,
				out_data);

		(*pixelFormatSuite->ClearSupportedPixelFormats)(in_dataP->effect_ref);
		(*pixelFormatSuite->AddSupportedPixelFormat)(
			in_dataP->effect_ref,
			PrPixelFormat_VUYA_4444_32f);
	}
	else {
		out_data->out_flags2 |= PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 | PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING;
	}

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

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Normal",
		"",
		TRUE,
		0,
		NORMAL_CHECKBOX_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Magnitude",
		0,        
		2000,     
		0,         
		2000,      
		50,       
		PF_Precision_INTEGER,
		0,
		0,
		AUTOSHAKE_MAGNITUDE);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Frequency",
		0,        
		16,       
		0,         
		5,         
		2,        
		PF_Precision_HUNDREDTHS,
		0,
		0,
		AUTOSHAKE_FREQUENCY);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Evolution",
		0,        
		2000,     
		0,         
		2,         
		0,        
		PF_Precision_HUNDREDTHS,
		0,
		0,
		AUTOSHAKE_EVOLUTION);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Seed",
		0,        
		5,        
		0,         
		5,         
		0,        
		PF_Precision_HUNDREDTHS,
		0,
		0,
		AUTOSHAKE_SEED);

	AEFX_CLR_STRUCT(def);
	PF_ADD_ANGLE("Angle",
		45,
		AUTOSHAKE_ANGLE);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Slack",
		0,
		1,
		0,
		1,
		0.25,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		AUTOSHAKE_SLACK, );

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Z Shake",
		0,      
		2000,      
		0,       
		200,       
		0,     
		PF_Precision_INTEGER,
		0,
		0,
		AUTOSHAKE_ZSHAKE);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Tiles");
	def.flags = PF_ParamFlag_START_COLLAPSED;      
	PF_ADD_PARAM(in_data, TILES_GROUP_START, &def);

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
	PF_ADD_PARAM(in_data, TILES_GROUP_END, &def);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Compatibility");
	def.flags = PF_ParamFlag_START_COLLAPSED;      
	PF_ADD_PARAM(in_data, COMPATIBILITY_GROUP_START, &def);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Compatibility",
		"",
		FALSE,
		0,
		COMPATIBILITY_CHECKBOX_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Magnitude",
		0,        
		2000,     
		0,         
		2000,      
		50,       
		PF_Precision_INTEGER,
		0,
		0,
		COMPATIBILITY_MAGNITUDE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Speed",
		0.00,     
		2000.00,  
		0.00,      
		2000.00,   
		1.00,     
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_SPEED_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Evolution",
		0.00,     
		2000.00,  
		0.00,      
		2000.00,   
		0.00,     
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_EVOLUTION_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Seed",
		0.00,     
		5.00,     
		0.00,      
		5.00,      
		0.00,     
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_SEED_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_ANGLE("Angle",
		45,
		COMPATIBILITY_ANGLE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Slack",
		0.00,     
		1.00,     
		0.00,      
		1.00,      
		0.25,     
		PF_Precision_HUNDREDTHS,
		0,
		0,
		COMPATIBILITY_SLACK_DISK_ID);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_END;
	PF_ADD_PARAM(in_data, COMPATIBILITY_GROUP_END, &def);

	out_data->num_params = AUTOSHAKE_NUM_PARAMS;

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
				AUTOSHAKE_FREQUENCY,
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

	if (stream_index == AUTOSHAKE_FREQUENCY) {
		bool isKeyed = HasAnyFrequencyKeyframes(in_data);

		bool isHz = true;

		if (isHz && isKeyed) {
			float fps = 120.0f;
			int totalSteps = (int)roundf(duration * fps);
			int curSteps = (int)roundf(fps * time_secs);

			// Initialize accumulated phase if needed
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



struct OpenCLGPUData
{
	cl_kernel autoshake_kernel;
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
	ShaderObjectPtr mAutoShakeShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState>autoshake_pipeline;
};
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

		size_t sizes[] = { strlen(k16fString), strlen(kAutoShakeKernel_OpenCLString) };
		char const* strings[] = { k16fString, kAutoShakeKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->autoshake_kernel = clCreateKernel(program, "AutoShakeKernel", &result);
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
		dx_gpu_data->mAutoShakeShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"AutoShakeKernel", csoPath, sigPath));

		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mAutoShakeShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		NSString* source = [NSString stringWithCString : kAutoShake_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
			id<MTLFunction> autoshake_function = nil;
			NSString* autoshake_name = [NSString stringWithCString : "AutoShakeKernel" encoding : NSUTF8StringEncoding];

			autoshake_function = [[library newFunctionWithName : autoshake_name]autorelease];

			if (!autoshake_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->autoshake_pipeline = [device newComputePipelineStateWithFunction : autoshake_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->autoshake_kernel);

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
		dx_gpu_dataP->mAutoShakeShader.reset();

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

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
			x = fmodf(fmodf(x, width) + width, width);
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
			y = fmodf(fmodf(y, height) + height, height);
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
ProcessAutoShake(
	void* refcon,
	A_long xL,
	A_long yL,
	void* inP,
	void* outP,
	PF_PixelFormat format)
{
	PF_Err err = PF_Err_NONE;

	ThreadRenderData* renderData = reinterpret_cast<ThreadRenderData*>(refcon);
	ShakeInfo* info = &renderData->info;

	if ((info->normal_mode == FALSE && info->compatibility_mode == FALSE) ||
		(info->normal_mode == TRUE && info->compatibility_mode == TRUE)) {
		if (format == PF_PixelFormat_ARGB128) {
			*static_cast<PF_PixelFloat*>(outP) = *static_cast<PF_PixelFloat*>(inP);
		}
		else if (format == PF_PixelFormat_ARGB64) {
			*static_cast<PF_Pixel16*>(outP) = *static_cast<PF_Pixel16*>(inP);
		}
		else {
			*static_cast<PF_Pixel8*>(outP) = *static_cast<PF_Pixel8*>(inP);
		}
		return err;
	}

	PF_FpLong angleRad, s, c;
	PF_FpLong evolutionValue;
	PF_FpLong dx, dy, dz;

	if (info->normal_mode == TRUE) {
		angleRad = info->angle * (PF_PI / 180.0);
		s = sin(angleRad);
		c = cos(angleRad);

		// Use accumulated phase if available, otherwise use traditional calculation
		if (renderData->has_frequency_keyframes && renderData->accumulated_phase_initialized) {
			evolutionValue = info->evolution + renderData->accumulated_phase;
		}
		else {
			evolutionValue = info->evolution + info->frequency * renderData->current_time;
		}

		dx = SimplexNoise::noise(evolutionValue, info->seed * 49235.319798);
		dy = SimplexNoise::noise(evolutionValue + 7468.329, info->seed * 19337.940385);
		dz = SimplexNoise::noise(evolutionValue + 14192.277, info->seed * 71401.168533);

		dx *= info->magnitude;
		dy *= info->magnitude * info->slack;
		dz *= info->zshake;
	}
	else {
		angleRad = info->compatibility_angle * (PF_PI / 180.0);
		s = sin(angleRad);
		c = cos(angleRad);

		evolutionValue = info->compatibility_evolution +
			(renderData->current_time * info->compatibility_speed) -
			info->compatibility_speed;

		dx = SimplexNoise::noise(info->compatibility_seed * 54623.245, 0,
			evolutionValue + info->compatibility_seed * 49235.319798);
		dy = SimplexNoise::noise(0, info->compatibility_seed * 8723.5647,
			evolutionValue + 7468.329 + info->compatibility_seed * 19337.940385);

		dx *= info->compatibility_magnitude;
		dy *= info->compatibility_magnitude * info->compatibility_slack;
		dz = 0;
	}

	dz = -dz;

	PF_FpLong rx = dx * c - dy * s;
	PF_FpLong ry = dx * s + dy * c;

	PF_FpLong srcX = xL - rx;
	PF_FpLong srcY = yL - ry;

	if (dz != 0) {
		PF_FpLong centerX = renderData->width / 2.0;
		PF_FpLong centerY = renderData->height / 2.0;

		PF_FpLong relX = srcX - centerX;
		PF_FpLong relY = srcY - centerY;

		PF_FpLong safe_dz = MAX(-900.0, MIN(900.0, dz));

		PF_FpLong scale = 1000.0 / (1000.0 - safe_dz);

		scale = MIN(scale, 1.0);

		srcX = relX / scale + centerX;
		srcY = relY / scale + centerY;
	}

	if (format == PF_PixelFormat_ARGB128) {
		PF_PixelFloat* pixel_in = static_cast<PF_PixelFloat*>(inP);
		PF_PixelFloat* pixel_out = static_cast<PF_PixelFloat*>(outP);
		*pixel_out = SampleBilinear<PF_PixelFloat>(
			static_cast<PF_PixelFloat*>(renderData->input_data),
			srcX, srcY,
			renderData->width, renderData->height,
			renderData->input_rowbytes,
			info->x_tiles,
			info->y_tiles,
			info->mirror
		);
	}
	else if (format == PF_PixelFormat_ARGB64) {
		PF_Pixel16* pixel_in = static_cast<PF_Pixel16*>(inP);
		PF_Pixel16* pixel_out = static_cast<PF_Pixel16*>(outP);
		*pixel_out = SampleBilinear<PF_Pixel16>(
			static_cast<PF_Pixel16*>(renderData->input_data),
			srcX, srcY,
			renderData->width, renderData->height,
			renderData->input_rowbytes,
			info->x_tiles,
			info->y_tiles,
			info->mirror
		);
	}
	else {
		PF_Pixel8* pixel_in = static_cast<PF_Pixel8*>(inP);
		PF_Pixel8* pixel_out = static_cast<PF_Pixel8*>(outP);
		*pixel_out = SampleBilinear<PF_Pixel8>(
			static_cast<PF_Pixel8*>(renderData->input_data),
			srcX, srcY,
			renderData->width, renderData->height,
			renderData->input_rowbytes,
			info->x_tiles,
			info->y_tiles,
			info->mirror
		);
	}

	return err;
}


static PF_Err
ProcessAutoShake8(
	void* refcon,
	A_long xL,
	A_long yL,
	PF_Pixel* inP,
	PF_Pixel* outP)
{
	return ProcessAutoShake(refcon, xL, yL, inP, outP, PF_PixelFormat_ARGB32);
}

static PF_Err
ProcessAutoShake16(
	void* refcon,
	A_long xL,
	A_long yL,
	PF_Pixel16* inP,
	PF_Pixel16* outP)
{
	return ProcessAutoShake(refcon, xL, yL, inP, outP, PF_PixelFormat_ARGB64);
}

static PF_Err
ProcessAutoShakeFloat(
	void* refcon,
	A_long xL,
	A_long yL,
	PF_PixelFloat* inP,
	PF_PixelFloat* outP)
{
	return ProcessAutoShake(refcon, xL, yL, inP, outP, PF_PixelFormat_ARGB128);
}


static PF_Err
LegacyRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	ThreadRenderData render_data;
	AEFX_CLR_STRUCT(render_data);

	render_data.info.magnitude = params[AUTOSHAKE_MAGNITUDE]->u.fs_d.value;
	render_data.info.frequency = params[AUTOSHAKE_FREQUENCY]->u.fs_d.value;
	render_data.info.evolution = params[AUTOSHAKE_EVOLUTION]->u.fs_d.value;
	render_data.info.seed = params[AUTOSHAKE_SEED]->u.fs_d.value;
	render_data.info.angle = params[AUTOSHAKE_ANGLE]->u.fs_d.value;
	render_data.info.slack = params[AUTOSHAKE_SLACK]->u.fs_d.value;
	render_data.info.zshake = params[AUTOSHAKE_ZSHAKE]->u.fs_d.value;

	render_data.info.x_tiles = params[X_TILES_DISK_ID]->u.bd.value;
	render_data.info.y_tiles = params[Y_TILES_DISK_ID]->u.bd.value;
	render_data.info.mirror = params[MIRROR_DISK_ID]->u.bd.value;

	bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);

	AEGP_LayerIDVal layer_id = 0;
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

	render_data.current_time = current_time;

	PF_PixelFormat pixelFormat;
	PF_WorldSuite2* wsP = NULL;
	ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
	if (!err) {
		ERR(wsP->PF_GetPixelFormat(output, &pixelFormat));
		suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
	}

	render_data.width = output->width;
	render_data.height = output->height;
	render_data.input_data = params[AUTOSHAKE_INPUT]->u.ld.data;
	render_data.input_rowbytes = params[AUTOSHAKE_INPUT]->u.ld.rowbytes;

	switch (pixelFormat) {
	case PF_PixelFormat_ARGB128: {
		AEFX_SuiteScoper<PF_iterateFloatSuite2> iterateFloatSuite =
			AEFX_SuiteScoper<PF_iterateFloatSuite2>(in_data,
				kPFIterateFloatSuite,
				kPFIterateFloatSuiteVersion2,
				out_data);
		ERR(iterateFloatSuite->iterate(in_data,
			0,
			output->height,
			&params[AUTOSHAKE_INPUT]->u.ld,
			NULL,
			(void*)&render_data,
			ProcessAutoShakeFloat,
			output));
		break;
	}
	case PF_PixelFormat_ARGB64: {
		AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
			AEFX_SuiteScoper<PF_iterate16Suite2>(in_data,
				kPFIterate16Suite,
				kPFIterate16SuiteVersion2,
				out_data);
		ERR(iterate16Suite->iterate(in_data,
			0,
			output->height,
			&params[AUTOSHAKE_INPUT]->u.ld,
			NULL,
			(void*)&render_data,
			ProcessAutoShake16,
			output));
		break;
	}
	case PF_PixelFormat_ARGB32:
	default: {
		AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
			AEFX_SuiteScoper<PF_Iterate8Suite2>(in_data,
				kPFIterate8Suite,
				kPFIterate8SuiteVersion2,
				out_data);
		ERR(iterate8Suite->iterate(in_data,
			0,
			output->height,
			&params[AUTOSHAKE_INPUT]->u.ld,
			NULL,
			(void*)&render_data,
			ProcessAutoShake8,
			output));
		break;
	}
	}

	return err;
}



static void
DisposePreRenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		ThreadRenderData* renderData = reinterpret_cast<ThreadRenderData*>(pre_render_dataPV);
		free(renderData);
	}
}

static PF_Err
PreRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PreRenderExtra* extra)
{
	PF_Err err = PF_Err_NONE;
	PF_CheckoutResult in_result;
	PF_RenderRequest req = extra->input->output_request;

	extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

	ThreadRenderData* renderData = reinterpret_cast<ThreadRenderData*>(malloc(sizeof(ThreadRenderData)));
	AEFX_CLR_STRUCT(*renderData);

	if (renderData) {
		PF_ParamDef param_copy;
		AEFX_CLR_STRUCT(param_copy);

		ERR(PF_CHECKOUT_PARAM(in_data, NORMAL_CHECKBOX_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.normal_mode = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_MAGNITUDE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.magnitude = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_FREQUENCY, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.frequency = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_EVOLUTION, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.evolution = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_SEED, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.seed = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.angle = param_copy.u.ad.value / 65536.0;

		ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_SLACK, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.slack = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, AUTOSHAKE_ZSHAKE, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.zshake = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, X_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.x_tiles = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, Y_TILES_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.y_tiles = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, MIRROR_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.mirror = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_CHECKBOX_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.compatibility_mode = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_MAGNITUDE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.compatibility_magnitude = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_SPEED_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.compatibility_speed = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_EVOLUTION_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.compatibility_evolution = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_SEED_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.compatibility_seed = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_ANGLE_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.compatibility_angle = param_copy.u.ad.value / 65536.0;

		ERR(PF_CHECKOUT_PARAM(in_data, COMPATIBILITY_SLACK_DISK_ID, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
		if (!err) renderData->info.compatibility_slack = param_copy.u.fs_d.value;

		bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);
		renderData->has_frequency_keyframes = has_frequency_keyframes;

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

		renderData->current_time = current_time;
		renderData->prev_time = prev_time;
		renderData->duration = duration;

		renderData->accumulated_phase = 0.0f;
		renderData->accumulated_phase_initialized = false;

		if (has_frequency_keyframes && renderData->info.frequency > 0) {
			PF_FpLong accumulated_phase;
			err = valueAtTimeHz(in_data, AUTOSHAKE_FREQUENCY, current_time, duration, renderData, &accumulated_phase);
			if (!err) {
				renderData->accumulated_phase = accumulated_phase;
				renderData->accumulated_phase_initialized = true;
			}
		}

		extra->output->pre_render_data = renderData;
		extra->output->delete_pre_render_data_func = DisposePreRenderData;

		ERR(extra->cb->checkout_layer(in_data->effect_ref,
			AUTOSHAKE_INPUT,
			AUTOSHAKE_INPUT,
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
				ShakeInfo info;
			} detection_data;

			detection_data.has_frequency_keyframes = has_frequency_keyframes ? 1 : 0;
			detection_data.time_offset = 0; // No longer using time_offset for half-frame
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


static size_t
RoundUp(
	size_t inValue,
	size_t inMultiple)
{
	return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}

size_t DivideRoundUp(
	size_t inValue,
	size_t inMultiple)
{
	return inValue ? (inValue + inMultiple - 1) / inMultiple : 0;
}


static PF_Err
SmartRenderCPU(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	ThreadRenderData* renderData)
{
	PF_Err err = PF_Err_NONE;

	PF_RationalScale downsample_x = in_data->downsample_x;
	PF_RationalScale downsample_y = in_data->downsample_y;

	PF_FpLong downsample_factor_x = (PF_FpLong)downsample_x.num / (PF_FpLong)downsample_x.den;
	PF_FpLong downsample_factor_y = (PF_FpLong)downsample_y.num / (PF_FpLong)downsample_y.den;

	renderData->width = input_worldP->width;
	renderData->height = input_worldP->height;
	renderData->input_data = input_worldP->data;
	renderData->input_rowbytes = input_worldP->rowbytes;

	PF_FpLong original_magnitude = renderData->info.magnitude;
	PF_FpLong original_slack = renderData->info.slack;
	PF_FpLong original_compatibility_magnitude = renderData->info.compatibility_magnitude;
	PF_FpLong original_compatibility_slack = renderData->info.compatibility_slack;

	if (renderData->info.normal_mode) {
		renderData->info.magnitude *= downsample_factor_x;      
		renderData->info.slack *= downsample_factor_y / downsample_factor_x;         
	}

	if (renderData->info.compatibility_mode) {
		renderData->info.compatibility_magnitude *= downsample_factor_x;      
		renderData->info.compatibility_slack *= downsample_factor_y / downsample_factor_x;         
	}

	if (!err) {
		switch (pixel_format) {
		case PF_PixelFormat_ARGB128: {
			AEFX_SuiteScoper<PF_iterateFloatSuite2> iterateFloatSuite =
				AEFX_SuiteScoper<PF_iterateFloatSuite2>(in_data,
					kPFIterateFloatSuite,
					kPFIterateFloatSuiteVersion2,
					out_data);
			ERR(iterateFloatSuite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)renderData,
				ProcessAutoShakeFloat,
				output_worldP));
			break;
		}
		case PF_PixelFormat_ARGB64: {
			AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
				AEFX_SuiteScoper<PF_iterate16Suite2>(in_data,
					kPFIterate16Suite,
					kPFIterate16SuiteVersion2,
					out_data);
			ERR(iterate16Suite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)renderData,
				ProcessAutoShake16,
				output_worldP));
			break;
		}
		case PF_PixelFormat_ARGB32: {
			AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
				AEFX_SuiteScoper<PF_Iterate8Suite2>(in_data,
					kPFIterate8Suite,
					kPFIterate8SuiteVersion2,
					out_data);
			ERR(iterate8Suite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)renderData,
				ProcessAutoShake8,
				output_worldP));
			break;
		}
		default:
			err = PF_Err_BAD_CALLBACK_PARAM;
			break;
		}
	}

	renderData->info.magnitude = original_magnitude;
	renderData->info.slack = original_slack;
	renderData->info.compatibility_magnitude = original_compatibility_magnitude;
	renderData->info.compatibility_slack = original_compatibility_slack;

	return err;
}

typedef struct
{
	int mSrcPitch;
	int mDstPitch;
	int m16f;
	int mWidth;
	int mHeight;
	float mMagnitude;
	float mFrequency;
	float mEvolution;
	float mSeed;
	float mAngle;
	float mSlack;
	float mZShake;
	int mXTiles;
	int mYTiles;
	int mMirror;
	float mCurrentTime;
	float mDownsampleX;
	float mDownsampleY;
	int mNormalMode;
	int mCompatibilityMode;
	float mCompatibilityMagnitude;
	float mCompatibilitySpeed;
	float mCompatibilityEvolution;
	float mCompatibilitySeed;
	float mCompatibilityAngle;
	float mCompatibilitySlack;
	float mAccumulatedPhase;
	int mHasFrequencyKeyframes;
} AutoShakeParams;


static PF_Err
SmartRenderGPU(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	ThreadRenderData* renderData)
{
	PF_Err err = PF_Err_NONE;

	PF_RationalScale downsample_x = in_dataP->downsample_x;
	PF_RationalScale downsample_y = in_dataP->downsample_y;

	PF_FpLong downsample_factor_x = (PF_FpLong)downsample_x.num / (PF_FpLong)downsample_x.den;
	PF_FpLong downsample_factor_y = (PF_FpLong)downsample_y.num / (PF_FpLong)downsample_y.den;

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite = AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP,
		kPFGPUDeviceSuite,
		kPFGPUDeviceSuiteVersion1,
		out_dataP);

	if (pixel_format != PF_PixelFormat_GPU_BGRA128) {
		err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
	}
	A_long bytes_per_pixel = 16;

	PF_GPUDeviceInfo device_info;
	ERR(gpu_suite->GetDeviceInfo(in_dataP->effect_ref, extraP->input->device_index, &device_info));

	void* src_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, input_worldP, &src_mem));

	void* dst_mem = 0;
	ERR(gpu_suite->GetGPUWorldData(in_dataP->effect_ref, output_worldP, &dst_mem));

	AutoShakeParams autoshake_params;
	autoshake_params.mWidth = input_worldP->width;
	autoshake_params.mHeight = input_worldP->height;
	autoshake_params.mSrcPitch = input_worldP->rowbytes / bytes_per_pixel;
	autoshake_params.mDstPitch = output_worldP->rowbytes / bytes_per_pixel;
	autoshake_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

	autoshake_params.mMagnitude = renderData->info.magnitude;
	autoshake_params.mFrequency = renderData->info.frequency;
	autoshake_params.mEvolution = renderData->info.evolution;
	autoshake_params.mSeed = renderData->info.seed;
	autoshake_params.mAngle = renderData->info.angle;
	autoshake_params.mSlack = renderData->info.slack;
	autoshake_params.mZShake = renderData->info.zshake;
	autoshake_params.mXTiles = renderData->info.x_tiles;
	autoshake_params.mYTiles = renderData->info.y_tiles;
	autoshake_params.mMirror = renderData->info.mirror;
	autoshake_params.mCurrentTime = renderData->current_time;
	autoshake_params.mDownsampleX = downsample_factor_x;
	autoshake_params.mDownsampleY = downsample_factor_y;
	autoshake_params.mAccumulatedPhase = renderData->accumulated_phase;
	autoshake_params.mHasFrequencyKeyframes = renderData->has_frequency_keyframes ? 1 : 0;

	int normal_mode = renderData->info.normal_mode;
	int compatibility_mode = renderData->info.compatibility_mode;

	float compatibility_magnitude = renderData->info.compatibility_magnitude;
	float compatibility_speed = renderData->info.compatibility_speed;
	float compatibility_evolution = renderData->info.compatibility_evolution;
	float compatibility_seed = renderData->info.compatibility_seed;
	float compatibility_angle = renderData->info.compatibility_angle;
	float compatibility_slack = renderData->info.compatibility_slack;

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mMagnitude));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mFrequency));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mEvolution));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mSeed));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mAngle));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mSlack));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mZShake));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.mXTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.mYTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.mMirror));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mCurrentTime));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mDownsampleX));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mDownsampleY));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &normal_mode));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &compatibility_mode));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &compatibility_magnitude));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &compatibility_speed));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &compatibility_evolution));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &compatibility_seed));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &compatibility_angle));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &compatibility_slack));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(float), &autoshake_params.mAccumulatedPhase));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->autoshake_kernel, param_index++, sizeof(int), &autoshake_params.mHasFrequencyKeyframes));

		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(autoshake_params.mWidth, threadBlock[0]), RoundUp(autoshake_params.mHeight, threadBlock[1]) };

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->autoshake_kernel,
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
		AutoShake_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			autoshake_params.mSrcPitch,
			autoshake_params.mDstPitch,
			autoshake_params.m16f,
			autoshake_params.mWidth,
			autoshake_params.mHeight,
			autoshake_params.mMagnitude,
			autoshake_params.mFrequency,
			autoshake_params.mEvolution,
			autoshake_params.mSeed,
			autoshake_params.mAngle,
			autoshake_params.mSlack,
			autoshake_params.mZShake,
			autoshake_params.mXTiles,
			autoshake_params.mYTiles,
			autoshake_params.mMirror,
			autoshake_params.mCurrentTime,
			autoshake_params.mDownsampleX,
			autoshake_params.mDownsampleY,
			normal_mode,
			compatibility_mode,
			compatibility_magnitude,
			compatibility_speed,
			compatibility_evolution,
			compatibility_seed,
			compatibility_angle,
			compatibility_slack,
			autoshake_params.mAccumulatedPhase,
			autoshake_params.mHasFrequencyKeyframes);

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

		struct CompleteAutoShakeParams {
			int mSrcPitch;
			int mDstPitch;
			int m16f;
			int mWidth;
			int mHeight;
			float mMagnitude;
			float mFrequency;
			float mEvolution;
			float mSeed;
			float mAngle;
			float mSlack;
			float mZShake;
			int mXTiles;
			int mYTiles;
			int mMirror;
			float mCurrentTime;
			float mDownsampleX;
			float mDownsampleY;
			int mNormalMode;
			int mCompatibilityMode;
			float mCompatibilityMagnitude;
			float mCompatibilitySpeed;
			float mCompatibilityEvolution;
			float mCompatibilitySeed;
			float mCompatibilityAngle;
			float mCompatibilitySlack;
			float mAccumulatedPhase;
			int mHasFrequencyKeyframes;
		} complete_params;

		complete_params.mSrcPitch = autoshake_params.mSrcPitch;
		complete_params.mDstPitch = autoshake_params.mDstPitch;
		complete_params.m16f = autoshake_params.m16f;
		complete_params.mWidth = autoshake_params.mWidth;
		complete_params.mHeight = autoshake_params.mHeight;
		complete_params.mMagnitude = autoshake_params.mMagnitude;
		complete_params.mFrequency = autoshake_params.mFrequency;
		complete_params.mEvolution = autoshake_params.mEvolution;
		complete_params.mSeed = autoshake_params.mSeed;
		complete_params.mAngle = autoshake_params.mAngle;
		complete_params.mSlack = autoshake_params.mSlack;
		complete_params.mZShake = autoshake_params.mZShake;
		complete_params.mXTiles = autoshake_params.mXTiles;
		complete_params.mYTiles = autoshake_params.mYTiles;
		complete_params.mMirror = autoshake_params.mMirror;
		complete_params.mCurrentTime = autoshake_params.mCurrentTime;
		complete_params.mDownsampleX = autoshake_params.mDownsampleX;
		complete_params.mDownsampleY = autoshake_params.mDownsampleY;

		complete_params.mNormalMode = normal_mode;
		complete_params.mCompatibilityMode = compatibility_mode;
		complete_params.mCompatibilityMagnitude = compatibility_magnitude;
		complete_params.mCompatibilitySpeed = compatibility_speed;
		complete_params.mCompatibilityEvolution = compatibility_evolution;
		complete_params.mCompatibilitySeed = compatibility_seed;
		complete_params.mCompatibilityAngle = compatibility_angle;
		complete_params.mCompatibilitySlack = compatibility_slack;
		complete_params.mAccumulatedPhase = autoshake_params.mAccumulatedPhase;
		complete_params.mHasFrequencyKeyframes = autoshake_params.mHasFrequencyKeyframes;

		DXShaderExecution shaderExecution(
			dx_gpu_data->mContext,
			dx_gpu_data->mAutoShakeShader,
			3);

		DX_ERR(shaderExecution.SetParamBuffer(&complete_params, sizeof(CompleteAutoShakeParams)));
		DX_ERR(shaderExecution.SetUnorderedAccessView(
			(ID3D12Resource*)dst_mem,
			autoshake_params.mHeight * output_worldP->rowbytes));
		DX_ERR(shaderExecution.SetShaderResourceView(
			(ID3D12Resource*)src_mem,
			autoshake_params.mHeight * input_worldP->rowbytes));
		DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(autoshake_params.mWidth, 16), (UINT)DivideRoundUp(autoshake_params.mHeight, 16)));
	}
#endif
#if HAS_METAL
	else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		Handle metal_handle = (Handle)extraP->input->gpu_data;
		MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

		struct MetalAutoShakeParams {
			int mSrcPitch;
			int mDstPitch;
			int m16f;
			unsigned int mWidth;
			unsigned int mHeight;
			float mMagnitude;
			float mFrequency;
			float mEvolution;
			float mSeed;
			float mAngle;
			float mSlack;
			float mZShake;
			int mXTiles;
			int mYTiles;
			int mMirror;
			float mCurrentTime;
			float mDownsampleX;
			float mDownsampleY;
			int mNormalMode;
			int mCompatibilityMode;
			float mCompatibilityMagnitude;
			float mCompatibilitySpeed;
			float mCompatibilityEvolution;
			float mCompatibilitySeed;
			float mCompatibilityAngle;
			float mCompatibilitySlack;
			float mAccumulatedPhase;
			int mHasFrequencyKeyframes;
		} metal_params;

		metal_params.mSrcPitch = autoshake_params.mSrcPitch;
		metal_params.mDstPitch = autoshake_params.mDstPitch;
		metal_params.m16f = autoshake_params.m16f;
		metal_params.mWidth = autoshake_params.mWidth;
		metal_params.mHeight = autoshake_params.mHeight;
		metal_params.mMagnitude = autoshake_params.mMagnitude;
		metal_params.mFrequency = autoshake_params.mFrequency;
		metal_params.mEvolution = autoshake_params.mEvolution;
		metal_params.mSeed = autoshake_params.mSeed;
		metal_params.mAngle = autoshake_params.mAngle;
		metal_params.mSlack = autoshake_params.mSlack;
		metal_params.mZShake = autoshake_params.mZShake;
		metal_params.mXTiles = autoshake_params.mXTiles;
		metal_params.mYTiles = autoshake_params.mYTiles;
		metal_params.mMirror = autoshake_params.mMirror;
		metal_params.mCurrentTime = autoshake_params.mCurrentTime;
		metal_params.mDownsampleX = autoshake_params.mDownsampleX;
		metal_params.mDownsampleY = autoshake_params.mDownsampleY;

		metal_params.mNormalMode = normal_mode;
		metal_params.mCompatibilityMode = compatibility_mode;
		metal_params.mCompatibilityMagnitude = compatibility_magnitude;
		metal_params.mCompatibilitySpeed = compatibility_speed;
		metal_params.mCompatibilityEvolution = compatibility_evolution;
		metal_params.mCompatibilitySeed = compatibility_seed;
		metal_params.mCompatibilityAngle = compatibility_angle;
		metal_params.mCompatibilitySlack = compatibility_slack;
		metal_params.mAccumulatedPhase = autoshake_params.mAccumulatedPhase;
		metal_params.mHasFrequencyKeyframes = autoshake_params.mHasFrequencyKeyframes;

		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &metal_params
			length : sizeof(MetalAutoShakeParams)
			options : MTLResourceStorageModeManaged]autorelease];

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->autoshake_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = { DivideRoundUp(autoshake_params.mWidth, threadsPerGroup.width), DivideRoundUp(autoshake_params.mHeight, threadsPerGroup.height), 1 };

		[computeEncoder setComputePipelineState : metal_dataP->autoshake_pipeline] ;
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

	ThreadRenderData* renderData = reinterpret_cast<ThreadRenderData*>(extraP->input->pre_render_data);

	if (renderData) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, AUTOSHAKE_INPUT, &input_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
			kPFWorldSuite,
			kPFWorldSuiteVersion2,
			out_data);
		PF_PixelFormat	pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		if (isGPU) {
			ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, renderData));
		}
		else {
			ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, renderData));
		}
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, AUTOSHAKE_INPUT));
	}
	else {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
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
		"Auto-Shake",     
		"DKT Auto-Shake",   
		"DKT Effects",    
		AE_RESERVED_INFO,   
		"EffectMain",      
		"https://www.adobe.com");   

	return result;
}

PF_Err
EffectMain(
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

		case PF_Cmd_GPU_DEVICE_SETUP:
			err = GPUDeviceSetup(in_data, out_data, (PF_GPUDeviceSetupExtra*)extra);
			break;

		case PF_Cmd_GPU_DEVICE_SETDOWN:
			err = GPUDeviceSetdown(in_data, out_data, (PF_GPUDeviceSetdownExtra*)extra);
			break;

		case PF_Cmd_SMART_PRE_RENDER:
			err = PreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
			break;

		case PF_Cmd_SMART_RENDER:
			err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, false);
			break;

		case PF_Cmd_SMART_RENDER_GPU:
			err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, true);
			break;

		case PF_Cmd_RENDER:
			err = LegacyRender(in_data, out_data, params, output);
			break;
		}
	}
	catch (PF_Err& thrown_err) {
		err = thrown_err;
	}

	return err;
}




