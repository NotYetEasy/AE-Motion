#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "RandomDisplacement.h"
#include "SimplexNoise.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

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

extern void RandomDisplacement_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float magnitude,
	float evolution,
	float seed,
	float scatter,
	int x_tiles,
	int y_tiles,
	int mirror,
	float downsample_x,
	float downsample_y);


static PF_Err
About(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_SPRINTF(out_data->return_msg,
		"%s, v%d.%d\r%s",
		NAME,
		MAJOR_VERSION,
		MINOR_VERSION,
		DESCRIPTION);

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

	SimplexNoise::initPerm();

	out_data->my_version = PF_VERSION(MAJOR_VERSION,
		MINOR_VERSION,
		BUG_VERSION,
		STAGE_VERSION,
		BUILD_VERSION);

	out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
		PF_OutFlag_NON_PARAM_VARY;

	out_data->out_flags2 = PF_OutFlag2_FLOAT_COLOR_AWARE |
		PF_OutFlag2_SUPPORTS_SMART_RENDER |
		PF_OutFlag2_SUPPORTS_THREADED_RENDERING;

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
	PF_Err			err = PF_Err_NONE;
	PF_ParamDef		def;

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX("Magnitude",
		0,
		2000,
		0,
		200,
		50,
		PF_Precision_INTEGER,
		0,
		0,
		RANDOM_DISPLACEMENT_MAGNITUDE);

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX("Evolution",
		0,
		2000,
		0,
		5,
		0,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		RANDOM_DISPLACEMENT_EVOLUTION);

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
		RANDOM_DISPLACEMENT_SEED);

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX("Scatter",
		0,
		2,
		0,
		2,
		0.5,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		RANDOM_DISPLACEMENT_SCATTER);

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

	out_data->num_params = RANDOM_DISPLACEMENT_NUM_PARAMS;

	return err;
}

#if HAS_METAL
PF_Err NSError2PFErr(NSError* inError)
{
	if (inError)
	{
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
	return PF_Err_NONE;
}
#endif 

struct OpenCLGPUData
{
	cl_kernel displacement_kernel;
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
	ShaderObjectPtr mDisplacementShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState>displacement_pipeline;
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

		size_t sizes[] = { strlen(k16fString), strlen(kRandomDisplacementKernel_OpenCLString) };
		char const* strings[] = { k16fString, kRandomDisplacementKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->displacement_kernel = clCreateKernel(program, "RandomDisplacementKernel", &result);
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
		dx_gpu_data->mDisplacementShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"RandomDisplacementKernel", csoPath, sigPath));

		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mDisplacementShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		NSString* source = [NSString stringWithCString : kRandomDisplacement_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
			id<MTLFunction> displacement_function = nil;
			NSString* displacement_name = [NSString stringWithCString : "RandomDisplacementKernel" encoding : NSUTF8StringEncoding];

			displacement_function = [[library newFunctionWithName : displacement_name]autorelease];

			if (!displacement_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->displacement_pipeline = [device newComputePipelineStateWithFunction : displacement_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->displacement_kernel);

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
		dx_gpu_dataP->mDisplacementShader.reset();

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

	return err;
}



static void ComputeDisplacement(
	double x,
	double y,
	double evolution,
	double seed,
	double scatter,
	double magnitude,
	double* dx,
	double* dy)
{
	double noise_dx = SimplexNoise::simplex_noise(x * scatter / 50.0 + seed * 54623.245, y * scatter / 500.0, evolution + seed * 49235.319798, 3);

	double noise_dy = SimplexNoise::simplex_noise(x * scatter / 50.0, y * scatter / 500.0 + seed * 8723.5647, evolution + 7468.329 + seed * 19337.940385, 3);

	*dx = -magnitude * noise_dx;
	*dy = magnitude * noise_dy;
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

static void
DisposePreRenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		DisplacementInfo* infoP = reinterpret_cast<DisplacementInfo*>(pre_render_dataPV);
		free(infoP);
	}
}

static PF_Err
PreRender(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_PreRenderExtra* extraP)
{
	PF_Err err = PF_Err_NONE;
	PF_CheckoutResult in_result;
	PF_RenderRequest req = extraP->input->output_request;

	extraP->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;

	DisplacementInfo* infoP = reinterpret_cast<DisplacementInfo*>(malloc(sizeof(DisplacementInfo)));

	if (infoP) {
		PF_ParamDef param_copy;
		AEFX_CLR_STRUCT(param_copy);

		ERR(PF_CHECKOUT_PARAM(in_dataP, RANDOM_DISPLACEMENT_MAGNITUDE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &param_copy));
		infoP->magnitude = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, RANDOM_DISPLACEMENT_EVOLUTION, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &param_copy));
		infoP->evolution = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, RANDOM_DISPLACEMENT_SEED, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &param_copy));
		infoP->seed = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, RANDOM_DISPLACEMENT_SCATTER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &param_copy));
		infoP->scatter = param_copy.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, X_TILES_DISK_ID, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &param_copy));
		infoP->x_tiles = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, Y_TILES_DISK_ID, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &param_copy));
		infoP->y_tiles = param_copy.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, MIRROR_DISK_ID, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &param_copy));
		infoP->mirror = param_copy.u.bd.value;

		ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
			RANDOM_DISPLACEMENT_INPUT,
			RANDOM_DISPLACEMENT_INPUT,
			&req,
			in_dataP->current_time,
			in_dataP->time_step,
			in_dataP->time_scale,
			&in_result));

		if (!err) {
			infoP->width = in_result.max_result_rect.right - in_result.max_result_rect.left;
			infoP->height = in_result.max_result_rect.bottom - in_result.max_result_rect.top;
		}

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;

		UnionLRect(&in_result.result_rect, &extraP->output->result_rect);
		UnionLRect(&in_result.max_result_rect, &extraP->output->max_result_rect);
	}
	else {
		err = PF_Err_OUT_OF_MEMORY;
	}
	return err;
}

static PF_Err
SmartRenderCPU(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	DisplacementInfo* infoP)
{
	PF_Err err = PF_Err_NONE;

	PF_RationalScale downsample_x = in_data->downsample_x;
	PF_RationalScale downsample_y = in_data->downsample_y;

	PF_FpLong downsample_factor_x = (PF_FpLong)downsample_x.num / (PF_FpLong)downsample_x.den;
	PF_FpLong downsample_factor_y = (PF_FpLong)downsample_y.num / (PF_FpLong)downsample_y.den;

	A_long width = output_worldP->width;
	A_long height = output_worldP->height;
	A_long input_width = input_worldP->width;
	A_long input_height = input_worldP->height;
	A_long rowbytes = input_worldP->rowbytes;

	double layerX = width / 2.0;
	double layerY = height / 2.0;
	double dx_original, dy_original;
	double dx, dy;

	ComputeDisplacement(
		layerX,
		layerY,
		infoP->evolution,
		infoP->seed,
		infoP->scatter,
		infoP->magnitude,
		&dx_original,
		&dy_original
	);

	ComputeDisplacement(
		layerX,
		layerY,
		infoP->evolution,
		infoP->seed,
		infoP->scatter / downsample_factor_x,
		infoP->magnitude * downsample_factor_x,
		&dx,
		&dy
	);

	dy *= downsample_factor_y / downsample_factor_x;

	switch (pixel_format) {
	case PF_PixelFormat_ARGB128: {
		for (A_long y = 0; y < height; y++) {
			PF_PixelFloat* outP = (PF_PixelFloat*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = x - dx;
				PF_FpLong srcY = y - dy;

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
				PF_FpLong srcX = x - dx;
				PF_FpLong srcY = y - dy;

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
	case PF_PixelFormat_ARGB32: {
		for (A_long y = 0; y < height; y++) {
			PF_Pixel8* outP = (PF_Pixel8*)((char*)output_worldP->data + y * output_worldP->rowbytes);

			for (A_long x = 0; x < width; x++, outP++) {
				PF_FpLong srcX = x - dx;
				PF_FpLong srcY = y - dy;

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
	default:
		err = PF_Err_BAD_CALLBACK_PARAM;
		break;
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


typedef struct
{
	int mSrcPitch;
	int mDstPitch;
	int m16f;
	int mWidth;
	int mHeight;
	float mMagnitude;
	float mEvolution;
	float mSeed;
	float mScatter;
	int mXTiles;
	int mYTiles;
	int mMirror;
	float mDownsampleX;
	float mDownsampleY;
} DisplacementParams;


static PF_Err
SmartRenderGPU(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	DisplacementInfo* infoP)
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

	DisplacementParams params;

	params.mWidth = input_worldP->width;
	params.mHeight = input_worldP->height;

	A_long src_row_bytes = input_worldP->rowbytes;
	A_long dst_row_bytes = output_worldP->rowbytes;

	params.mSrcPitch = src_row_bytes / bytes_per_pixel;
	params.mDstPitch = dst_row_bytes / bytes_per_pixel;
	params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

	params.mMagnitude = infoP->magnitude;
	params.mEvolution = infoP->evolution;
	params.mSeed = infoP->seed;
	params.mScatter = infoP->scatter;
	params.mXTiles = infoP->x_tiles;
	params.mYTiles = infoP->y_tiles;
	params.mMirror = infoP->mirror;
	params.mDownsampleX = downsample_factor_x;
	params.mDownsampleY = downsample_factor_y;

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(int), &params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(int), &params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(int), &params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(int), &params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(int), &params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(float), &params.mMagnitude));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(float), &params.mEvolution));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(float), &params.mSeed));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(float), &params.mScatter));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(int), &params.mXTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(int), &params.mYTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(int), &params.mMirror));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(float), &params.mDownsampleX));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->displacement_kernel, param_index++, sizeof(float), &params.mDownsampleY));

		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->displacement_kernel,
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
		RandomDisplacement_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			params.mSrcPitch,
			params.mDstPitch,
			params.m16f,
			params.mWidth,
			params.mHeight,
			params.mMagnitude,
			params.mEvolution,
			params.mSeed,
			params.mScatter,
			params.mXTiles,
			params.mYTiles,
			params.mMirror,
			params.mDownsampleX,
			params.mDownsampleY);

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
			dx_gpu_data->mDisplacementShader,
			3);

		DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(DisplacementParams)));
		DX_ERR(shaderExecution.SetUnorderedAccessView(
			(ID3D12Resource*)dst_mem,
			params.mHeight * dst_row_bytes));
		DX_ERR(shaderExecution.SetShaderResourceView(
			(ID3D12Resource*)src_mem,
			params.mHeight * src_row_bytes));
		DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(params.mWidth, 16), (UINT)DivideRoundUp(params.mHeight, 16)));
	}
#endif
#if HAS_METAL
	else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		Handle metal_handle = (Handle)extraP->input->gpu_data;
		MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> param_buffer = [[device newBufferWithBytes : &params
			length : sizeof(DisplacementParams)
			options : MTLResourceStorageModeManaged]autorelease];

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->displacement_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = { DivideRoundUp(params.mWidth, threadsPerGroup.width), DivideRoundUp(params.mHeight, threadsPerGroup.height), 1 };

		[computeEncoder setComputePipelineState : metal_dataP->displacement_pipeline] ;
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

	DisplacementInfo* infoP = reinterpret_cast<DisplacementInfo*>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, RANDOM_DISPLACEMENT_INPUT, &input_worldP)));

		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
			kPFWorldSuite,
			kPFWorldSuiteVersion2,
			out_data);
		PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		if (isGPU) {
			ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
		}
		else {
			ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
		}
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, RANDOM_DISPLACEMENT_INPUT));
	}
	else {
		err = PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	return err;
}

static PF_Err
Render(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err err = PF_Err_NONE;

	if (in_dataP->appl_id == 'PrMr')
	{
		PF_LayerDef* src = &params[0]->u.ld;
		PF_LayerDef* dest = output;

		const char* srcData = (const char*)src->data;
		char* destData = (char*)dest->data;

		float magnitude = (float)params[RANDOM_DISPLACEMENT_MAGNITUDE]->u.fs_d.value;
		float evolution = (float)params[RANDOM_DISPLACEMENT_EVOLUTION]->u.fs_d.value;
		float seed = (float)params[RANDOM_DISPLACEMENT_SEED]->u.fs_d.value;
		float scatter = (float)params[RANDOM_DISPLACEMENT_SCATTER]->u.fs_d.value;
		bool x_tiles = params[X_TILES_DISK_ID]->u.bd.value;
		bool y_tiles = params[Y_TILES_DISK_ID]->u.bd.value;
		bool mirror = params[MIRROR_DISK_ID]->u.bd.value;

		for (int y = 0; y < output->height; ++y, srcData += src->rowbytes, destData += dest->rowbytes)
		{
			memcpy(destData, srcData, src->width * 4 * sizeof(float));
		}
	}
	else
	{
		PF_LayerDef* src = &params[RANDOM_DISPLACEMENT_INPUT]->u.ld;
		PF_LayerDef* dst = output;

		DisplacementInfo info;
		info.magnitude = params[RANDOM_DISPLACEMENT_MAGNITUDE]->u.fs_d.value;
		info.evolution = params[RANDOM_DISPLACEMENT_EVOLUTION]->u.fs_d.value;
		info.seed = params[RANDOM_DISPLACEMENT_SEED]->u.fs_d.value;
		info.scatter = params[RANDOM_DISPLACEMENT_SCATTER]->u.fs_d.value;
		info.x_tiles = params[X_TILES_DISK_ID]->u.bd.value;
		info.y_tiles = params[Y_TILES_DISK_ID]->u.bd.value;
		info.mirror = params[MIRROR_DISK_ID]->u.bd.value;
		info.width = src->width;
		info.height = src->height;

		PF_PixelFormat pixelFormat;
		AEGP_SuiteHandler suites(in_dataP->pica_basicP);
		PF_WorldSuite2* wsP = NULL;
		ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
		ERR(wsP->PF_GetPixelFormat(src, &pixelFormat));
		suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);

		double layerX = dst->width / 2.0;
		double layerY = dst->height / 2.0;
		double dx, dy;

		ComputeDisplacement(
			layerX,
			layerY,
			info.evolution,
			info.seed,
			info.scatter,
			info.magnitude,
			&dx,
			&dy
		);

		switch (pixelFormat) {
		case PF_PixelFormat_ARGB128: {
			for (A_long y = 0; y < dst->height; y++) {
				PF_PixelFloat* dstP = (PF_PixelFloat*)((char*)dst->data + y * dst->rowbytes);

				for (A_long x = 0; x < dst->width; x++, dstP++) {
					PF_FpLong srcX = x - dx;
					PF_FpLong srcY = y - dy;

					*dstP = SampleBilinear<PF_PixelFloat>(
						(PF_PixelFloat*)src->data,
						srcX, srcY,
						info.width, info.height,
						src->rowbytes,
						info.x_tiles,
						info.y_tiles,
						info.mirror
					);
				}
			}
			break;
		}
		case PF_PixelFormat_ARGB64: {
			for (A_long y = 0; y < dst->height; y++) {
				PF_Pixel16* dstP = (PF_Pixel16*)((char*)dst->data + y * dst->rowbytes);

				for (A_long x = 0; x < dst->width; x++, dstP++) {
					PF_FpLong srcX = x - dx;
					PF_FpLong srcY = y - dy;

					*dstP = SampleBilinear<PF_Pixel16>(
						(PF_Pixel16*)src->data,
						srcX, srcY,
						info.width, info.height,
						src->rowbytes,
						info.x_tiles,
						info.y_tiles,
						info.mirror
					);
				}
			}
			break;
		}
		case PF_PixelFormat_ARGB32: {
			for (A_long y = 0; y < dst->height; y++) {
				PF_Pixel8* dstP = (PF_Pixel8*)((char*)dst->data + y * dst->rowbytes);

				for (A_long x = 0; x < dst->width; x++, dstP++) {
					PF_FpLong srcX = x - dx;
					PF_FpLong srcY = y - dy;

					*dstP = SampleBilinear<PF_Pixel8>(
						(PF_Pixel8*)src->data,
						srcX, srcY,
						info.width, info.height,
						src->rowbytes,
						info.x_tiles,
						info.y_tiles,
						info.mirror
					);
				}
			}
			break;
		}
		}
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
		"Random Displacement",
		"DKT Random Displacement",
		"DKT Effects",
		AE_RESERVED_INFO,
		"EffectMain",
		"");

	return result;
}

PF_Err
EffectMain(
	PF_Cmd cmd,
	PF_InData* in_dataP,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output,
	void* extra)
{
	PF_Err err = PF_Err_NONE;

	try {
		switch (cmd)
		{
		case PF_Cmd_ABOUT:
			err = About(in_dataP, out_data, params, output);
			break;
		case PF_Cmd_GLOBAL_SETUP:
			err = GlobalSetup(in_dataP, out_data, params, output);
			break;
		case PF_Cmd_PARAMS_SETUP:
			err = ParamsSetup(in_dataP, out_data, params, output);
			break;
		case PF_Cmd_GPU_DEVICE_SETUP:
			err = GPUDeviceSetup(in_dataP, out_data, (PF_GPUDeviceSetupExtra*)extra);
			break;
		case PF_Cmd_GPU_DEVICE_SETDOWN:
			err = GPUDeviceSetdown(in_dataP, out_data, (PF_GPUDeviceSetdownExtra*)extra);
			break;
		case PF_Cmd_RENDER:
			err = Render(in_dataP, out_data, params, output);
			break;
		case PF_Cmd_SMART_PRE_RENDER:
			err = PreRender(in_dataP, out_data, (PF_PreRenderExtra*)extra);
			break;
		case PF_Cmd_SMART_RENDER:
			err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, false);
			break;
		case PF_Cmd_SMART_RENDER_GPU:
			err = SmartRender(in_dataP, out_data, (PF_SmartRenderExtra*)extra, true);
			break;
		}
	}
	catch (PF_Err& thrown_err) {
		err = thrown_err;
	}
	return err;
}

