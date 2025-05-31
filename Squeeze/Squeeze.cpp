#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "Squeeze.h"
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


extern void Squeeze_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float inStrength,
	int inXTiles,
	int inYTiles,
	int inMirror);


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

	out_data->my_version = PF_VERSION(MAJOR_VERSION,
		MINOR_VERSION,
		BUG_VERSION,
		STAGE_VERSION,
		BUILD_VERSION);

	out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;

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
		SQUEEZE_STRENGTH);

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
		SQUEEZE_X_TILES);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Y Tiles",
		"",
		FALSE,
		0,
		SQUEEZE_Y_TILES);

	AEFX_CLR_STRUCT(def);
	PF_ADD_CHECKBOX("Mirror",
		"",
		FALSE,
		0,
		SQUEEZE_MIRROR);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_END;
	PF_ADD_PARAM(in_data, -1, &def);

	out_data->num_params = SQUEEZE_NUM_PARAMS;

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
	cl_kernel squeeze_kernel;
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
	ShaderObjectPtr mSqueezeShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState> squeeze_pipeline;
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

		size_t sizes[] = { strlen(k16fString), strlen(kSqueezeKernel_OpenCLString) };
		char const* strings[] = { k16fString, kSqueezeKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->squeeze_kernel = clCreateKernel(program, "SqueezeKernel", &result);
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
		dx_gpu_data->mSqueezeShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"SqueezeKernel", csoPath, sigPath));

		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mSqueezeShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		NSString* source = [NSString stringWithCString : kSqueezeKernel_MetalString encoding : NSUTF8StringEncoding];
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
			id<MTLFunction> squeeze_function = nil;
			NSString* squeeze_name = [NSString stringWithCString : "SqueezeKernel" encoding : NSUTF8StringEncoding];

			squeeze_function = [[library newFunctionWithName : squeeze_name]autorelease];

			if (!squeeze_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->squeeze_pipeline = [device newComputePipelineStateWithFunction : squeeze_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->squeeze_kernel);

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
		dx_gpu_dataP->mSqueezeShader.reset();

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

	return err;
}

static float mix(float a, float b, float t) {
	return a * (1.0f - t) + b * t;
}

static float clamp(float value, float min_val, float max_val) {
	if (value < min_val) return min_val;
	if (value > max_val) return max_val;
	return value;
}

template <typename PixelType>
static PF_Err
SqueezeFunc(
	void* refcon,
	A_long      xL,
	A_long      yL,
	PixelType* inP,
	PixelType* outP)
{
	PF_Err          err = PF_Err_NONE;
	SqueezeInfo* siP = reinterpret_cast<SqueezeInfo*>(refcon);

	if (siP) {
		float width = static_cast<float>(siP->width);
		float height = static_cast<float>(siP->height);

		float normX = static_cast<float>(xL) / width;
		float normY = static_cast<float>(yL) / height;

		float stX = 2.0f * normX - 1.0f;
		float stY = 2.0f * normY - 1.0f;

		float str = siP->strength / 2.0f;

		float absY = fabsf(stY);
		float absX = fabsf(stX);

		const float epsilon = 0.0001f;

		float xdiv = 1.0f + (1.0f - (absY * absY)) * -str;
		float ydiv = 1.0f + (1.0f - (absX * absX)) * str;

		xdiv = clamp(xdiv, epsilon, 2.0f);
		ydiv = clamp(ydiv, epsilon, 2.0f);

		stX /= xdiv;
		stY /= ydiv;

		float newNormX = stX / 2.0f + 0.5f;
		float newNormY = stY / 2.0f + 0.5f;

		float sourceX = newNormX * width;
		float sourceY = newNormY * height;

		bool outsideBounds = false;

		if (siP->x_tiles) {
			if (siP->mirror) {
				float intPart;
				float fracPart = modff(fabsf(sourceX / width), &intPart);
				int isOdd = (int)(sourceX / width) & 1;
				sourceX = isOdd ? width * (1.0f - fracPart) : width * fracPart;
			}
			else {
				sourceX = fmodf(sourceX, width);
				if (sourceX < 0) sourceX += width;
			}
		}
		else {
			if (sourceX < 0 || sourceX >= width) {
				outsideBounds = true;
			}
		}

		if (siP->y_tiles) {
			if (siP->mirror) {
				float intPart;
				float fracPart = modff(fabsf(sourceY / height), &intPart);
				int isOdd = (int)(sourceY / height) & 1;
				sourceY = isOdd ? height * (1.0f - fracPart) : height * fracPart;
			}
			else {
				sourceY = fmodf(sourceY, height);
				if (sourceY < 0) sourceY += height;
			}
		}
		else {
			if (sourceY < 0 || sourceY >= height) {
				outsideBounds = true;
			}
		}

		if (outsideBounds) {
			outP->alpha = 0;
			outP->red = 0;
			outP->green = 0;
			outP->blue = 0;
			return err;
		}

		int x1 = static_cast<int>(sourceX);
		int y1 = static_cast<int>(sourceY);
		int x2 = x1 + 1;
		int y2 = y1 + 1;

		float fx = sourceX - x1;
		float fy = sourceY - y1;

		x1 = MIN(MAX(x1, 0), siP->width - 1);
		y1 = MIN(MAX(y1, 0), siP->height - 1);
		x2 = MIN(MAX(x2, 0), siP->width - 1);
		y2 = MIN(MAX(y2, 0), siP->height - 1);

		PixelType* p11, * p12, * p21, * p22;

		PixelType* base = reinterpret_cast<PixelType*>(siP->src);

		p11 = reinterpret_cast<PixelType*>((char*)base + (y1 * siP->rowbytes)) + x1;
		p12 = reinterpret_cast<PixelType*>((char*)base + (y2 * siP->rowbytes)) + x1;
		p21 = reinterpret_cast<PixelType*>((char*)base + (y1 * siP->rowbytes)) + x2;
		p22 = reinterpret_cast<PixelType*>((char*)base + (y2 * siP->rowbytes)) + x2;

		float oneMinusFx = 1.0f - fx;
		float oneMinusFy = 1.0f - fy;

		float w00 = oneMinusFx * oneMinusFy;
		float w10 = fx * oneMinusFy;
		float w01 = oneMinusFx * fy;
		float w11 = fx * fy;

		outP->alpha = p11->alpha * w00 + p21->alpha * w10 + p12->alpha * w01 + p22->alpha * w11;

		if (outP->alpha > 0) {
			outP->red = p11->red * w00 + p21->red * w10 + p12->red * w01 + p22->red * w11;
			outP->green = p11->green * w00 + p21->green * w10 + p12->green * w01 + p22->green * w11;
			outP->blue = p11->blue * w00 + p21->blue * w10 + p12->blue * w01 + p22->blue * w11;
		}
		else {
			outP->red = 0;
			outP->green = 0;
			outP->blue = 0;
		}
	}
	else {
		*outP = *inP;
	}

	return err;
}

static void
DisposePreRenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		SqueezeInfo* infoP = reinterpret_cast<SqueezeInfo*>(pre_render_dataPV);
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

	SqueezeInfo* infoP = reinterpret_cast<SqueezeInfo*>(malloc(sizeof(SqueezeInfo)));

	if (infoP) {
		PF_ParamDef cur_param;
		ERR(PF_CHECKOUT_PARAM(in_dataP, SQUEEZE_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->strength = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, SQUEEZE_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->x_tiles = cur_param.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, SQUEEZE_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->y_tiles = cur_param.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, SQUEEZE_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mirror = cur_param.u.bd.value;

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;

		ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
			SQUEEZE_INPUT,
			SQUEEZE_INPUT,
			&req,
			in_dataP->current_time,
			in_dataP->time_step,
			in_dataP->time_scale,
			&in_result));

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
	SqueezeInfo* infoP)
{
	PF_Err err = PF_Err_NONE;

	if (!err) {
		infoP->width = input_worldP->width;
		infoP->height = input_worldP->height;
		infoP->rowbytes = input_worldP->rowbytes;
		infoP->src = input_worldP->data;

		switch (pixel_format) {
		case PF_PixelFormat_ARGB128: {
			AEFX_SuiteScoper<PF_iterateFloatSuite1> iterateFloatSuite =
				AEFX_SuiteScoper<PF_iterateFloatSuite1>(in_data,
					kPFIterateFloatSuite,
					kPFIterateFloatSuiteVersion1,
					out_data);
			iterateFloatSuite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)infoP,
				SqueezeFunc<PF_PixelFloat>,
				output_worldP);
			break;
		}

		case PF_PixelFormat_ARGB64: {
			AEFX_SuiteScoper<PF_iterate16Suite1> iterate16Suite =
				AEFX_SuiteScoper<PF_iterate16Suite1>(in_data,
					kPFIterate16Suite,
					kPFIterate16SuiteVersion1,
					out_data);
			iterate16Suite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)infoP,
				SqueezeFunc<PF_Pixel16>,
				output_worldP);
			break;
		}

		case PF_PixelFormat_ARGB32: {
			AEFX_SuiteScoper<PF_Iterate8Suite1> iterate8Suite =
				AEFX_SuiteScoper<PF_Iterate8Suite1>(in_data,
					kPFIterate8Suite,
					kPFIterate8SuiteVersion1,
					out_data);

			iterate8Suite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)infoP,
				SqueezeFunc<PF_Pixel8>,
				output_worldP);
			break;
		}

		default:
			err = PF_Err_BAD_CALLBACK_PARAM;
			break;
		}
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

typedef struct
{
	int mSrcPitch;
	int mDstPitch;
	int m16f;
	int mWidth;
	int mHeight;
	float mStrength;
	int mXTiles;
	int mYTiles;
	int mMirror;
} SqueezeParams;

size_t DivideRoundUp(
	size_t inValue,
	size_t inMultiple)
{
	return inValue ? (inValue + inMultiple - 1) / inMultiple : 0;
}

static PF_Err
SmartRenderGPU(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	SqueezeInfo* infoP)
{
	PF_Err err = PF_Err_NONE;

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

	SqueezeParams squeeze_params;

	squeeze_params.mWidth = input_worldP->width;
	squeeze_params.mHeight = input_worldP->height;
	squeeze_params.mStrength = infoP->strength;
	squeeze_params.mXTiles = infoP->x_tiles;
	squeeze_params.mYTiles = infoP->y_tiles;
	squeeze_params.mMirror = infoP->mirror;

	A_long src_row_bytes = input_worldP->rowbytes;
	A_long dst_row_bytes = output_worldP->rowbytes;

	squeeze_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
	squeeze_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
	squeeze_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint squeeze_param_index = 0;

		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(int), &squeeze_params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(int), &squeeze_params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(int), &squeeze_params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(int), &squeeze_params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(int), &squeeze_params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(float), &squeeze_params.mStrength));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(int), &squeeze_params.mXTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(int), &squeeze_params.mYTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->squeeze_kernel, squeeze_param_index++, sizeof(int), &squeeze_params.mMirror));

		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(squeeze_params.mWidth, threadBlock[0]), RoundUp(squeeze_params.mHeight, threadBlock[1]) };

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->squeeze_kernel,
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
		Squeeze_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			squeeze_params.mSrcPitch,
			squeeze_params.mDstPitch,
			squeeze_params.m16f,
			squeeze_params.mWidth,
			squeeze_params.mHeight,
			squeeze_params.mStrength,
			squeeze_params.mXTiles,
			squeeze_params.mYTiles,
			squeeze_params.mMirror);

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

		{
			DXShaderExecution shaderExecution(
				dx_gpu_data->mContext,
				dx_gpu_data->mSqueezeShader,
				3);

			DX_ERR(shaderExecution.SetParamBuffer(&squeeze_params, sizeof(SqueezeParams)));
			DX_ERR(shaderExecution.SetUnorderedAccessView(
				(ID3D12Resource*)dst_mem,
				squeeze_params.mHeight * dst_row_bytes));
			DX_ERR(shaderExecution.SetShaderResourceView(
				(ID3D12Resource*)src_mem,
				squeeze_params.mHeight * src_row_bytes));
			DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(squeeze_params.mWidth, 16), (UINT)DivideRoundUp(squeeze_params.mHeight, 16)));
		}
	}
#endif
#if HAS_METAL
	else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		Handle metal_handle = (Handle)extraP->input->gpu_data;
		MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> squeeze_param_buffer = [[device newBufferWithBytes : &squeeze_params
			length : sizeof(SqueezeParams)
			options : MTLResourceStorageModeManaged]autorelease];

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->squeeze_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = { DivideRoundUp(squeeze_params.mWidth, threadsPerGroup.width), DivideRoundUp(squeeze_params.mHeight, threadsPerGroup.height), 1 };

		[computeEncoder setComputePipelineState : metal_dataP->squeeze_pipeline] ;
		[computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
		[computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
		[computeEncoder setBuffer : squeeze_param_buffer offset : 0 atIndex : 2] ;
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

	SqueezeInfo* infoP = reinterpret_cast<SqueezeInfo*>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, SQUEEZE_INPUT, &input_worldP)));
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
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, SQUEEZE_INPUT));
	}
	else {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
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
	AEGP_SuiteHandler suites(in_dataP->pica_basicP);

	SqueezeInfo siP;
	AEFX_CLR_STRUCT(siP);

	siP.strength = params[SQUEEZE_STRENGTH]->u.fs_d.value;
	siP.x_tiles = params[SQUEEZE_X_TILES]->u.bd.value;
	siP.y_tiles = params[SQUEEZE_Y_TILES]->u.bd.value;
	siP.mirror = params[SQUEEZE_MIRROR]->u.bd.value;

	siP.width = output->width;
	siP.height = output->height;
	siP.rowbytes = params[SQUEEZE_INPUT]->u.ld.rowbytes;
	siP.src = params[SQUEEZE_INPUT]->u.ld.data;

	double bytesPerPixel = static_cast<double>(siP.rowbytes) / static_cast<double>(siP.width);

	A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

	if (bytesPerPixel >= 16.0) {   
		ERR(suites.IterateFloatSuite1()->iterate(
			in_dataP,
			0,                                
			linesL,                           
			&params[SQUEEZE_INPUT]->u.ld,     
			NULL,                                 
			static_cast<void*>(&siP),             
			SqueezeFunc<PF_PixelFloat>,        
			output));
	}
	else if (bytesPerPixel >= 8.0) {  
		ERR(suites.Iterate16Suite2()->iterate(
			in_dataP,
			0,                                
			linesL,                           
			&params[SQUEEZE_INPUT]->u.ld,     
			NULL,                                 
			static_cast<void*>(&siP),             
			SqueezeFunc<PF_Pixel16>,           
			output));
	}
	else {  
		ERR(suites.Iterate8Suite2()->iterate(
			in_dataP,
			0,                                
			linesL,                           
			&params[SQUEEZE_INPUT]->u.ld,     
			NULL,                                 
			static_cast<void*>(&siP),             
			SqueezeFunc<PF_Pixel8>,            
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
		"Squeeze",  
		"DKT Squeeze",   
		"DKT Effects",  
		AE_RESERVED_INFO,   
		"EffectMain",	  
		"");	   

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
			err = PreRender(in_data,
				out_data,
				(PF_PreRenderExtra*)extra);
			break;

		case PF_Cmd_SMART_RENDER:
			err = SmartRender(in_data,
				out_data,
				(PF_SmartRenderExtra*)extra,
				false);
			break;

		case PF_Cmd_SMART_RENDER_GPU:
			err = SmartRender(in_data,
				out_data,
				(PF_SmartRenderExtra*)extra,
				true);
			break;

		case PF_Cmd_GPU_DEVICE_SETUP:
			err = GPUDeviceSetup(in_data,
				out_data,
				(PF_GPUDeviceSetupExtra*)extra);
			break;

		case PF_Cmd_GPU_DEVICE_SETDOWN:
			err = GPUDeviceSetdown(in_data,
				out_data,
				(PF_GPUDeviceSetdownExtra*)extra);
			break;

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