#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "LensBlur.h"
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef CLAMP
#define CLAMP(value, min_val, max_val) ((value) < (min_val) ? (min_val) : ((value) > (max_val) ? (max_val) : (value)))
#endif

#ifndef MIN
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#endif

#ifndef MAX
#define MAX(A, B) ((A) > (B) ? (A) : (B))
#endif

inline PF_Err CL2Err(cl_int cl_result) {
	if (cl_result == CL_SUCCESS) {
		return PF_Err_NONE;
	}
	else {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}
}

#define CL_ERR(FUNC) ERR(CL2Err(FUNC))

extern void LensBlur_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float centerX,
	float centerY,
	float strength,
	float radius);

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
	PF_Err      err = PF_Err_NONE;
	PF_ParamDef	def;

	AEFX_CLR_STRUCT(def);
	PF_ADD_POINT(STR_CENTER_PARAM,
		50,
		50,
		0,
		CENTER_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(STR_STRENGTH_PARAM,
		STRENGTH_MIN_VALUE,
		STRENGTH_MAX_VALUE,
		STRENGTH_MIN_SLIDER,
		STRENGTH_MAX_SLIDER,
		STRENGTH_DFLT,
		PF_Precision_THOUSANDTHS,
		0,
		0,
		STRENGTH_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX(STR_RADIUS_PARAM,
		RADIUS_MIN_VALUE,
		RADIUS_MAX_VALUE,
		RADIUS_MIN_SLIDER,
		RADIUS_MAX_SLIDER,
		RADIUS_DFLT,
		PF_Precision_THOUSANDTHS,
		0,
		0,
		RADIUS_DISK_ID);

	out_data->num_params = LENSBLUR_NUM_PARAMS;

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
	cl_kernel lensblur_kernel;
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
	ShaderObjectPtr mLensBlurShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState>lensblur_pipeline;
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

		size_t sizes[] = { strlen(k16fString), strlen(kLensBlurKernel_OpenCLString) };
		char const* strings[] = { k16fString, kLensBlurKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->lensblur_kernel = clCreateKernel(program, "LensBlurKernel", &result);
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
		dx_gpu_data->mLensBlurShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"LensBlurKernel", csoPath, sigPath));

		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mLensBlurShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		NSString* source = [NSString stringWithCString : kSDK_Invert_ProcAmp_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
			id<MTLFunction> lensblur_function = nil;
			NSString* lensblur_name = [NSString stringWithCString : "LensBlurKernel" encoding : NSUTF8StringEncoding];

			lensblur_function = [[library newFunctionWithName : lensblur_name]autorelease];

			if (!lensblur_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->lensblur_pipeline = [device newComputePipelineStateWithFunction : lensblur_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->lensblur_kernel);

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
		dx_gpu_dataP->mLensBlurShader.reset();

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

	return err;
}


inline float length(float x, float y) {
	return sqrt(x * x + y * y);
}

inline void normalize(float& x, float& y) {
	float len = length(x, y);
	if (len > 0.0001f) {
		x /= len;
		y /= len;
	}
}

template<typename PixelT>
PixelT SampleBilinear(
	PixelT* srcData,
	float x,
	float y,
	int width,
	int height)
{
	x = CLAMP(x, 0.0f, static_cast<float>(width - 1));
	y = CLAMP(y, 0.0f, static_cast<float>(height - 1));

	int x0 = static_cast<int>(x);
	int y0 = static_cast<int>(y);
	int x1 = MIN(x0 + 1, width - 1);
	int y1 = MIN(y0 + 1, height - 1);

	float fx = x - static_cast<float>(x0);
	float fy = y - static_cast<float>(y0);

	PixelT p00 = srcData[y0 * width + x0];
	PixelT p10 = srcData[y0 * width + x1];
	PixelT p01 = srcData[y1 * width + x0];
	PixelT p11 = srcData[y1 * width + x1];

	PixelT result;

	if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
		result.alpha = static_cast<A_u_char>(
			(1.0f - fx) * (1.0f - fy) * p00.alpha +
			fx * (1.0f - fy) * p10.alpha +
			(1.0f - fx) * fy * p01.alpha +
			fx * fy * p11.alpha + 0.5f);

		result.red = static_cast<A_u_char>(
			(1.0f - fx) * (1.0f - fy) * p00.red +
			fx * (1.0f - fy) * p10.red +
			(1.0f - fx) * fy * p01.red +
			fx * fy * p11.red + 0.5f);

		result.green = static_cast<A_u_char>(
			(1.0f - fx) * (1.0f - fy) * p00.green +
			fx * (1.0f - fy) * p10.green +
			(1.0f - fx) * fy * p01.green +
			fx * fy * p11.green + 0.5f);

		result.blue = static_cast<A_u_char>(
			(1.0f - fx) * (1.0f - fy) * p00.blue +
			fx * (1.0f - fy) * p10.blue +
			(1.0f - fx) * fy * p01.blue +
			fx * fy * p11.blue + 0.5f);
	}
	else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
		result.alpha = static_cast<A_u_short>(
			(1.0f - fx) * (1.0f - fy) * p00.alpha +
			fx * (1.0f - fy) * p10.alpha +
			(1.0f - fx) * fy * p01.alpha +
			fx * fy * p11.alpha + 0.5f);

		result.red = static_cast<A_u_short>(
			(1.0f - fx) * (1.0f - fy) * p00.red +
			fx * (1.0f - fy) * p10.red +
			(1.0f - fx) * fy * p01.red +
			fx * fy * p11.red + 0.5f);

		result.green = static_cast<A_u_short>(
			(1.0f - fx) * (1.0f - fy) * p00.green +
			fx * (1.0f - fy) * p10.green +
			(1.0f - fx) * fy * p01.green +
			fx * fy * p11.green + 0.5f);

		result.blue = static_cast<A_u_short>(
			(1.0f - fx) * (1.0f - fy) * p00.blue +
			fx * (1.0f - fy) * p10.blue +
			(1.0f - fx) * fy * p01.blue +
			fx * fy * p11.blue + 0.5f);
	}
	else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
		result.alpha =
			(1.0f - fx) * (1.0f - fy) * p00.alpha +
			fx * (1.0f - fy) * p10.alpha +
			(1.0f - fx) * fy * p01.alpha +
			fx * fy * p11.alpha;

		result.red =
			(1.0f - fx) * (1.0f - fy) * p00.red +
			fx * (1.0f - fy) * p10.red +
			(1.0f - fx) * fy * p01.red +
			fx * fy * p11.red;

		result.green =
			(1.0f - fx) * (1.0f - fy) * p00.green +
			fx * (1.0f - fy) * p10.green +
			(1.0f - fx) * fy * p01.green +
			fx * fy * p11.green;

		result.blue =
			(1.0f - fx) * (1.0f - fy) * p00.blue +
			fx * (1.0f - fy) * p10.blue +
			(1.0f - fx) * fy * p01.blue +
			fx * fy * p11.blue;
	}

	return result;
}

template<typename PixelT>
static PF_Err
LensBlurFunc(
	void* refcon,
	A_long      xL,
	A_long      yL,
	PixelT* inP,
	PixelT* outP)
{
	PF_Err err = PF_Err_NONE;
	BlurInfo* biP = reinterpret_cast<BlurInfo*>(refcon);

	if (!biP) return PF_Err_BAD_CALLBACK_PARAM;

	float texelSizeX = 1.0f / static_cast<float>(biP->width);
	float texelSizeY = 1.0f / static_cast<float>(biP->height);

	float width = static_cast<float>(biP->width);
	float height = static_cast<float>(biP->height);

	float x = static_cast<float>(xL) / static_cast<float>(biP->width);
	float y = static_cast<float>(yL) / static_cast<float>(biP->height);

	float centerX = biP->centerX / width;
	float centerY = biP->centerY / height;

	float vx = x - centerX;
	float vy = y - centerY;

	float dist = length(vx, vy);

	float blurStrength = 0.0f;
	if (dist <= biP->radiusF) {
		blurStrength = 0.0f;
	}
	else if (dist >= 1.0f) {
		blurStrength = 1.0f;
	}
	else {
		blurStrength = (dist - biP->radiusF) / (1.0f - biP->radiusF);
		blurStrength = blurStrength * blurStrength * (3.0f - 2.0f * blurStrength);
	}

	float speed = biP->strengthF / 2.0f;
	speed /= texelSizeX;
	speed *= blurStrength;

	int nSamples = static_cast<int>(CLAMP(speed, 1.01f, 100.01f));

	if (nSamples <= 1) {
		*outP = *inP;
		return err;
	}

	float len = sqrt(vx * vx + vy * vy);
	if (len > 0.0001f) {
		vx /= len;
		vy /= len;
	}

	vx *= texelSizeX * speed;
	vy *= texelSizeY * speed;

	PixelT* baseP = inP - (yL * biP->width + xL);

	float accum_a = 0.0f;
	float accum_r = 0.0f;
	float accum_g = 0.0f;
	float accum_b = 0.0f;

	for (int i = 0; i < nSamples; i++) {
		float t = static_cast<float>(i) / static_cast<float>(nSamples - 1) - 0.5f;

		float sampleX = static_cast<float>(xL) - vx * biP->width * t;
		float sampleY = static_cast<float>(yL) - vy * biP->height * t;

		PixelT sample = SampleBilinear(baseP, sampleX, sampleY, biP->width, biP->height);

		if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
			accum_a += static_cast<float>(sample.alpha);
			accum_r += static_cast<float>(sample.red);
			accum_g += static_cast<float>(sample.green);
			accum_b += static_cast<float>(sample.blue);
		}
		else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
			accum_a += static_cast<float>(sample.alpha);
			accum_r += static_cast<float>(sample.red);
			accum_g += static_cast<float>(sample.green);
			accum_b += static_cast<float>(sample.blue);
		}
		else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
			accum_a += sample.alpha;
			accum_r += sample.red;
			accum_g += sample.green;
			accum_b += sample.blue;
		}
	}

	accum_a /= static_cast<float>(nSamples);
	accum_r /= static_cast<float>(nSamples);
	accum_g /= static_cast<float>(nSamples);
	accum_b /= static_cast<float>(nSamples);

	if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
		outP->alpha = static_cast<A_u_char>(CLAMP(accum_a + 0.5f, 0.0f, 255.0f));
		outP->red = static_cast<A_u_char>(CLAMP(accum_r + 0.5f, 0.0f, 255.0f));
		outP->green = static_cast<A_u_char>(CLAMP(accum_g + 0.5f, 0.0f, 255.0f));
		outP->blue = static_cast<A_u_char>(CLAMP(accum_b + 0.5f, 0.0f, 255.0f));
	}
	else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
		outP->alpha = static_cast<A_u_short>(CLAMP(accum_a + 0.5f, 0.0f, 32767.0f));
		outP->red = static_cast<A_u_short>(CLAMP(accum_r + 0.5f, 0.0f, 32767.0f));
		outP->green = static_cast<A_u_short>(CLAMP(accum_g + 0.5f, 0.0f, 32767.0f));
		outP->blue = static_cast<A_u_short>(CLAMP(accum_b + 0.5f, 0.0f, 32767.0f));
	}
	else if constexpr (std::is_same_v<PixelT, PF_PixelFloat>) {
		outP->alpha = CLAMP(accum_a, 0.0f, 1.0f);
		outP->red = CLAMP(accum_r, 0.0f, 1.0f);
		outP->green = CLAMP(accum_g, 0.0f, 1.0f);
		outP->blue = CLAMP(accum_b, 0.0f, 1.0f);
	}

	return err;
}



static void
DisposePreRenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		BlurInfo* infoP = reinterpret_cast<BlurInfo*>(pre_render_dataPV);
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

	BlurInfo* infoP = reinterpret_cast<BlurInfo*>(malloc(sizeof(BlurInfo)));

	if (infoP) {
		PF_ParamDef cur_param;
		ERR(PF_CHECKOUT_PARAM(in_dataP, LENSBLUR_CENTER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->centerX = static_cast<float>(cur_param.u.td.x_value) / 65536.0f;
		infoP->centerY = static_cast<float>(cur_param.u.td.y_value) / 65536.0f;

		ERR(PF_CHECKOUT_PARAM(in_dataP, LENSBLUR_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->strengthF = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, LENSBLUR_RADIUS, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radiusF = cur_param.u.fs_d.value;

		infoP->width = 0;
		infoP->height = 0;

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;

		ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
			LENSBLUR_INPUT,
			LENSBLUR_INPUT,
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

typedef struct
{
	int mSrcPitch;
	int mDstPitch;
	int m16f;
	int mWidth;
	int mHeight;
	float mCenterX;
	float mCenterY;
	float mStrength;
	float mRadius;
} LensBlurParams;

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
	BlurInfo* infoP)
{
	PF_Err err = PF_Err_NONE;

	infoP->width = input_worldP->width;
	infoP->height = input_worldP->height;

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
			LensBlurFunc<PF_PixelFloat>,
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
			LensBlurFunc<PF_Pixel16>,
			output_worldP);
		break;
	}

	case PF_PixelFormat_ARGB32:
	default: {
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
			LensBlurFunc<PF_Pixel8>,
			output_worldP);
		break;
	}
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
	BlurInfo* infoP)
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

	LensBlurParams params;
	params.mWidth = input_worldP->width;
	params.mHeight = input_worldP->height;
	params.mCenterX = infoP->centerX / (float)input_worldP->width;
	params.mCenterY = infoP->centerY / (float)input_worldP->height;
	params.mStrength = infoP->strengthF;
	params.mRadius = infoP->radiusF;

	A_long src_row_bytes = input_worldP->rowbytes;
	A_long dst_row_bytes = output_worldP->rowbytes;

	params.mSrcPitch = src_row_bytes / bytes_per_pixel;
	params.mDstPitch = dst_row_bytes / bytes_per_pixel;
	params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(int), &params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(int), &params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(int), &params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(int), &params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(int), &params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(float), &params.mCenterX));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(float), &params.mCenterY));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(float), &params.mStrength));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->lensblur_kernel, param_index++, sizeof(float), &params.mRadius));

		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->lensblur_kernel,
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
		LensBlur_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			params.mSrcPitch,
			params.mDstPitch,
			params.m16f,
			params.mWidth,
			params.mHeight,
			params.mCenterX,
			params.mCenterY,
			params.mStrength,
			params.mRadius);

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
			dx_gpu_data->mLensBlurShader,
			3);

		DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(LensBlurParams)));
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
			length : sizeof(LensBlurParams)
			options : MTLResourceStorageModeManaged]autorelease];

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->lensblur_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = { DivideRoundUp(params.mWidth, threadsPerGroup.width), DivideRoundUp(params.mHeight, threadsPerGroup.height), 1 };

		[computeEncoder setComputePipelineState : metal_dataP->lensblur_pipeline] ;
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
	PF_Err err = PF_Err_NONE;

	PF_EffectWorld* input_worldP = NULL;
	PF_EffectWorld* output_worldP = NULL;

	BlurInfo* infoP = reinterpret_cast<BlurInfo*>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, LENSBLUR_INPUT, &input_worldP)));
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
		ERR(extraP->cb->checkin_layer_pixels(in_data->effect_ref, LENSBLUR_INPUT));
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
		"Lens Blur",
		"DKT Lens Blur",
		"DKT Effects",
		AE_RESERVED_INFO,
		"EffectMain",
		"https://www.adobe.com");

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