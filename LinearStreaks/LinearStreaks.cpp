#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#define NOMINMAX
#include "LinearStreaks.h"
#include <iostream>
#include <algorithm>

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

extern void Linear_Streaks_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float strength,
	float angle,
	float alpha,
	float bias,
	int rMode,
	int gMode,
	int bMode,
	int aMode);

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

	PF_ADD_FLOAT_SLIDERX(
		"Strength",
		0.0f,
		1.0f,
		0.0f,
		1.0f,
		0.15f,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		STRENGTH_DISK_ID);

	AEFX_CLR_STRUCT(def);


	PF_ADD_ANGLE(
		"Angle",
		0,          
		ANGLE_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX(
		"Alpha",
		0.0f,
		1.0f,
		0.0f,
		1.0f,
		1.0f,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		ALPHA_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_FLOAT_SLIDERX(
		"Bias",
		-1.0f,
		1.0f,
		-1.0f,
		1.0f,
		0.0f,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		BIAS_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_POPUP(
		"R",
		4,                         
		2,                         
		"MIN|MAX|AVG|OFF",            
		R_MODE_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_POPUP(
		"G",
		4,                         
		2,                         
		"MIN|MAX|AVG|OFF",            
		G_MODE_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_POPUP(
		"B",
		4,                         
		2,                         
		"MIN|MAX|AVG|OFF",            
		B_MODE_DISK_ID);

	AEFX_CLR_STRUCT(def);

	PF_ADD_POPUP(
		"A",
		4,                         
		2,                         
		"MIN|MAX|AVG|OFF",            
		A_MODE_DISK_ID);

	out_data->num_params = LINEAR_STREAKS_NUM_PARAMS;

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
	cl_kernel linear_streaks_kernel;
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
	ShaderObjectPtr mLinearStreaksShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState> linear_streaks_pipeline;
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

		size_t sizes[] = { strlen(k16fString), strlen(kLinearStreaksKernel_OpenCLString) };
		char const* strings[] = { k16fString, kLinearStreaksKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->linear_streaks_kernel = clCreateKernel(program, "LinearStreaksKernel", &result);
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
		dx_gpu_data->mLinearStreaksShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"LinearStreaksKernel", csoPath, sigPath));

		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mLinearStreaksShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		NSString* source = [NSString stringWithCString : kLinearStreaksKernel_MetalString encoding : NSUTF8StringEncoding];
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
			id<MTLFunction> linear_streaks_function = nil;
			NSString* linear_streaks_name = [NSString stringWithCString : "LinearStreaksKernel" encoding : NSUTF8StringEncoding];

			linear_streaks_function = [[library newFunctionWithName : linear_streaks_name]autorelease];

			if (!linear_streaks_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->linear_streaks_pipeline = [device newComputePipelineStateWithFunction : linear_streaks_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->linear_streaks_kernel);

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
		dx_gpu_dataP->mLinearStreaksShader.reset();

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
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*gpu_dataH);

		[metal_dataP->linear_streaks_pipeline release] ;

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

	return err;
}

template<typename PixelT>
static PF_Err
LinearStreaksFunc(
	void* refcon,
	A_long xL,
	A_long yL,
	PixelT* inP,
	PixelT* outP)
{
	PF_Err err = PF_Err_NONE;
	LinearStreaksInfo* lsP = reinterpret_cast<LinearStreaksInfo*>(refcon);

	if (!lsP) return PF_Err_BAD_CALLBACK_PARAM;

	float texelSizeX = 1.0f / lsP->inputWidth;
	float texelSizeY = 1.0f / lsP->inputHeight;

	float rad = lsP->angle * 0.0174533f;

	float velocityX = cos(rad) * lsP->strength;
	float velocityY = -sin(rad) * lsP->strength;

	velocityX *= lsP->outputHeight / (float)lsP->outputWidth;

	float speed = sqrt(velocityX * velocityX / (texelSizeX * texelSizeX) +
		velocityY * velocityY / (texelSizeY * texelSizeY));

	int nSamples = std::min(std::max((int)speed, 1), 100);

	vec4 originalColor;
	if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
		originalColor.r = inP->red / 255.0f;
		originalColor.g = inP->green / 255.0f;
		originalColor.b = inP->blue / 255.0f;
		originalColor.a = inP->alpha / 255.0f;
	}
	else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
		originalColor.r = inP->red / 32768.0f;
		originalColor.g = inP->green / 32768.0f;
		originalColor.b = inP->blue / 32768.0f;
		originalColor.a = inP->alpha / 32768.0f;
	}
	else {  
		originalColor.r = inP->red;
		originalColor.g = inP->green;
		originalColor.b = inP->blue;
		originalColor.a = inP->alpha;
	}

	vec4 outColor = originalColor;

	if (outColor.a > 0.0f) {
		outColor.r /= outColor.a;
		outColor.g /= outColor.a;
		outColor.b /= outColor.a;
	}

	for (int i = 1; i < nSamples; i++) {
		float t = (float)i / (float)(nSamples - 1) - (0.5f + lsP->bias / 2.0f);
		float offsetX = velocityX * t;
		float offsetY = velocityY * t;

		float sampleX = (xL + 0.5f) / lsP->inputWidth - offsetX;
		float sampleY = (yL + 0.5f) / lsP->inputHeight - offsetY;

		bool insideTexture = (sampleX >= 0.0f && sampleX <= 1.0f &&
			sampleY >= 0.0f && sampleY <= 1.0f);

		vec4 c = { 0, 0, 0, 0 };
		if (insideTexture) {
			int pixX = (int)(sampleX * lsP->inputWidth);
			int pixY = (int)(sampleY * lsP->inputHeight);

			pixX = std::min(std::max(pixX, 0), lsP->inputWidth - 1);
			pixY = std::min(std::max(pixY, 0), lsP->inputHeight - 1);

			PixelT* samplePixel = GetPixelAddress<PixelT>(lsP->inputP, pixX, pixY, lsP->inputRowBytes);

			if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
				c.r = samplePixel->red / 255.0f;
				c.g = samplePixel->green / 255.0f;
				c.b = samplePixel->blue / 255.0f;
				c.a = samplePixel->alpha / 255.0f;
			}
			else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
				c.r = samplePixel->red / 32768.0f;
				c.g = samplePixel->green / 32768.0f;
				c.b = samplePixel->blue / 32768.0f;
				c.a = samplePixel->alpha / 32768.0f;
			}
			else {  
				c.r = samplePixel->red;
				c.g = samplePixel->green;
				c.b = samplePixel->blue;
				c.a = samplePixel->alpha;
			}

			if (c.a > 0.0f) {
				c.r /= c.a;
				c.g /= c.a;
				c.b /= c.a;
			}
		}

		float r, g, b, a;

		switch (lsP->rmode) {
		case 0:  
			r = std::min(outColor.r, c.r);
			break;
		case 1:  
			r = std::max(outColor.r, c.r);
			break;
		case 3:  
			r = outColor.r + c.r;
			break;
		default:  
			r = outColor.r;
			break;
		}

		switch (lsP->gmode) {
		case 0:  
			g = std::min(outColor.g, c.g);
			break;
		case 1:  
			g = std::max(outColor.g, c.g);
			break;
		case 3:  
			g = outColor.g + c.g;
			break;
		default:  
			g = outColor.g;
			break;
		}

		switch (lsP->bmode) {
		case 0:  
			b = std::min(outColor.b, c.b);
			break;
		case 1:  
			b = std::max(outColor.b, c.b);
			break;
		case 3:  
			b = outColor.b + c.b;
			break;
		default:  
			b = outColor.b;
			break;
		}

		switch (lsP->amode) {
		case 0:  
			a = std::min(outColor.a, c.a);
			break;
		case 1:  
			a = std::max(outColor.a, c.a);
			break;
		case 3:  
			a = outColor.a + c.a;
			break;
		default:  
			a = outColor.a;
			break;
		}

		outColor.r = r;
		outColor.g = g;
		outColor.b = b;
		outColor.a = a;
	}

	outColor.r /= (lsP->rmode == 3) ? (float)nSamples : 1.0f;
	outColor.g /= (lsP->gmode == 3) ? (float)nSamples : 1.0f;
	outColor.b /= (lsP->bmode == 3) ? (float)nSamples : 1.0f;
	outColor.a /= (lsP->amode == 3) ? (float)nSamples : 1.0f;

	outColor.r *= outColor.a;
	outColor.g *= outColor.a;
	outColor.b *= outColor.a;

	vec4 finalColor;
	finalColor.r = originalColor.r * (1.0f - lsP->alpha) + outColor.r * lsP->alpha;
	finalColor.g = originalColor.g * (1.0f - lsP->alpha) + outColor.g * lsP->alpha;
	finalColor.b = originalColor.b * (1.0f - lsP->alpha) + outColor.b * lsP->alpha;
	finalColor.a = originalColor.a * (1.0f - lsP->alpha) + outColor.a * lsP->alpha;

	if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
		outP->red = (A_u_char)std::min(std::max((int)(finalColor.r * 255.0f + 0.5f), 0), 255);
		outP->green = (A_u_char)std::min(std::max((int)(finalColor.g * 255.0f + 0.5f), 0), 255);
		outP->blue = (A_u_char)std::min(std::max((int)(finalColor.b * 255.0f + 0.5f), 0), 255);
		outP->alpha = (A_u_char)std::min(std::max((int)(finalColor.a * 255.0f + 0.5f), 0), 255);
	}
	else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
		outP->red = (A_u_short)std::min(std::max((int)(finalColor.r * 32768.0f + 0.5f), 0), 32768);
		outP->green = (A_u_short)std::min(std::max((int)(finalColor.g * 32768.0f + 0.5f), 0), 32768);
		outP->blue = (A_u_short)std::min(std::max((int)(finalColor.b * 32768.0f + 0.5f), 0), 32768);
		outP->alpha = (A_u_short)std::min(std::max((int)(finalColor.a * 32768.0f + 0.5f), 0), 32768);
	}
	else {  
		outP->red = finalColor.r;
		outP->green = finalColor.g;
		outP->blue = finalColor.b;
		outP->alpha = finalColor.a;
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

	LinearStreaksInfo lsInfo;
	AEFX_CLR_STRUCT(lsInfo);
	A_long linesL = 0;

	linesL = output->extent_hint.bottom - output->extent_hint.top;

	lsInfo.strength = params[LINEAR_STREAKS_STRENGTH]->u.fs_d.value;
	lsInfo.angle = -params[LINEAR_STREAKS_ANGLE]->u.fs_d.value;
	lsInfo.alpha = params[LINEAR_STREAKS_ALPHA]->u.fs_d.value;
	lsInfo.bias = params[LINEAR_STREAKS_BIAS]->u.fs_d.value;
	lsInfo.rmode = params[LINEAR_STREAKS_R_MODE]->u.pd.value - 1;
	lsInfo.gmode = params[LINEAR_STREAKS_G_MODE]->u.pd.value - 1;
	lsInfo.bmode = params[LINEAR_STREAKS_B_MODE]->u.pd.value - 1;
	lsInfo.amode = params[LINEAR_STREAKS_A_MODE]->u.pd.value - 1;

	lsInfo.inputP = params[LINEAR_STREAKS_INPUT]->u.ld.data;
	lsInfo.inputWidth = params[LINEAR_STREAKS_INPUT]->u.ld.width;
	lsInfo.inputHeight = params[LINEAR_STREAKS_INPUT]->u.ld.height;
	lsInfo.inputRowBytes = params[LINEAR_STREAKS_INPUT]->u.ld.rowbytes;
	lsInfo.outputWidth = output->width;
	lsInfo.outputHeight = output->height;

	AEFX_SuiteScoper<PF_WorldSuite2> world_suite =
		AEFX_SuiteScoper<PF_WorldSuite2>(in_dataP, kPFWorldSuite, kPFWorldSuiteVersion2, out_dataP);

	PF_PixelFormat pixel_format;
	ERR(world_suite->PF_GetPixelFormat(output, &pixel_format));

	AEFX_SuiteScoper<PF_iterateFloatSuite2> float_suite =
		AEFX_SuiteScoper<PF_iterateFloatSuite2>(in_dataP, kPFIterateFloatSuite, kPFIterateFloatSuiteVersion2, out_dataP);

	AEFX_SuiteScoper<PF_iterate16Suite2> iterate16_suite =
		AEFX_SuiteScoper<PF_iterate16Suite2>(in_dataP, kPFIterate16Suite, kPFIterate16SuiteVersion2, out_dataP);

	AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8_suite =
		AEFX_SuiteScoper<PF_Iterate8Suite2>(in_dataP, kPFIterate8Suite, kPFIterate8SuiteVersion2, out_dataP);

	switch (pixel_format) {
	case PF_PixelFormat_ARGB128:
		ERR(float_suite->iterate(
			in_dataP,
			0,                            
			linesL,                       
			&params[LINEAR_STREAKS_INPUT]->u.ld,  
			NULL,                             
			(void*)&lsInfo,                   
			LinearStreaksFunc<PF_PixelFloat>,     
			output));
		break;

	case PF_PixelFormat_ARGB64:
		ERR(iterate16_suite->iterate(
			in_dataP,
			0,                            
			linesL,                       
			&params[LINEAR_STREAKS_INPUT]->u.ld,  
			NULL,                             
			(void*)&lsInfo,                   
			LinearStreaksFunc<PF_Pixel16>,        
			output));
		break;

	case PF_PixelFormat_ARGB32:
	default:
		ERR(iterate8_suite->iterate(
			in_dataP,
			0,                            
			linesL,                       
			&params[LINEAR_STREAKS_INPUT]->u.ld,  
			NULL,                             
			(void*)&lsInfo,                   
			LinearStreaksFunc<PF_Pixel8>,         
			output));
		break;
	}

	return err;
}

static void
DisposePreRenderData(void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		LinearStreaksInfo* infoP = reinterpret_cast<LinearStreaksInfo*>(pre_render_dataPV);
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

	LinearStreaksInfo* infoP = reinterpret_cast<LinearStreaksInfo*>(malloc(sizeof(LinearStreaksInfo)));

	if (infoP) {
		PF_ParamDef strength_param, angle_param, alpha_param, bias_param;
		PF_ParamDef rmode_param, gmode_param, bmode_param, amode_param;

		AEFX_CLR_STRUCT(strength_param);
		AEFX_CLR_STRUCT(angle_param);
		AEFX_CLR_STRUCT(alpha_param);
		AEFX_CLR_STRUCT(bias_param);
		AEFX_CLR_STRUCT(rmode_param);
		AEFX_CLR_STRUCT(gmode_param);
		AEFX_CLR_STRUCT(bmode_param);
		AEFX_CLR_STRUCT(amode_param);

		ERR(PF_CHECKOUT_PARAM(in_dataP, LINEAR_STREAKS_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &strength_param));
		ERR(PF_CHECKOUT_PARAM(in_dataP, LINEAR_STREAKS_ANGLE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &angle_param));
		ERR(PF_CHECKOUT_PARAM(in_dataP, LINEAR_STREAKS_ALPHA, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &alpha_param));
		ERR(PF_CHECKOUT_PARAM(in_dataP, LINEAR_STREAKS_BIAS, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &bias_param));
		ERR(PF_CHECKOUT_PARAM(in_dataP, LINEAR_STREAKS_R_MODE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &rmode_param));
		ERR(PF_CHECKOUT_PARAM(in_dataP, LINEAR_STREAKS_G_MODE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &gmode_param));
		ERR(PF_CHECKOUT_PARAM(in_dataP, LINEAR_STREAKS_B_MODE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &bmode_param));
		ERR(PF_CHECKOUT_PARAM(in_dataP, LINEAR_STREAKS_A_MODE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &amode_param));

		infoP->strength = strength_param.u.fs_d.value;
		infoP->angle = -angle_param.u.ad.value / 65536.0f;
		infoP->alpha = alpha_param.u.fs_d.value;
		infoP->bias = bias_param.u.fs_d.value;
		infoP->rmode = rmode_param.u.pd.value - 1;
		infoP->gmode = gmode_param.u.pd.value - 1;
		infoP->bmode = bmode_param.u.pd.value - 1;
		infoP->amode = amode_param.u.pd.value - 1;

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;

		PF_CHECKIN_PARAM(in_dataP, &strength_param);
		PF_CHECKIN_PARAM(in_dataP, &angle_param);
		PF_CHECKIN_PARAM(in_dataP, &alpha_param);
		PF_CHECKIN_PARAM(in_dataP, &bias_param);
		PF_CHECKIN_PARAM(in_dataP, &rmode_param);
		PF_CHECKIN_PARAM(in_dataP, &gmode_param);
		PF_CHECKIN_PARAM(in_dataP, &bmode_param);
		PF_CHECKIN_PARAM(in_dataP, &amode_param);

		ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
			LINEAR_STREAKS_INPUT,
			LINEAR_STREAKS_INPUT,
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
	LinearStreaksInfo* infoP)
{
	PF_Err err = PF_Err_NONE;

	infoP->inputP = input_worldP->data;
	infoP->inputWidth = input_worldP->width;
	infoP->inputHeight = input_worldP->height;
	infoP->inputRowBytes = input_worldP->rowbytes;
	infoP->outputWidth = output_worldP->width;
	infoP->outputHeight = output_worldP->height;

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
			LinearStreaksFunc<PF_PixelFloat>,
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
			LinearStreaksFunc<PF_Pixel16>,
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
			LinearStreaksFunc<PF_Pixel8>,
			output_worldP);
		break;
	}

	default:
		err = PF_Err_BAD_CALLBACK_PARAM;
		break;
	}

	return err;
}

static size_t
RoundUp(size_t inValue, size_t inMultiple)
{
	return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}

size_t DivideRoundUp(size_t inValue, size_t inMultiple)
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
	float mStrength;
	float mAngle;
	float mAlpha;
	float mBias;
	int mRMode;
	int mGMode;
	int mBMode;
	int mAMode;
} LinearStreaksParams;

static PF_Err
SmartRenderGPU(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	LinearStreaksInfo* infoP)
{
	PF_Err err = PF_Err_NONE;

	AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite =
		AEFX_SuiteScoper<PF_GPUDeviceSuite1>(in_dataP, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_dataP);

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

	LinearStreaksParams params;
	params.mWidth = input_worldP->width;
	params.mHeight = input_worldP->height;
	params.mSrcPitch = input_worldP->rowbytes / bytes_per_pixel;
	params.mDstPitch = output_worldP->rowbytes / bytes_per_pixel;
	params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);
	params.mStrength = infoP->strength;
	params.mAngle = infoP->angle;
	params.mAlpha = infoP->alpha;
	params.mBias = infoP->bias;
	params.mRMode = infoP->rmode;
	params.mGMode = infoP->gmode;
	params.mBMode = infoP->bmode;
	params.mAMode = infoP->amode;

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(float), &params.mStrength));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(float), &params.mAngle));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(float), &params.mAlpha));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(float), &params.mBias));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.mRMode));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.mGMode));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.mBMode));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->linear_streaks_kernel, param_index++, sizeof(int), &params.mAMode));

		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->linear_streaks_kernel,
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
		Linear_Streaks_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			params.mSrcPitch,
			params.mDstPitch,
			params.m16f,
			params.mWidth,
			params.mHeight,
			params.mStrength,
			params.mAngle,
			params.mAlpha,
			params.mBias,
			params.mRMode,
			params.mGMode,
			params.mBMode,
			params.mAMode);

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
			dx_gpu_data->mLinearStreaksShader,
			3);

		DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(LinearStreaksParams)));
		DX_ERR(shaderExecution.SetUnorderedAccessView(
			(ID3D12Resource*)dst_mem,
			params.mHeight * output_worldP->rowbytes));
		DX_ERR(shaderExecution.SetShaderResourceView(
			(ID3D12Resource*)src_mem,
			params.mHeight * input_worldP->rowbytes));
		DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(params.mWidth, 16), (UINT)DivideRoundUp(params.mHeight, 16)));
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
			length : sizeof(LinearStreaksParams)
			options : MTLResourceStorageModeManaged]autorelease];

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->linear_streaks_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = { DivideRoundUp(params.mWidth, threadsPerGroup.width),
									DivideRoundUp(params.mHeight, threadsPerGroup.height), 1 };

		[computeEncoder setComputePipelineState : metal_dataP->linear_streaks_pipeline] ;
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
	PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
	PF_EffectWorld* input_worldP = NULL, * output_worldP = NULL;

	LinearStreaksInfo* infoP = reinterpret_cast<LinearStreaksInfo*>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, LINEAR_STREAKS_INPUT, &input_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite =
			AEFX_SuiteScoper<PF_WorldSuite2>(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);

		PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		if (isGPU) {
			ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
		}
		else {
			ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
		}

		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, LINEAR_STREAKS_INPUT));
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
		"Linear Streaks",           
		"DKT Linear Streaks",        
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
