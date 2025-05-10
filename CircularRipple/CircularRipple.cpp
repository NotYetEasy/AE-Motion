#if HAS_CUDA
#include <cuda_runtime.h>

#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "CircularRipple.h"
#include <iostream>

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
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


extern void CircularRipple_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float centerX,
	float centerY,
	float frequency,
	float strength,
	float phase,
	float radius,
	float feather,
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
		"Circular Ripple\r"
		"Created by DKT with Unknown's help.\r"
		"Under development!!\r"
		"Discord: dkt0 and unknown1234\r"
		"Contact us if you want to contribute or report bugs!");
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
	PF_ParamDef def;

	AEFX_CLR_STRUCT(def);

	PF_ADD_POINT("Center",
		0,
		0,
		0,
		CENTER_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Frequency",
		CIRCULAR_RIPPLE_FREQ_MIN,           
		CIRCULAR_RIPPLE_FREQ_MAX,           
		CIRCULAR_RIPPLE_FREQ_MIN,            
		CIRCULAR_RIPPLE_FREQ_MAX,            
		CIRCULAR_RIPPLE_FREQ_DFLT,          
		PF_Precision_HUNDREDTHS,
		0,
		0,
		FREQUENCY_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Strength",
		CIRCULAR_RIPPLE_STRENGTH_MIN,       
		CIRCULAR_RIPPLE_STRENGTH_MAX,       
		CIRCULAR_RIPPLE_STRENGTH_MIN,        
		CIRCULAR_RIPPLE_STRENGTH_MAX,        
		CIRCULAR_RIPPLE_STRENGTH_DFLT,      
		PF_Precision_THOUSANDTHS,
		0,
		0,
		STRENGTH_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Phase",
		-1000,          
		1000,          
		-1,           
		1,           
		0,         
		PF_Precision_HUNDREDTHS,
		0,
		0,
		PHASE_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Radius",
		CIRCULAR_RIPPLE_RADIUS_MIN,         
		CIRCULAR_RIPPLE_RADIUS_MAX,         
		CIRCULAR_RIPPLE_RADIUS_MIN,          
		CIRCULAR_RIPPLE_RADIUS_MAX,          
		CIRCULAR_RIPPLE_RADIUS_DFLT,        
		PF_Precision_HUNDREDTHS,
		0,
		0,
		RADIUS_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Feather",
		CIRCULAR_RIPPLE_FEATHER_MIN,        
		CIRCULAR_RIPPLE_FEATHER_MAX,        
		CIRCULAR_RIPPLE_FEATHER_MIN,         
		CIRCULAR_RIPPLE_FEATHER_MAX,         
		CIRCULAR_RIPPLE_FEATHER_DFLT,       
		PF_Precision_THOUSANDTHS,
		0,
		0,
		FEATHER_DISK_ID);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_GROUP_START;
	PF_STRCPY(def.name, "Tiles");
	def.flags = PF_ParamFlag_START_COLLAPSED;      
	PF_ADD_PARAM(in_data, CIRCULAR_RIPPLE_TILES_GROUP, &def);

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
	PF_ADD_PARAM(in_data, CIRCULAR_RIPPLE_TILES_GROUP_END, &def);

	out_data->num_params = CIRCULAR_RIPPLE_NUM_PARAMS;

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
	cl_kernel ripple_kernel;
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
	ShaderObjectPtr mRippleShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState>ripple_pipeline;
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

		size_t sizes[] = { strlen(k16fString), strlen(kCircularRippleKernel_OpenCLString) };
		char const* strings[] = { k16fString, kCircularRippleKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->ripple_kernel = clCreateKernel(program, "CircularRippleKernel", &result);
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
		dx_gpu_data->mRippleShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"CircularRippleKernel", csoPath, sigPath));

		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mRippleShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		NSString* source = [NSString stringWithCString : kCircularRippleKernel_MetalString encoding : NSUTF8StringEncoding];
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
			id<MTLFunction> ripple_function = nil;
			NSString* ripple_name = [NSString stringWithCString : "CircularRippleKernel" encoding : NSUTF8StringEncoding];

			ripple_function = [[library newFunctionWithName : ripple_name]autorelease];

			if (!ripple_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->ripple_pipeline = [device newComputePipelineStateWithFunction : ripple_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->ripple_kernel);

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
		dx_gpu_dataP->mRippleShader.reset();

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

static float smoothstep(float edge0, float edge1, float x) {
	float t = MAX(0.0f, MIN(1.0f, (x - edge0) / (edge1 - edge0)));
	return t * t * (3.0f - 2.0f * t);
}

template <typename PixelType>
static void
SampleBilinear(
	PF_EffectWorld* input,
	float x,
	float y,
	PixelType* outP,
	bool x_tiles = false,
	bool y_tiles = false,
	bool mirror = false)
{
	if (!input || !input->data || !outP) {
		if (outP) {
			outP->alpha = 0;
			outP->red = 0;
			outP->green = 0;
			outP->blue = 0;
		}
		return;
	}

	A_long width = input->width;
	A_long height = input->height;
	A_long rowbytes = input->rowbytes;

	if (width <= 0 || height <= 0 || rowbytes <= 0) {
		outP->alpha = 0;
		outP->red = 0;
		outP->green = 0;
		outP->blue = 0;
		return;
	}

	float orig_x = x;
	float orig_y = y;

	if (x_tiles || y_tiles) {
		if (x_tiles) {
			if (mirror) {
				float normalizedX = x / width;
				float flooredX = floor(normalizedX);
				float fractX = normalizedX - flooredX;

				if (static_cast<int>(flooredX) % 2 == 0) {
					x = width * fractX;
				}
				else {
					x = width * (1.0f - fractX);
				}
			}
			else {
				x = fmodf(x, width);
				if (x < 0) x += width;    
			}
		}
		else {
			if (x < 0 || x >= width) {
				outP->alpha = 0;
				outP->red = 0;
				outP->green = 0;
				outP->blue = 0;
				return;
			}
		}

		if (y_tiles) {
			if (mirror) {
				float normalizedY = y / height;
				float flooredY = floor(normalizedY);
				float fractY = normalizedY - flooredY;

				if (static_cast<int>(flooredY) % 2 == 0) {
					y = height * fractY;
				}
				else {
					y = height * (1.0f - fractY);
				}
			}
			else {
				y = fmodf(y, height);
				if (y < 0) y += height;    
			}
		}
		else {
			if (y < 0 || y >= height) {
				outP->alpha = 0;
				outP->red = 0;
				outP->green = 0;
				outP->blue = 0;
				return;
			}
		}
	}
	else {
		if (x < 0 || x >= width || y < 0 || y >= height) {
			outP->alpha = 0;
			outP->red = 0;
			outP->green = 0;
			outP->blue = 0;
			return;
		}
	}

	x = MAX(0, MIN(width - 1.001f, x));
	y = MAX(0, MIN(height - 1.001f, y));

	int x0 = (int)x;
	int y0 = (int)y;
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	float fx = x - x0;
	float fy = y - y0;

	x0 = MIN(MAX(x0, 0), width - 1);
	y0 = MIN(MAX(y0, 0), height - 1);
	x1 = MIN(MAX(x1, 0), width - 1);
	y1 = MIN(MAX(y1, 0), height - 1);

	try {
		if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
			PF_PixelFloat* base = reinterpret_cast<PF_PixelFloat*>(input->data);

			size_t row_stride = input->rowbytes / sizeof(PF_PixelFloat);
			if (row_stride <= 0 || y0 >= height || y1 >= height) {
				outP->alpha = 0;
				outP->red = 0;
				outP->green = 0;
				outP->blue = 0;
				return;
			}

			PF_PixelFloat* p00 = base + y0 * row_stride + x0;
			PF_PixelFloat* p01 = base + y1 * row_stride + x0;
			PF_PixelFloat* p10 = base + y0 * row_stride + x1;
			PF_PixelFloat* p11 = base + y1 * row_stride + x1;

			float oneMinusFx = 1.0f - fx;
			float oneMinusFy = 1.0f - fy;

			float w00 = oneMinusFx * oneMinusFy;
			float w10 = fx * oneMinusFy;
			float w01 = oneMinusFx * fy;
			float w11 = fx * fy;

			outP->alpha = p00->alpha * w00 + p10->alpha * w10 + p01->alpha * w01 + p11->alpha * w11;
			outP->red = p00->red * w00 + p10->red * w10 + p01->red * w01 + p11->red * w11;
			outP->green = p00->green * w00 + p10->green * w10 + p01->green * w01 + p11->green * w11;
			outP->blue = p00->blue * w00 + p10->blue * w10 + p01->blue * w01 + p11->blue * w11;
		}
		else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
			PF_Pixel16* base = reinterpret_cast<PF_Pixel16*>(input->data);

			size_t row_stride = input->rowbytes / sizeof(PF_Pixel16);
			if (row_stride <= 0 || y0 >= height || y1 >= height) {
				outP->alpha = 0;
				outP->red = 0;
				outP->green = 0;
				outP->blue = 0;
				return;
			}

			PF_Pixel16* p00 = base + y0 * row_stride + x0;
			PF_Pixel16* p01 = base + y1 * row_stride + x0;
			PF_Pixel16* p10 = base + y0 * row_stride + x1;
			PF_Pixel16* p11 = base + y1 * row_stride + x1;

			float oneMinusFx = 1.0f - fx;
			float oneMinusFy = 1.0f - fy;

			float w00 = oneMinusFx * oneMinusFy;
			float w10 = fx * oneMinusFy;
			float w01 = oneMinusFx * fy;
			float w11 = fx * fy;

			outP->alpha = (A_u_short)(p00->alpha * w00 + p10->alpha * w10 + p01->alpha * w01 + p11->alpha * w11 + 0.5f);
			outP->red = (A_u_short)(p00->red * w00 + p10->red * w10 + p01->red * w01 + p11->red * w11 + 0.5f);
			outP->green = (A_u_short)(p00->green * w00 + p10->green * w10 + p01->green * w01 + p11->green * w11 + 0.5f);
			outP->blue = (A_u_short)(p00->blue * w00 + p10->blue * w10 + p01->blue * w01 + p11->blue * w11 + 0.5f);
		}
		else {  
			PF_Pixel8* base = reinterpret_cast<PF_Pixel8*>(input->data);

			size_t row_stride = input->rowbytes / sizeof(PF_Pixel8);
			if (row_stride <= 0 || y0 >= height || y1 >= height) {
				outP->alpha = 0;
				outP->red = 0;
				outP->green = 0;
				outP->blue = 0;
				return;
			}

			PF_Pixel8* p00 = base + y0 * row_stride + x0;
			PF_Pixel8* p01 = base + y1 * row_stride + x0;
			PF_Pixel8* p10 = base + y0 * row_stride + x1;
			PF_Pixel8* p11 = base + y1 * row_stride + x1;

			float oneMinusFx = 1.0f - fx;
			float oneMinusFy = 1.0f - fy;

			float w00 = oneMinusFx * oneMinusFy;
			float w10 = fx * oneMinusFy;
			float w01 = oneMinusFx * fy;
			float w11 = fx * fy;

			outP->alpha = (A_u_char)(p00->alpha * w00 + p10->alpha * w10 + p01->alpha * w01 + p11->alpha * w11 + 0.5f);
			outP->red = (A_u_char)(p00->red * w00 + p10->red * w10 + p01->red * w01 + p11->red * w11 + 0.5f);
			outP->green = (A_u_char)(p00->green * w00 + p10->green * w10 + p01->green * w01 + p11->green * w11 + 0.5f);
			outP->blue = (A_u_char)(p00->blue * w00 + p10->blue * w10 + p01->blue * w01 + p11->blue * w11 + 0.5f);
		}
	}
	catch (...) {
		outP->alpha = 0;
		outP->red = 0;
		outP->green = 0;
		outP->blue = 0;
	}
}



template <typename PixelType>
static PF_Err
RippleFuncGeneric(
	void* refcon,
	A_long      xL,
	A_long      yL,
	PixelType* inP,
	PixelType* outP)
{
	PF_Err err = PF_Err_NONE;

	if (!refcon || !inP || !outP) {
		if (inP && outP) {
			*outP = *inP;
		}
		else if (outP) {
			outP->alpha = 0;
			outP->red = 0;
			outP->green = 0;
			outP->blue = 0;
		}
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	RippleInfo* rippleP = reinterpret_cast<RippleInfo*>(refcon);

	*outP = *inP;

	if (!rippleP->src || rippleP->width <= 0 || rippleP->height <= 0 || rippleP->rowbytes <= 0) {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	PF_FpLong width = (PF_FpLong)rippleP->width;
	PF_FpLong height = (PF_FpLong)rippleP->height;

	PF_FpLong centerX = width / 2.0 + (PF_FpLong)rippleP->center.x / 65536.0;
	PF_FpLong centerY = height / 2.0 - (PF_FpLong)rippleP->center.y / 65536.0;

	PF_FpLong uvX = (PF_FpLong)xL / width;
	PF_FpLong uvY = (PF_FpLong)yL / height;

	PF_FpLong centerNormX = centerX / width;
	PF_FpLong centerNormY = centerY / height;

	PF_FpLong offsetX = uvX - centerNormX;
	PF_FpLong offsetY = uvY - centerNormY;

	offsetY *= (height / width);

	PF_FpLong dist = sqrt(offsetX * offsetX + offsetY * offsetY);

	PF_FpLong featherSize = rippleP->radius * 0.5 * rippleP->feather;
	PF_FpLong innerRadius = MAX(0.0, rippleP->radius - featherSize);
	PF_FpLong outerRadius = MAX(innerRadius + 0.00001, rippleP->radius + featherSize);

	PF_FpLong damping;
	if (dist >= outerRadius) {
		damping = 0.0;
	}
	else if (dist <= innerRadius) {
		damping = 1.0;
	}
	else {
		PF_FpLong t = (dist - innerRadius) / (outerRadius - innerRadius);
		t = 1.0 - t;   
		damping = t * t * (3.0 - 2.0 * t);
	}

	const PF_FpLong PI = 3.14159265358979323846;
	PF_FpLong angle = (dist * rippleP->frequency * PI * 2.0) + (rippleP->phase * PI * 2.0);
	PF_FpLong sinVal = sin(angle);

	PF_FpLong len = sqrt(offsetX * offsetX + offsetY * offsetY);
	PF_FpLong normX = 0, normY = 0;
	if (len > 0.0001) {
		normX = offsetX / len;
		normY = offsetY / len;
	}

	PF_FpLong strength_factor = rippleP->strength / 2.0;
	PF_FpLong offsetFactorX = sinVal * strength_factor * normX * damping;
	PF_FpLong offsetFactorY = sinVal * strength_factor * normY * damping;

	offsetX += offsetFactorX;
	offsetY += offsetFactorY;

	offsetY /= (height / width);

	PF_FpLong sampleX = (offsetX + centerNormX) * width;
	PF_FpLong sampleY = (offsetY + centerNormY) * height;

	PF_EffectWorld inputWorld;
	inputWorld.data = rippleP->src;
	inputWorld.width = rippleP->width;
	inputWorld.height = rippleP->height;
	inputWorld.rowbytes = rippleP->rowbytes;

	SampleBilinear(&inputWorld,
		static_cast<float>(sampleX),
		static_cast<float>(sampleY),
		outP,
		rippleP->x_tiles,        
		rippleP->y_tiles,        
		rippleP->mirror);       

	return err;
}

static void
DisposePreRenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		RippleInfo* infoP = reinterpret_cast<RippleInfo*>(pre_render_dataPV);
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

	RippleInfo* infoP = reinterpret_cast<RippleInfo*>(malloc(sizeof(RippleInfo)));

	if (infoP) {
		AEFX_CLR_STRUCT(*infoP);

		PF_ParamDef cur_param;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_CENTER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->center.x = cur_param.u.td.x_value;
		infoP->center.y = cur_param.u.td.y_value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_FREQUENCY, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->frequency = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->strength = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_PHASE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->phase = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_RADIUS, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->radius = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_FEATHER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->feather = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->x_tiles = cur_param.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->y_tiles = cur_param.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, CIRCULAR_RIPPLE_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mirror = cur_param.u.bd.value;

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;

		ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
			CIRCULAR_RIPPLE_INPUT,
			CIRCULAR_RIPPLE_INPUT,
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
	PF_PixelFormat			pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	RippleInfo* infoP)
{
	PF_Err			err = PF_Err_NONE;

	if (!err) {
		switch (pixel_format) {

		case PF_PixelFormat_ARGB128: {
			AEFX_SuiteScoper<PF_iterateFloatSuite2> iterateFloatSuite =
				AEFX_SuiteScoper<PF_iterateFloatSuite2>(in_data,
					kPFIterateFloatSuite,
					kPFIterateFloatSuiteVersion2,
					out_data);
			iterateFloatSuite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)infoP,
				RippleFuncGeneric<PF_PixelFloat>,
				output_worldP);
			break;
		}

		case PF_PixelFormat_ARGB64: {
			AEFX_SuiteScoper<PF_iterate16Suite2> iterate16Suite =
				AEFX_SuiteScoper<PF_iterate16Suite2>(in_data,
					kPFIterate16Suite,
					kPFIterate16SuiteVersion2,
					out_data);
			iterate16Suite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)infoP,
				RippleFuncGeneric<PF_Pixel16>,
				output_worldP);
			break;
		}

		case PF_PixelFormat_ARGB32: {
			AEFX_SuiteScoper<PF_Iterate8Suite2> iterate8Suite =
				AEFX_SuiteScoper<PF_Iterate8Suite2>(in_data,
					kPFIterate8Suite,
					kPFIterate8SuiteVersion2,
					out_data);

			iterate8Suite->iterate(in_data,
				0,
				output_worldP->height,
				input_worldP,
				NULL,
				(void*)infoP,
				RippleFuncGeneric<PF_Pixel8>,
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
	float mCenterX;
	float mCenterY;
	float mFrequency;
	float mStrength;
	float mPhase;
	float mRadius;
	float mFeather;
	int mXTiles;
	int mYTiles;
	int mMirror;
} CircularRippleParams;


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
	PF_PixelFormat			pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	RippleInfo* infoP)
{
	PF_Err			err = PF_Err_NONE;

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

	CircularRippleParams ripple_params;
	ripple_params.mWidth = input_worldP->width;
	ripple_params.mHeight = input_worldP->height;

	A_long src_row_bytes = input_worldP->rowbytes;
	A_long dst_row_bytes = output_worldP->rowbytes;

	ripple_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
	ripple_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
	ripple_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

	ripple_params.mCenterX = (float)(input_worldP->width / 2.0 + (PF_FpLong)infoP->center.x / 65536.0);
	ripple_params.mCenterY = (float)(input_worldP->height / 2.0 - (PF_FpLong)infoP->center.y / 65536.0);

	ripple_params.mFrequency = (float)infoP->frequency;
	ripple_params.mStrength = (float)infoP->strength;
	ripple_params.mPhase = (float)infoP->phase;
	ripple_params.mRadius = (float)infoP->radius;
	ripple_params.mFeather = (float)infoP->feather;
	ripple_params.mXTiles = infoP->x_tiles;
	ripple_params.mYTiles = infoP->y_tiles;
	ripple_params.mMirror = infoP->mirror;

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(int), &ripple_params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(int), &ripple_params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(int), &ripple_params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(int), &ripple_params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(int), &ripple_params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(float), &ripple_params.mCenterX));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(float), &ripple_params.mCenterY));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(float), &ripple_params.mFrequency));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(float), &ripple_params.mStrength));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(float), &ripple_params.mPhase));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(float), &ripple_params.mRadius));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(float), &ripple_params.mFeather));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(int), &ripple_params.mXTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(int), &ripple_params.mYTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->ripple_kernel, param_index++, sizeof(int), &ripple_params.mMirror));

		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(ripple_params.mWidth, threadBlock[0]), RoundUp(ripple_params.mHeight, threadBlock[1]) };

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->ripple_kernel,
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

		CircularRipple_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			ripple_params.mSrcPitch,
			ripple_params.mDstPitch,
			ripple_params.m16f,
			ripple_params.mWidth,
			ripple_params.mHeight,
			ripple_params.mCenterX,
			ripple_params.mCenterY,
			ripple_params.mFrequency,
			ripple_params.mStrength,
			ripple_params.mPhase,
			ripple_params.mRadius,
			ripple_params.mFeather,
			ripple_params.mXTiles,
			ripple_params.mYTiles,
			ripple_params.mMirror);

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
			dx_gpu_data->mRippleShader,
			3);

		DX_ERR(shaderExecution.SetParamBuffer(&ripple_params, sizeof(CircularRippleParams)));
		DX_ERR(shaderExecution.SetUnorderedAccessView(
			(ID3D12Resource*)dst_mem,
			ripple_params.mHeight * dst_row_bytes));
		DX_ERR(shaderExecution.SetShaderResourceView(
			(ID3D12Resource*)src_mem,
			ripple_params.mHeight * src_row_bytes));
		DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(ripple_params.mWidth, 16), (UINT)DivideRoundUp(ripple_params.mHeight, 16)));
	}
#endif
#if HAS_METAL
	else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		Handle metal_handle = (Handle)extraP->input->gpu_data;
		MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> ripple_param_buffer = [[device newBufferWithBytes : &ripple_params
			length : sizeof(CircularRippleParams)
			options : MTLResourceStorageModeManaged]autorelease];

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->ripple_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = { DivideRoundUp(ripple_params.mWidth, threadsPerGroup.width), DivideRoundUp(ripple_params.mHeight, threadsPerGroup.height), 1 };

		[computeEncoder setComputePipelineState : metal_dataP->ripple_pipeline] ;
		[computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
		[computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
		[computeEncoder setBuffer : ripple_param_buffer offset : 0 atIndex : 2] ;
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

	if (!extraP || !extraP->input) {
		return PF_Err_INTERNAL_STRUCT_DAMAGED;
	}

	PF_EffectWorld* input_worldP = NULL,
		* output_worldP = NULL;

	RippleInfo* infoP = NULL;

	if (extraP->input->pre_render_data) {
		infoP = reinterpret_cast<RippleInfo*>(extraP->input->pre_render_data);
	}
	else {
		infoP = (RippleInfo*)malloc(sizeof(RippleInfo));
		if (infoP) {
			memset(infoP, 0, sizeof(RippleInfo));
			infoP->frequency = CIRCULAR_RIPPLE_FREQ_DFLT;
			infoP->strength = CIRCULAR_RIPPLE_STRENGTH_DFLT;
		}
	}

	if (infoP) {
		err = extraP->cb->checkout_layer_pixels(in_data->effect_ref, CIRCULAR_RIPPLE_INPUT, &input_worldP);

		if (err || !input_worldP) {
			if (infoP && !extraP->input->pre_render_data) {
				free(infoP);       
			}
			return err ? err : PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		err = extraP->cb->checkout_output(in_data->effect_ref, &output_worldP);

		if (err || !output_worldP) {
			if (input_worldP) {
				extraP->cb->checkin_layer_pixels(in_data->effect_ref, CIRCULAR_RIPPLE_INPUT);
			}
			if (infoP && !extraP->input->pre_render_data) {
				free(infoP);
			}
			return err ? err : PF_Err_INTERNAL_STRUCT_DAMAGED;
		}

		infoP->width = input_worldP->width;
		infoP->height = input_worldP->height;
		infoP->src = input_worldP->data;
		infoP->rowbytes = input_worldP->rowbytes;

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
			kPFWorldSuite,
			kPFWorldSuiteVersion2,
			out_data);
		PF_PixelFormat pixel_format = PF_PixelFormat_INVALID;
		err = world_suite->PF_GetPixelFormat(input_worldP, &pixel_format);

		if (!err) {
			if (isGPU) {
				err = SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP);
			}
			else {
				err = SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP);
			}
		}

		if (infoP && !extraP->input->pre_render_data) {
			free(infoP);
		}

		err2 = extraP->cb->checkin_layer_pixels(in_data->effect_ref, CIRCULAR_RIPPLE_INPUT);
		if (!err) err = err2;
	}
	else {
		err = extraP->cb->checkout_layer_pixels(in_data->effect_ref, CIRCULAR_RIPPLE_INPUT, &input_worldP);
		if (!err && input_worldP) {
			err = extraP->cb->checkout_output(in_data->effect_ref, &output_worldP);
			if (!err && output_worldP) {
				AEFX_SuiteScoper<PF_WorldTransformSuite1> transform_suite =
					AEFX_SuiteScoper<PF_WorldTransformSuite1>(in_data,
						kPFWorldTransformSuite,
						kPFWorldTransformSuiteVersion1,
						out_data);

				err = transform_suite->copy(in_data->effect_ref, input_worldP, output_worldP, NULL, NULL);
			}
			err2 = extraP->cb->checkin_layer_pixels(in_data->effect_ref, CIRCULAR_RIPPLE_INPUT);
			if (!err) err = err2;
		}
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
	PF_Err                err = PF_Err_NONE;
	AEGP_SuiteHandler    suites(in_data->pica_basicP);

	RippleInfo            ripple;
	AEFX_CLR_STRUCT(ripple);

	ripple.center.x = params[CIRCULAR_RIPPLE_CENTER]->u.td.x_value;
	ripple.center.y = params[CIRCULAR_RIPPLE_CENTER]->u.td.y_value;
	ripple.frequency = params[CIRCULAR_RIPPLE_FREQUENCY]->u.fs_d.value;
	ripple.strength = params[CIRCULAR_RIPPLE_STRENGTH]->u.fs_d.value;
	ripple.phase = params[CIRCULAR_RIPPLE_PHASE]->u.fs_d.value;
	ripple.radius = params[CIRCULAR_RIPPLE_RADIUS]->u.fs_d.value;
	ripple.feather = params[CIRCULAR_RIPPLE_FEATHER]->u.fs_d.value;

	ripple.x_tiles = params[CIRCULAR_RIPPLE_X_TILES]->u.bd.value;
	ripple.y_tiles = params[CIRCULAR_RIPPLE_Y_TILES]->u.bd.value;
	ripple.mirror = params[CIRCULAR_RIPPLE_MIRROR]->u.bd.value;

	ripple.width = params[CIRCULAR_RIPPLE_INPUT]->u.ld.width;
	ripple.height = params[CIRCULAR_RIPPLE_INPUT]->u.ld.height;
	ripple.src = params[CIRCULAR_RIPPLE_INPUT]->u.ld.data;
	ripple.rowbytes = params[CIRCULAR_RIPPLE_INPUT]->u.ld.rowbytes;

	double bytesPerPixel = (double)ripple.rowbytes / (double)ripple.width;
	bool is16bit = false;
	bool is32bit = false;

	if (bytesPerPixel >= 16.0) {        
		is32bit = true;
	}
	else if (bytesPerPixel >= 8.0) {       
		is16bit = true;
	}

	A_long linesL = output->extent_hint.bottom - output->extent_hint.top;

	if (is32bit) {
		ERR(suites.IterateFloatSuite1()->iterate(
			in_data,
			0,                                  
			linesL,                             
			&params[CIRCULAR_RIPPLE_INPUT]->u.ld,   
			NULL,                                  
			(void*)&ripple,                      
			RippleFuncGeneric<PF_PixelFloat>,                    
			output));
	}
	else if (is16bit) {
		ERR(suites.Iterate16Suite2()->iterate(
			in_data,
			0,                                  
			linesL,                             
			&params[CIRCULAR_RIPPLE_INPUT]->u.ld,   
			NULL,                                  
			(void*)&ripple,                      
			RippleFuncGeneric<PF_Pixel16>,                       
			output));
	}
	else {
		ERR(suites.Iterate8Suite2()->iterate(
			in_data,
			0,                                  
			linesL,                             
			&params[CIRCULAR_RIPPLE_INPUT]->u.ld,   
			NULL,                                  
			(void*)&ripple,                      
			RippleFuncGeneric<PF_Pixel8>,                        
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
		"Circular Ripple",  
		"DKT Circular Ripple",   
		"DKT Effects",  
		AE_RESERVED_INFO,   
		"EffectMain",	  
		"https://www.adobe.com");	  

	return result;
}


PF_Err
EffectMain(
	PF_Cmd			cmd,
	PF_InData* in_dataP,
	PF_OutData* out_data,
	PF_ParamDef* params[],
	PF_LayerDef* output,
	void* extra)
{
	PF_Err		err = PF_Err_NONE;

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
