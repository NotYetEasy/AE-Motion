#if HAS_CUDA
#include <cuda_runtime.h>
// SDK_Invert_ProcAmp.h defines these and are needed whereas the cuda_runtime ones are not.
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "PinchBulge.h"
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


// CUDA kernel; see SDK_Invert_ProcAmp.cu.
extern void PinchBulge_CUDA(
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
	float radius,
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

	// For Premiere - declare supported pixel formats
	if (in_dataP->appl_id == 'PrMr') {

		AEFX_SuiteScoper<PF_PixelFormatSuite1> pixelFormatSuite =
			AEFX_SuiteScoper<PF_PixelFormatSuite1>(in_dataP,
				kPFPixelFormatSuite,
				kPFPixelFormatSuiteVersion1,
				out_data);

		//	Add the pixel formats we support in order of preference.
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

	// Add the Center parameter
	PF_ADD_POINT("Center",
		PINCH_CENTER_X_DFLT,
		PINCH_CENTER_Y_DFLT,
		0,
		CENTER_DISK_ID);

	// Add the "Strength" parameter
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Strength",
		PINCH_STRENGTH_MIN,
		PINCH_STRENGTH_MAX,
		PINCH_STRENGTH_MIN,
		PINCH_STRENGTH_MAX,
		PINCH_STRENGTH_DFLT,
		PF_Precision_HUNDREDTHS,
		0,
		0,
		STRENGTH_DISK_ID);

	// Add the "Radius" parameter
	AEFX_CLR_STRUCT(def);
	PF_ADD_FLOAT_SLIDERX("Radius",
		PINCH_RADIUS_MIN,
		PINCH_RADIUS_MAX,
		PINCH_RADIUS_MIN,
		PINCH_RADIUS_MAX,
		PINCH_RADIUS_DFLT,
		PF_Precision_THOUSANDTHS,
		0,
		0,
		RADIUS_DISK_ID);

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

	out_data->num_params = PINCH_NUM_PARAMS;

	return err;
}

#if HAS_METAL
PF_Err NSError2PFErr(NSError* inError)
{
	if (inError)
	{
		return PF_Err_INTERNAL_STRUCT_DAMAGED;  //For debugging, uncomment above line and set breakpoint here
	}
	return PF_Err_NONE;
}
#endif //HAS_METAL


// GPU data initialized at GPU setup and used during render.
struct OpenCLGPUData
{
	cl_kernel pinchbulge_kernel;
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
	ShaderObjectPtr mPinchBulgeShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState>pinchbulge_pipeline;
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

		size_t sizes[] = { strlen(k16fString), strlen(kPinchBulgeKernel_OpenCLString) };
		char const* strings[] = { k16fString, kPinchBulgeKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->pinchbulge_kernel = clCreateKernel(program, "PinchBulgeKernel", &result);
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
		dx_gpu_data->mPinchBulgeShader = std::make_shared<ShaderObject>();

		// Create the DXContext
		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"PinchBulgeKernel", csoPath, sigPath));

		// Load the shader
		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mPinchBulgeShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		//Create a library from source
		NSString* source = [NSString stringWithCString : kPinchBulgeKernel_MetalString encoding : NSUTF8StringEncoding];
		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;

		NSError* error = nil;
		id<MTLLibrary> library = [[device newLibraryWithSource : source options : nil error : &error]autorelease];

		// An error code is set for Metal compile warnings, so use nil library as the error signal
		if (!err && !library) {
			err = NSError2PFErr(error);
		}

		// For debugging only. This will contain Metal compile warnings and erorrs.
		NSString* getError = error.localizedDescription;

		PF_Handle metal_handle = handle_suite->host_new_handle(sizeof(MetalGPUData));
		MetalGPUData* metal_data = reinterpret_cast<MetalGPUData*>(*metal_handle);

		//Create pipeline state from function extracted from library
		if (err == PF_Err_NONE)
		{
			id<MTLFunction> pinchbulge_function = nil;
			NSString* pinchbulge_name = [NSString stringWithCString : "PinchBulgeKernel" encoding : NSUTF8StringEncoding];

			pinchbulge_function = [[library newFunctionWithName : pinchbulge_name]autorelease];

			if (!pinchbulge_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->pinchbulge_pipeline = [device newComputePipelineStateWithFunction : pinchbulge_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->pinchbulge_kernel);

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
		dx_gpu_dataP->mPinchBulgeShader.reset();

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

		// Release Metal resources
		[metal_dataP->pinchbulge_pipeline release] ;

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif

	return err;
}

// Helper function to implement smoothstep operation
static float smoothstep(float edge0, float edge1, float x) {
	float t = MAX(0.0f, MIN(1.0f, (x - edge0) / (edge1 - edge0)));
	return t * t * (3.0f - 2.0f * t);
}

// Helper function to implement mix operation
static float mix(float a, float b, float t) {
	return a * (1.0f - t) + b * t;
}

/**
 * Function to sample pixel with bilinear interpolation and tiling support
 */
template<typename PixelType>
static void SampleBilinear(
	PF_EffectWorld* input,
	float u,
	float v,
	PixelType* outPixel,
	bool x_tiles,
	bool y_tiles,
	bool mirror)
{
	// Get input dimensions
	int width = input->width;
	int height = input->height;

	// Initialize variables to track if we're sampling outside the image
	bool outsideBounds = false;
	float originalU = u;
	float originalV = v;

	// Handle tiling based on X and Y Tiles parameters
	if (x_tiles) {
		// X tiling is enabled
		if (mirror) {
			// Mirror tiling: create ping-pong pattern
			float intPart;
			float fracPart = modff(fabsf(u), &intPart);
			int isOdd = (int)intPart & 1;
			u = isOdd ? 1.0f - fracPart : fracPart;
		}
		else {
			// Regular repeat tiling
			u = u - floorf(u);
		}
	}
	else {
		// X tiling is disabled - check if outside bounds
		if (u < 0.0f || u >= 1.0f) {
			outsideBounds = true;
		}
	}

	// Apply Y tiling
	if (y_tiles) {
		// Y tiling is enabled
		if (mirror) {
			// Mirror tiling: create ping-pong pattern
			float intPart;
			float fracPart = modff(fabsf(v), &intPart);
			int isOdd = (int)intPart & 1;
			v = isOdd ? 1.0f - fracPart : fracPart;
		}
		else {
			// Regular repeat tiling
			v = v - floorf(v);
		}
	}
	else {
		// Y tiling is disabled - check if outside bounds
		if (v < 0.0f || v >= 1.0f) {
			outsideBounds = true;
		}
	}

	// If we're outside bounds and tiling is disabled, return transparent pixel
	if (outsideBounds) {
		if (sizeof(PixelType) == sizeof(PF_Pixel8)) {
			PF_Pixel8* out_8 = reinterpret_cast<PF_Pixel8*>(outPixel);
			out_8->alpha = 0;
			out_8->red = 0;
			out_8->green = 0;
			out_8->blue = 0;
		}
		else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
			PF_Pixel16* out_16 = reinterpret_cast<PF_Pixel16*>(outPixel);
			out_16->alpha = 0;
			out_16->red = 0;
			out_16->green = 0;
			out_16->blue = 0;
		}
		else { // PF_PixelFloat
			PF_PixelFloat* out_f = reinterpret_cast<PF_PixelFloat*>(outPixel);
			out_f->alpha = 0.0f;
			out_f->red = 0.0f;
			out_f->green = 0.0f;
			out_f->blue = 0.0f;
		}
		return;
	}

	// Clamp u and v to [0,1) range for sampling within the image
	u = MAX(0.0f, MIN(0.9999f, u));
	v = MAX(0.0f, MIN(0.9999f, v));

	// Convert to pixel coordinates
	float x = u * width;
	float y = v * height;

	// Get the four surrounding pixel coordinates
	int x0 = (int)floorf(x);
	int y0 = (int)floorf(y);
	int x1 = x0 + 1;
	int y1 = y0 + 1;

	// Ensure coordinates are within bounds
	x0 = MAX(0, MIN(width - 1, x0));
	y0 = MAX(0, MIN(height - 1, y0));
	x1 = MAX(0, MIN(width - 1, x1));
	y1 = MAX(0, MIN(height - 1, y1));

	// Calculate interpolation factors
	float fx = x - x0;
	float fy = y - y0;

	// Get pointers to the four surrounding pixels
	PixelType* p00;
	PixelType* p10;
	PixelType* p01;
	PixelType* p11;

	// Calculate pixel addresses based on pixel type
	if (sizeof(PixelType) == sizeof(PF_Pixel8)) {
		PF_Pixel8* base = reinterpret_cast<PF_Pixel8*>(input->data);
		p00 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_Pixel8)) + x0);
		p10 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_Pixel8)) + x1);
		p01 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_Pixel8)) + x0);
		p11 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_Pixel8)) + x1);
	}
	else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
		PF_Pixel16* base = reinterpret_cast<PF_Pixel16*>(input->data);
		p00 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_Pixel16)) + x0);
		p10 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_Pixel16)) + x1);
		p01 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_Pixel16)) + x0);
		p11 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_Pixel16)) + x1);
	}
	else { // PF_PixelFloat
		PF_PixelFloat* base = reinterpret_cast<PF_PixelFloat*>(input->data);
		p00 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_PixelFloat)) + x0);
		p10 = reinterpret_cast<PixelType*>(base + y0 * (input->rowbytes / sizeof(PF_PixelFloat)) + x1);
		p01 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_PixelFloat)) + x0);
		p11 = reinterpret_cast<PixelType*>(base + y1 * (input->rowbytes / sizeof(PF_PixelFloat)) + x1);
	}

	// Perform bilinear interpolation based on pixel type
	if (sizeof(PixelType) == sizeof(PF_Pixel8)) {
		PF_Pixel8* p00_8 = reinterpret_cast<PF_Pixel8*>(p00);
		PF_Pixel8* p10_8 = reinterpret_cast<PF_Pixel8*>(p10);
		PF_Pixel8* p01_8 = reinterpret_cast<PF_Pixel8*>(p01);
		PF_Pixel8* p11_8 = reinterpret_cast<PF_Pixel8*>(p11);
		PF_Pixel8* out_8 = reinterpret_cast<PF_Pixel8*>(outPixel);

		// Bilinear interpolation for 8-bit pixels
		float oneMinusFx = 1.0f - fx;
		float oneMinusFy = 1.0f - fy;

		float w00 = oneMinusFx * oneMinusFy;
		float w10 = fx * oneMinusFy;
		float w01 = oneMinusFx * fy;
		float w11 = fx * fy;

		out_8->alpha = (A_u_char)(p00_8->alpha * w00 + p10_8->alpha * w10 + p01_8->alpha * w01 + p11_8->alpha * w11);
		out_8->red = (A_u_char)(p00_8->red * w00 + p10_8->red * w10 + p01_8->red * w01 + p11_8->red * w11);
		out_8->green = (A_u_char)(p00_8->green * w00 + p10_8->green * w10 + p01_8->green * w01 + p11_8->green * w11);
		out_8->blue = (A_u_char)(p00_8->blue * w00 + p10_8->blue * w10 + p01_8->blue * w01 + p11_8->blue * w11);
	}
	else if (sizeof(PixelType) == sizeof(PF_Pixel16)) {
		PF_Pixel16* p00_16 = reinterpret_cast<PF_Pixel16*>(p00);
		PF_Pixel16* p10_16 = reinterpret_cast<PF_Pixel16*>(p10);
		PF_Pixel16* p01_16 = reinterpret_cast<PF_Pixel16*>(p01);
		PF_Pixel16* p11_16 = reinterpret_cast<PF_Pixel16*>(p11);
		PF_Pixel16* out_16 = reinterpret_cast<PF_Pixel16*>(outPixel);

		// Bilinear interpolation for 16-bit pixels
		float oneMinusFx = 1.0f - fx;
		float oneMinusFy = 1.0f - fy;

		float w00 = oneMinusFx * oneMinusFy;
		float w10 = fx * oneMinusFy;
		float w01 = oneMinusFx * fy;
		float w11 = fx * fy;

		out_16->alpha = (A_u_short)(p00_16->alpha * w00 + p10_16->alpha * w10 + p01_16->alpha * w01 + p11_16->alpha * w11);
		out_16->red = (A_u_short)(p00_16->red * w00 + p10_16->red * w10 + p01_16->red * w01 + p11_16->red * w11);
		out_16->green = (A_u_short)(p00_16->green * w00 + p10_16->green * w10 + p01_16->green * w01 + p11_16->green * w11);
		out_16->blue = (A_u_short)(p00_16->blue * w00 + p10_16->blue * w10 + p01_16->blue * w01 + p11_16->blue * w11);
	}
	else { // PF_PixelFloat
		PF_PixelFloat* p00_f = reinterpret_cast<PF_PixelFloat*>(p00);
		PF_PixelFloat* p10_f = reinterpret_cast<PF_PixelFloat*>(p10);
		PF_PixelFloat* p01_f = reinterpret_cast<PF_PixelFloat*>(p01);
		PF_PixelFloat* p11_f = reinterpret_cast<PF_PixelFloat*>(p11);
		PF_PixelFloat* out_f = reinterpret_cast<PF_PixelFloat*>(outPixel);

		// Bilinear interpolation for float pixels
		float oneMinusFx = 1.0f - fx;
		float oneMinusFy = 1.0f - fy;

		float w00 = oneMinusFx * oneMinusFy;
		float w10 = fx * oneMinusFy;
		float w01 = oneMinusFx * fy;
		float w11 = fx * fy;

		out_f->alpha = p00_f->alpha * w00 + p10_f->alpha * w10 + p01_f->alpha * w01 + p11_f->alpha * w11;
		out_f->red = p00_f->red * w00 + p10_f->red * w10 + p01_f->red * w01 + p11_f->red * w11;
		out_f->green = p00_f->green * w00 + p10_f->green * w10 + p01_f->green * w01 + p11_f->green * w11;
		out_f->blue = p00_f->blue * w00 + p10_f->blue * w10 + p01_f->blue * w01 + p11_f->blue * w11;
	}
}


// Template function for pixel processing
template<typename PixelType>
static PF_Err
PinchFuncTemplate(
	void* refcon,
	A_long xL,
	A_long yL,
	PixelType* inP,
	PixelType* outP)
{
	PF_Err err = PF_Err_NONE;
	PinchInfo* piP = reinterpret_cast<PinchInfo*>(refcon);

	if (piP) {
		// Get input layer dimensions
		float width = (float)piP->input->width;
		float height = (float)piP->input->height;

		// Calculate center position based on user parameter
		float centerX = width / 2.0f + (PF_FpLong)piP->center_x / 65536.0f;
		float centerY = height / 2.0f - (PF_FpLong)piP->center_y / 65536.0f;

		// Adjust for pre-effect source origin (from previous effects)
		centerX += piP->in_data->pre_effect_source_origin_x;
		centerY += piP->in_data->pre_effect_source_origin_y;

		// Calculate normalized UV coordinates (0-1 range)
		float uvX = (float)xL / width;
		float uvY = (float)yL / height;

		// Calculate center in normalized coordinates
		float centerNormX = centerX / width;
		float centerNormY = centerY / height;

		// Offset UV by center
		float offsetX = uvX - centerNormX;
		float offsetY = uvY - centerNormY;

		// Adjust for aspect ratio (multiply X by aspect ratio)
		float aspectRatio = width / height;
		offsetX *= aspectRatio;

		// Calculate distance from center
		float dist = sqrt(offsetX * offsetX + offsetY * offsetY);

		// Store original offsets for reference
		float prevOffsetX = offsetX;
		float prevOffsetY = offsetY;

		// Apply distortion if within radius
		if (dist < piP->radius) {
			float p = dist / piP->radius;

			if (piP->strength > 0.0f) {
				// Bulge effect
				float factor = mix(1.0f, smoothstep(0.0f, piP->radius / dist, p), piP->strength * 0.75f);
				offsetX *= factor;
				offsetY *= factor;
			}
			else {
				// Pinch effect
				float factor = mix(1.0f, pow(p, 1.0f + piP->strength * 0.75f) * piP->radius / dist, 1.0f - p);
				offsetX *= factor;
				offsetY *= factor;
			}
		}

		// Undo aspect ratio adjustment
		offsetX /= aspectRatio;

		// Calculate final sampling coordinates in normalized space (0-1)
		float finalSrcX = offsetX + centerNormX;
		float finalSrcY = offsetY + centerNormY;

		// Use our custom SampleBilinear function with tiling support
		SampleBilinear(piP->input, finalSrcX, finalSrcY, outP, piP->x_tiles, piP->y_tiles, piP->mirror);
	}
	else {
		*outP = *inP;
	}

	return err;
}

// CPU implementation of PinchBulge for 8-bit pixels
static PF_Err
PinchFunc8(
	void* refcon,
	A_long        xL,
	A_long        yL,
	PF_Pixel8* inP,
	PF_Pixel8* outP)
{
	return PinchFuncTemplate<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

// CPU implementation of PinchBulge for 16-bit pixels
static PF_Err
PinchFunc16(
	void* refcon,
	A_long        xL,
	A_long        yL,
	PF_Pixel16* inP,
	PF_Pixel16* outP)
{
	return PinchFuncTemplate<PF_Pixel16>(refcon, xL, yL, inP, outP);
}


// CPU implementation of PinchBulge for 32-bit float pixels
static PF_Err
PinchFuncFloat(
	void* refcon,
	A_long            xL,
	A_long            yL,
	PF_PixelFloat* inP,
	PF_PixelFloat* outP)
{
	return PinchFuncTemplate<PF_PixelFloat>(refcon, xL, yL, inP, outP);
}


// Render function for Premiere
// Render function for After Effects
static PF_Err
Render(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_ParamDef* params[],
	PF_LayerDef* output)
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_dataP->pica_basicP);

	// For Premiere, we'll just copy input to output for now
	// This would need to be implemented for Premiere compatibility
	if (in_dataP->appl_id == 'PrMr')
	{
		PF_LayerDef* src = &params[0]->u.ld;
		PF_LayerDef* dest = output;

		const char* srcData = (const char*)src->data;
		char* destData = (char*)dest->data;

		// Simple copy of data
		for (int y = 0; y < output->height; ++y, srcData += src->rowbytes, destData += dest->rowbytes)
		{
			memcpy(destData, srcData, src->rowbytes);
		}
	}
	else // After Effects rendering
	{
		PinchInfo piP;
		AEFX_CLR_STRUCT(piP);

		// Get parameter values
		piP.center_x = params[PINCH_CENTER]->u.td.x_value;
		piP.center_y = params[PINCH_CENTER]->u.td.y_value;
		piP.strength = params[PINCH_STRENGTH]->u.fs_d.value;
		piP.radius = params[PINCH_RADIUS]->u.fs_d.value;
		piP.input = &params[PINCH_INPUT]->u.ld;
		piP.in_data = in_dataP;

		// Get tiling parameters
		piP.x_tiles = params[PINCH_X_TILES]->u.bd.value;
		piP.y_tiles = params[PINCH_Y_TILES]->u.bd.value;
		piP.mirror = params[PINCH_MIRROR]->u.bd.value;

		// Determine pixel depth by examining the world
		double bytesPerPixel = (double)piP.input->rowbytes / (double)piP.input->width;
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
				in_dataP,
				0,                                // progress base
				output->height,                   // progress final
				&params[PINCH_INPUT]->u.ld,        // src 
				NULL,                            // area - null for all pixels
				(void*)&piP,                    // refcon - custom data
				PinchFuncFloat,                 // pixel function pointer
				output));
		}
		else if (is16bit) {
			// Process with 16-bit iterate suite
			ERR(suites.Iterate16Suite2()->iterate(
				in_dataP,
				0,                                // progress base
				output->height,                   // progress final
				&params[PINCH_INPUT]->u.ld,        // src 
				NULL,                            // area - null for all pixels
				(void*)&piP,                    // refcon - custom data
				PinchFunc16,                    // pixel function pointer
				output));
		}
		else {
			// Process with 8-bit iterate suite
			ERR(suites.Iterate8Suite2()->iterate(
				in_dataP,
				0,                                // progress base
				output->height,                   // progress final
				&params[PINCH_INPUT]->u.ld,        // src 
				NULL,                            // area - null for all pixels
				(void*)&piP,                    // refcon - custom data
				PinchFunc8,                        // pixel function pointer
				output));
		}
	}

	return err;
}


// Function to dispose of pre-render data
static void
DisposePreRenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		PinchBulgeParams* infoP = reinterpret_cast<PinchBulgeParams*>(pre_render_dataPV);
		free(infoP);
	}
}

// PreRender function - prepares parameters for rendering
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

	PinchBulgeParams* infoP = reinterpret_cast<PinchBulgeParams*>(malloc(sizeof(PinchBulgeParams)));

	if (infoP) {
		// Checkout parameters
		PF_ParamDef cur_param;

		// Center parameter
		ERR(PF_CHECKOUT_PARAM(in_dataP, PINCH_CENTER, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mCenterX = (float)cur_param.u.td.x_value / 65536.0f;
		infoP->mCenterY = (float)cur_param.u.td.y_value / 65536.0f;

		// Strength parameter
		ERR(PF_CHECKOUT_PARAM(in_dataP, PINCH_STRENGTH, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mStrength = cur_param.u.fs_d.value;

		// Radius parameter
		ERR(PF_CHECKOUT_PARAM(in_dataP, PINCH_RADIUS, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mRadius = cur_param.u.fs_d.value;

		// X Tiles parameter
		ERR(PF_CHECKOUT_PARAM(in_dataP, PINCH_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mXTiles = cur_param.u.bd.value;

		// Y Tiles parameter
		ERR(PF_CHECKOUT_PARAM(in_dataP, PINCH_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mYTiles = cur_param.u.bd.value;

		// Mirror parameter
		ERR(PF_CHECKOUT_PARAM(in_dataP, PINCH_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mMirror = cur_param.u.bd.value;

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;

		ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
			PINCH_INPUT,
			PINCH_INPUT,
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

// CPU implementation of PinchBulge
static PF_Err
SmartRenderCPU(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	PinchBulgeParams* infoP)
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	// Convert PinchBulgeParams to PinchInfo for use with the template functions
	PinchInfo info;
	AEFX_CLR_STRUCT(info);

	info.in_data = in_data;
	info.center_x = (PF_FpLong)(infoP->mCenterX * 65536.0f);
	info.center_y = (PF_FpLong)(infoP->mCenterY * 65536.0f);
	info.strength = infoP->mStrength;
	info.radius = infoP->mRadius;
	info.x_tiles = infoP->mXTiles;
	info.y_tiles = infoP->mYTiles;
	info.mirror = infoP->mMirror;
	info.input = input_worldP;

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
			(void*)&info,
			PinchFuncFloat,
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
			(void*)&info,
			PinchFunc16,
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
			(void*)&info,
			PinchFunc8,
			output_worldP);
		break;
	}

	default:
		err = PF_Err_BAD_CALLBACK_PARAM;
		break;
	}

	return err;
}


// Utility function to round up values
static size_t
RoundUp(
	size_t inValue,
	size_t inMultiple)
{
	return inValue ? ((inValue + inMultiple - 1) / inMultiple) * inMultiple : 0;
}

// Utility function to divide and round up
size_t DivideRoundUp(
	size_t inValue,
	size_t inMultiple)
{
	return inValue ? (inValue + inMultiple - 1) / inMultiple : 0;
}

// Structure for PinchBulge kernel parameters
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
	int mXTiles;
	int mYTiles;
	int mMirror;
} PinchBulgeKernelParams;

// GPU implementation of PinchBulge
static PF_Err
SmartRenderGPU(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_PixelFormat			pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	PinchBulgeParams* infoP)
{
	PF_Err			err = PF_Err_NONE;

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

	// Set up parameters for PinchBulge kernel
	PinchBulgeKernelParams params;
	params.mWidth = input_worldP->width;
	params.mHeight = input_worldP->height;

	// Calculate center position based on user parameter and effect
	params.mCenterX = params.mWidth / 2.0f + infoP->mCenterX;
	params.mCenterY = params.mHeight / 2.0f - infoP->mCenterY;

	params.mStrength = infoP->mStrength;
	params.mRadius = infoP->mRadius;
	params.mXTiles = infoP->mXTiles;
	params.mYTiles = infoP->mYTiles;
	params.mMirror = infoP->mMirror;

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

		// Set up kernel for PinchBulge
		cl_uint param_index = 0;
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(int), &params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(int), &params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(int), &params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(int), &params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(int), &params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(float), &params.mCenterX));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(float), &params.mCenterY));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(float), &params.mStrength));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(float), &params.mRadius));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(int), &params.mXTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(int), &params.mYTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->pinchbulge_kernel, param_index++, sizeof(int), &params.mMirror));

		// Launch the kernel
		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(params.mWidth, threadBlock[0]), RoundUp(params.mHeight, threadBlock[1]) };

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->pinchbulge_kernel,
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
		PinchBulge_CUDA(
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
			params.mRadius,
			params.mXTiles,
			params.mYTiles,
			params.mMirror);

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

		// Execute PinchBulge
		DXShaderExecution shaderExecution(
			dx_gpu_data->mContext,
			dx_gpu_data->mPinchBulgeShader,
			3);

		// Note: The order of elements in the param structure should be identical to the order expected by the shader
		DX_ERR(shaderExecution.SetParamBuffer(&params, sizeof(PinchBulgeKernelParams)));
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

		//Set the arguments
		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> pinchbulge_param_buffer = [[device newBufferWithBytes : &params
			length : sizeof(PinchBulgeKernelParams)
			options : MTLResourceStorageModeManaged]autorelease];

		//Launch the command
		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->pinchbulge_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = { DivideRoundUp(params.mWidth, threadsPerGroup.width), DivideRoundUp(params.mHeight, threadsPerGroup.height), 1 };

		[computeEncoder setComputePipelineState : metal_dataP->pinchbulge_pipeline] ;
		[computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
		[computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
		[computeEncoder setBuffer : pinchbulge_param_buffer offset : 0 atIndex : 2] ;
		[computeEncoder dispatchThreadgroups : numThreadgroups threadsPerThreadgroup : threadsPerGroup] ;
		[computeEncoder endEncoding] ;
		[commandBuffer commit] ;

		err = NSError2PFErr([commandBuffer error]);
	}
#endif //HAS_METAL

	return err;
}


static PF_Err
SmartRender(
	PF_InData* in_data,
	PF_OutData* out_data,
	PF_SmartRenderExtra* extraP,
	bool					isGPU)
{
	PF_Err			err = PF_Err_NONE,
		err2 = PF_Err_NONE;

	PF_EffectWorld* input_worldP = NULL,
		* output_worldP = NULL;

	// Parameters can be queried during render. In this example, we pass them from PreRender as an example of using pre_render_data.
	PinchBulgeParams* infoP = reinterpret_cast<PinchBulgeParams*>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, PINCH_INPUT, &input_worldP)));
		ERR(extraP->cb->checkout_output(in_data->effect_ref, &output_worldP));

		AEFX_SuiteScoper<PF_WorldSuite2> world_suite = AEFX_SuiteScoper<PF_WorldSuite2>(in_data,
			kPFWorldSuite,
			kPFWorldSuiteVersion2,
			out_data);
		PF_PixelFormat	pixel_format = PF_PixelFormat_INVALID;
		ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

		if (isGPU) {
			ERR(SmartRenderGPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
		}
		else {
			ERR(SmartRenderCPU(in_data, out_data, pixel_format, input_worldP, output_worldP, extraP, infoP));
		}
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, PINCH_INPUT));
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
		"Pinch/Bulge", // Name
		"DKT Pinch/Bulge", // Match Name
		"DKT Effects", // Category
		AE_RESERVED_INFO, // Reserved Info
		"EffectMain",	// Entry point
		"https://www.adobe.com");	// support URL

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
		// Never EVER throw exceptions into AE.
		err = thrown_err;
	}
	return err;
}

