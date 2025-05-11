#if HAS_CUDA
#include <cuda_runtime.h>
#undef MAJOR_VERSION
#undef MINOR_VERSION
#endif

#include "Transforms.h"
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

extern void Transform_CUDA(
	float const* src,
	float* dst,
	unsigned int srcPitch,
	unsigned int dstPitch,
	int is16f,
	unsigned int width,
	unsigned int height,
	float xPos,
	float yPos,
	float rotation,
	float scale,
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

	PF_ADD_POINT("Position",
		50, 50,
		false,
		POSITION_DISK_ID);

	AEFX_CLR_STRUCT(def);
	PF_ADD_ANGLE("Rotation", 0, ROTATION_DISK_ID);

	AEFX_CLR_STRUCT(def);
	def.param_type = PF_Param_FLOAT_SLIDER;
	PF_STRCPY(def.name, "Scale");
	def.flags = 0;
	def.u.fs_d.valid_min = 0.0f;          
	def.u.fs_d.valid_max = 1000.0f;       
	def.u.fs_d.slider_min = 0.0f;         
	def.u.fs_d.slider_max = 500.0f;       
	def.u.fs_d.value = 100.0f;            
	def.u.fs_d.dephault = 100.0f;         
	def.u.fs_d.precision = 1;              
	def.u.fs_d.display_flags = 0;
	PF_ADD_PARAM(in_data, -1, &def);

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

	out_data->num_params = TRANSFORMS_NUM_PARAMS;

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
	cl_kernel transform_kernel;
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
	ShaderObjectPtr mTransformShader;
};
#endif

#if HAS_METAL
struct MetalGPUData
{
	id<MTLComputePipelineState> transform_pipeline;
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

		size_t sizes[] = { strlen(k16fString), strlen(kTransformsKernel_OpenCLString) };
		char const* strings[] = { k16fString, kTransformsKernel_OpenCLString };
		cl_context context = (cl_context)device_info.contextPV;
		cl_device_id device = (cl_device_id)device_info.devicePV;

		cl_program program;
		if (!err) {
			program = clCreateProgramWithSource(context, 2, &strings[0], &sizes[0], &result);
			CL_ERR(result);
		}

		CL_ERR(clBuildProgram(program, 1, &device, "-cl-single-precision-constant -cl-fast-relaxed-math", 0, 0));

		if (!err) {
			cl_gpu_data->transform_kernel = clCreateKernel(program, "TransformKernel", &result);
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
		dx_gpu_data->mTransformShader = std::make_shared<ShaderObject>();

		DX_ERR(dx_gpu_data->mContext->Initialize(
			(ID3D12Device*)device_info.devicePV,
			(ID3D12CommandQueue*)device_info.command_queuePV));

		std::wstring csoPath, sigPath;
		DX_ERR(GetShaderPath(L"TransformKernel", csoPath, sigPath));

		DX_ERR(dx_gpu_data->mContext->LoadShader(
			csoPath.c_str(),
			sigPath.c_str(),
			dx_gpu_data->mTransformShader));

		extraP->output->gpu_data = gpu_dataH;
		out_dataP->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
	}
#endif
#if HAS_METAL
	else if (extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		NSString* source = [NSString stringWithCString : kSDK_Transform_Kernel_MetalString encoding : NSUTF8StringEncoding];
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
			id<MTLFunction> transform_function = nil;
			NSString* transform_name = [NSString stringWithCString : "TransformKernel" encoding : NSUTF8StringEncoding];

			transform_function = [[library newFunctionWithName : transform_name]autorelease];

			if (!transform_function) {
				err = PF_Err_INTERNAL_STRUCT_DAMAGED;
			}

			if (!err) {
				metal_data->transform_pipeline = [device newComputePipelineStateWithFunction : transform_function error : &error];
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

		(void)clReleaseKernel(cl_gpu_dataP->transform_kernel);

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
		dx_gpu_dataP->mTransformShader.reset();

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

		AEFX_SuiteScoper<PF_HandleSuite1> handle_suite = AEFX_SuiteScoper<PF_HandleSuite1>(in_dataP,
			kPFHandleSuite,
			kPFHandleSuiteVersion1,
			out_dataP);

		handle_suite->host_dispose_handle(gpu_dataH);
	}
#endif
	return err;
}


template<typename T>
inline T CLAMP(T value, T min_val, T max_val) {
	return (value < min_val) ? min_val : ((value > max_val) ? max_val : value);
}

template<typename PixelT>
PixelT SampleBilinear(
	PF_EffectWorld* input,
	float x,
	float y,
	bool x_tiles,
	bool y_tiles,
	bool mirror)
{
	int width = input->width;
	int height = input->height;

	if (x_tiles) {
		if (mirror) {
			float intPartX;
			float fracPartX = modff(fabsf(x / width), &intPartX);
			int isOddX = (int)intPartX & 1;
			x = isOddX ? (1.0f - fracPartX) * width : fracPartX * width;
		}
		else {
			x = fmodf(x, (float)width);
			if (x < 0) x += width;
		}
	}
	else if (x < 0 || x >= width) {
		PixelT transparent;
		if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
			transparent.alpha = 0;
			transparent.red = 0;
			transparent.green = 0;
			transparent.blue = 0;
		}
		else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
			transparent.alpha = 0;
			transparent.red = 0;
			transparent.green = 0;
			transparent.blue = 0;
		}
		else {
			transparent.alpha = 0.0f;
			transparent.red = 0.0f;
			transparent.green = 0.0f;
			transparent.blue = 0.0f;
		}
		return transparent;
	}

	if (y_tiles) {
		if (mirror) {
			float intPartY;
			float fracPartY = modff(fabsf(y / height), &intPartY);
			int isOddY = (int)intPartY & 1;
			y = isOddY ? (1.0f - fracPartY) * height : fracPartY * height;
		}
		else {
			y = fmodf(y, (float)height);
			if (y < 0) y += height;
		}
	}
	else if (y < 0 || y >= height) {
		PixelT transparent;
		if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
			transparent.alpha = 0;
			transparent.red = 0;
			transparent.green = 0;
			transparent.blue = 0;
		}
		else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
			transparent.alpha = 0;
			transparent.red = 0;
			transparent.green = 0;
			transparent.blue = 0;
		}
		else {
			transparent.alpha = 0.0f;
			transparent.red = 0.0f;
			transparent.green = 0.0f;
			transparent.blue = 0.0f;
		}
		return transparent;
	}

	int x_int = static_cast<int>(x);
	int y_int = static_cast<int>(y);
	float x_frac = x - x_int;
	float y_frac = y - y_int;

	int x0 = CLAMP(x_int, 0, width - 1);
	int x1 = CLAMP(x_int + 1, 0, width - 1);
	int y0 = CLAMP(y_int, 0, height - 1);
	int y1 = CLAMP(y_int + 1, 0, height - 1);

	PixelT* p00;
	PixelT* p01;
	PixelT* p10;
	PixelT* p11;

	if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
		PF_Pixel8* row0 = reinterpret_cast<PF_Pixel8*>((char*)input->data + y0 * input->rowbytes);
		PF_Pixel8* row1 = reinterpret_cast<PF_Pixel8*>((char*)input->data + y1 * input->rowbytes);
		p00 = &row0[x0];
		p01 = &row0[x1];
		p10 = &row1[x0];
		p11 = &row1[x1];
	}
	else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
		PF_Pixel16* row0 = reinterpret_cast<PF_Pixel16*>((char*)input->data + y0 * input->rowbytes);
		PF_Pixel16* row1 = reinterpret_cast<PF_Pixel16*>((char*)input->data + y1 * input->rowbytes);
		p00 = &row0[x0];
		p01 = &row0[x1];
		p10 = &row1[x0];
		p11 = &row1[x1];
	}
	else {
		PF_PixelFloat* row0 = reinterpret_cast<PF_PixelFloat*>((char*)input->data + y0 * input->rowbytes);
		PF_PixelFloat* row1 = reinterpret_cast<PF_PixelFloat*>((char*)input->data + y1 * input->rowbytes);
		p00 = &row0[x0];
		p01 = &row0[x1];
		p10 = &row1[x0];
		p11 = &row1[x1];
	}

	PixelT result;

	if constexpr (std::is_same_v<PixelT, PF_Pixel8>) {
		result.alpha = static_cast<A_u_char>(
			(1 - x_frac) * (1 - y_frac) * p00->alpha +
			x_frac * (1 - y_frac) * p01->alpha +
			(1 - x_frac) * y_frac * p10->alpha +
			x_frac * y_frac * p11->alpha + 0.5f
			);
		result.red = static_cast<A_u_char>(
			(1 - x_frac) * (1 - y_frac) * p00->red +
			x_frac * (1 - y_frac) * p01->red +
			(1 - x_frac) * y_frac * p10->red +
			x_frac * y_frac * p11->red + 0.5f
			);
		result.green = static_cast<A_u_char>(
			(1 - x_frac) * (1 - y_frac) * p00->green +
			x_frac * (1 - y_frac) * p01->green +
			(1 - x_frac) * y_frac * p10->green +
			x_frac * y_frac * p11->green + 0.5f
			);
		result.blue = static_cast<A_u_char>(
			(1 - x_frac) * (1 - y_frac) * p00->blue +
			x_frac * (1 - y_frac) * p01->blue +
			(1 - x_frac) * y_frac * p10->blue +
			x_frac * y_frac * p11->blue + 0.5f
			);
	}
	else if constexpr (std::is_same_v<PixelT, PF_Pixel16>) {
		result.alpha = static_cast<A_u_short>(
			(1 - x_frac) * (1 - y_frac) * p00->alpha +
			x_frac * (1 - y_frac) * p01->alpha +
			(1 - x_frac) * y_frac * p10->alpha +
			x_frac * y_frac * p11->alpha + 0.5f
			);
		result.red = static_cast<A_u_short>(
			(1 - x_frac) * (1 - y_frac) * p00->red +
			x_frac * (1 - y_frac) * p01->red +
			(1 - x_frac) * y_frac * p10->red +
			x_frac * y_frac * p11->red + 0.5f
			);
		result.green = static_cast<A_u_short>(
			(1 - x_frac) * (1 - y_frac) * p00->green +
			x_frac * (1 - y_frac) * p01->green +
			(1 - x_frac) * y_frac * p10->green +
			x_frac * y_frac * p11->green + 0.5f
			);
		result.blue = static_cast<A_u_short>(
			(1 - x_frac) * (1 - y_frac) * p00->blue +
			x_frac * (1 - y_frac) * p01->blue +
			(1 - x_frac) * y_frac * p10->blue +
			x_frac * y_frac * p11->blue + 0.5f
			);
	}
	else {
		result.alpha =
			(1 - x_frac) * (1 - y_frac) * p00->alpha +
			x_frac * (1 - y_frac) * p01->alpha +
			(1 - x_frac) * y_frac * p10->alpha +
			x_frac * y_frac * p11->alpha;
		result.red =
			(1 - x_frac) * (1 - y_frac) * p00->red +
			x_frac * (1 - y_frac) * p01->red +
			(1 - x_frac) * y_frac * p10->red +
			x_frac * y_frac * p11->red;
		result.green =
			(1 - x_frac) * (1 - y_frac) * p00->green +
			x_frac * (1 - y_frac) * p01->green +
			(1 - x_frac) * y_frac * p10->green +
			x_frac * y_frac * p11->green;
		result.blue =
			(1 - x_frac) * (1 - y_frac) * p00->blue +
			x_frac * (1 - y_frac) * p01->blue +
			(1 - x_frac) * y_frac * p10->blue +
			x_frac * y_frac * p11->blue;
	}

	return result;
}

template<typename PixelT>
static PF_Err
TransformFunc(
	void* refcon,
	A_long xL,
	A_long yL,
	PixelT* inP,
	PixelT* outP)
{
	PF_Err err = PF_Err_NONE;
	TransformInfo* tiP = reinterpret_cast<TransformInfo*>(refcon);

	if (!tiP) return PF_Err_BAD_CALLBACK_PARAM;

	PF_EffectWorld* input = tiP->input_worldP;

	float x_pos = static_cast<float>(tiP->x_pos) / 65536.0f;
	float y_pos = static_cast<float>(tiP->y_pos) / 65536.0f;

	float rotation_radians = -static_cast<float>(tiP->rotation) / 65536.0f * 3.14159265358979323846f / 180.0f;

	float scale = static_cast<float>(tiP->scale) / 100.0f;       

	float center_x = input->width / 2.0f;
	float center_y = input->height / 2.0f;

	float curr_x = static_cast<float>(xL);
	float curr_y = static_cast<float>(yL);

	float dx = curr_x - x_pos;
	float dy = curr_y - y_pos;

	dx /= scale;
	dy /= scale;

	float cos_theta = cosf(rotation_radians);
	float sin_theta = sinf(rotation_radians);
	float rotated_x = dx * cos_theta - dy * sin_theta;
	float rotated_y = dx * sin_theta + dy * cos_theta;

	float src_x = rotated_x + center_x;
	float src_y = rotated_y + center_y;

	*outP = SampleBilinear<PixelT>(input, src_x, src_y, tiP->x_tiles, tiP->y_tiles, tiP->mirror);

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

	TransformInfo tiP;
	AEFX_CLR_STRUCT(tiP);
	A_long linesL = 0;

	linesL = output->extent_hint.bottom - output->extent_hint.top;
	tiP.x_pos = params[TRANSFORMS_POSITION]->u.td.x_value;
	tiP.y_pos = params[TRANSFORMS_POSITION]->u.td.y_value;
	tiP.rotation = params[TRANSFORMS_ROTATION]->u.ad.value;
	tiP.scale = params[TRANSFORMS_SCALE]->u.fs_d.value;
	tiP.x_tiles = params[TRANSFORMS_X_TILES]->u.bd.value;
	tiP.y_tiles = params[TRANSFORMS_Y_TILES]->u.bd.value;
	tiP.mirror = params[TRANSFORMS_MIRROR]->u.bd.value;
	tiP.input_worldP = &params[TRANSFORMS_INPUT]->u.ld;

	PF_PixelFormat pixelFormat;
	PF_WorldSuite2* wsP = NULL;
	ERR(suites.Pica()->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&wsP));
	ERR(wsP->PF_GetPixelFormat(output, &pixelFormat));
	ERR(suites.Pica()->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2));

	switch (pixelFormat) {
	case PF_PixelFormat_ARGB128:
		ERR(suites.IterateFloatSuite1()->iterate(
			in_dataP,
			0,
			linesL,
			&params[TRANSFORMS_INPUT]->u.ld,
			NULL,
			(void*)&tiP,
			(PF_IteratePixelFloatFunc)TransformFunc<PF_PixelFloat>,
			output));
		break;

	case PF_PixelFormat_ARGB64:
		ERR(suites.Iterate16Suite1()->iterate(
			in_dataP,
			0,
			linesL,
			&params[TRANSFORMS_INPUT]->u.ld,
			NULL,
			(void*)&tiP,
			TransformFunc<PF_Pixel16>,
			output));
		break;

	case PF_PixelFormat_ARGB32:
	default:
		ERR(suites.Iterate8Suite1()->iterate(
			in_dataP,
			0,
			linesL,
			&params[TRANSFORMS_INPUT]->u.ld,
			NULL,
			(void*)&tiP,
			TransformFunc<PF_Pixel8>,
			output));
		break;
	}

	return err;
}

static void
DisposePreRenderData(
	void* pre_render_dataPV)
{
	if (pre_render_dataPV) {
		TransformParams* infoP = reinterpret_cast<TransformParams*>(pre_render_dataPV);
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

	TransformParams* infoP = reinterpret_cast<TransformParams*>(malloc(sizeof(TransformParams)));

	if (infoP) {
		PF_ParamDef cur_param;
		ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORMS_POSITION, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->x_pos = cur_param.u.td.x_value / 65536.0f;
		infoP->y_pos = cur_param.u.td.y_value / 65536.0f;

		ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORMS_ROTATION, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->rotation = cur_param.u.ad.value / 65536.0f;

		ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORMS_SCALE, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->scale = cur_param.u.fs_d.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORMS_X_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->x_tiles = cur_param.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORMS_Y_TILES, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->y_tiles = cur_param.u.bd.value;

		ERR(PF_CHECKOUT_PARAM(in_dataP, TRANSFORMS_MIRROR, in_dataP->current_time, in_dataP->time_step, in_dataP->time_scale, &cur_param));
		infoP->mirror = cur_param.u.bd.value;

		extraP->output->pre_render_data = infoP;
		extraP->output->delete_pre_render_data_func = DisposePreRenderData;

		ERR(extraP->cb->checkout_layer(in_dataP->effect_ref,
			TRANSFORMS_INPUT,
			TRANSFORMS_INPUT,
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
	float mXPos;
	float mYPos;
	float mRotation;
	float mScale;
	int mXTiles;
	int mYTiles;
	int mMirror;
} TransformKernelParams;

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
SmartRenderGPU(
	PF_InData* in_dataP,
	PF_OutData* out_dataP,
	PF_PixelFormat pixel_format,
	PF_EffectWorld* input_worldP,
	PF_EffectWorld* output_worldP,
	PF_SmartRenderExtra* extraP,
	TransformParams* infoP)
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

	TransformKernelParams transform_params;
	transform_params.mWidth = input_worldP->width;
	transform_params.mHeight = input_worldP->height;
	transform_params.mXPos = infoP->x_pos;
	transform_params.mYPos = infoP->y_pos;
	transform_params.mRotation = infoP->rotation;
	transform_params.mScale = infoP->scale;
	transform_params.mXTiles = infoP->x_tiles;
	transform_params.mYTiles = infoP->y_tiles;
	transform_params.mMirror = infoP->mirror;

	A_long src_row_bytes = input_worldP->rowbytes;
	A_long dst_row_bytes = output_worldP->rowbytes;

	transform_params.mSrcPitch = src_row_bytes / bytes_per_pixel;
	transform_params.mDstPitch = dst_row_bytes / bytes_per_pixel;
	transform_params.m16f = (pixel_format != PF_PixelFormat_GPU_BGRA128);

	if (!err && extraP->input->what_gpu == PF_GPU_Framework_OPENCL)
	{
		PF_Handle gpu_dataH = (PF_Handle)extraP->input->gpu_data;
		OpenCLGPUData* cl_gpu_dataP = reinterpret_cast<OpenCLGPUData*>(*gpu_dataH);

		cl_mem cl_src_mem = (cl_mem)src_mem;
		cl_mem cl_dst_mem = (cl_mem)dst_mem;

		cl_uint param_index = 0;

		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(cl_mem), &cl_src_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(cl_mem), &cl_dst_mem));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mSrcPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mDstPitch));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.m16f));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mWidth));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mHeight));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mXPos));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mYPos));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mRotation));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(float), &transform_params.mScale));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mXTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mYTiles));
		CL_ERR(clSetKernelArg(cl_gpu_dataP->transform_kernel, param_index++, sizeof(int), &transform_params.mMirror));

		size_t threadBlock[2] = { 16, 16 };
		size_t grid[2] = { RoundUp(transform_params.mWidth, threadBlock[0]), RoundUp(transform_params.mHeight, threadBlock[1]) };

		CL_ERR(clEnqueueNDRangeKernel(
			(cl_command_queue)device_info.command_queuePV,
			cl_gpu_dataP->transform_kernel,
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
		Transform_CUDA(
			(const float*)src_mem,
			(float*)dst_mem,
			transform_params.mSrcPitch,
			transform_params.mDstPitch,
			transform_params.m16f,
			transform_params.mWidth,
			transform_params.mHeight,
			transform_params.mXPos,
			transform_params.mYPos,
			transform_params.mRotation,
			transform_params.mScale,
			transform_params.mXTiles,
			transform_params.mYTiles,
			transform_params.mMirror);

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
			dx_gpu_data->mTransformShader,
			3);

		DX_ERR(shaderExecution.SetParamBuffer(&transform_params, sizeof(TransformKernelParams)));
		DX_ERR(shaderExecution.SetUnorderedAccessView(
			(ID3D12Resource*)dst_mem,
			transform_params.mHeight * dst_row_bytes));
		DX_ERR(shaderExecution.SetShaderResourceView(
			(ID3D12Resource*)src_mem,
			transform_params.mHeight * src_row_bytes));
		DX_ERR(shaderExecution.Execute((UINT)DivideRoundUp(transform_params.mWidth, 16), (UINT)DivideRoundUp(transform_params.mHeight, 16)));
	}
#endif
#if HAS_METAL
	else if (!err && extraP->input->what_gpu == PF_GPU_Framework_METAL)
	{
		ScopedAutoreleasePool pool;

		PF_Handle metal_handle = (PF_Handle)extraP->input->gpu_data;
		MetalGPUData* metal_dataP = reinterpret_cast<MetalGPUData*>(*metal_handle);

		id<MTLDevice> device = (id<MTLDevice>)device_info.devicePV;
		id<MTLBuffer> transform_param_buffer = [[device newBufferWithBytes : &transform_params
			length : sizeof(TransformKernelParams)
			options : MTLResourceStorageModeManaged]autorelease];

		id<MTLCommandQueue> queue = (id<MTLCommandQueue>)device_info.command_queuePV;
		id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
		id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
		id<MTLBuffer> src_metal_buffer = (id<MTLBuffer>)src_mem;
		id<MTLBuffer> dst_metal_buffer = (id<MTLBuffer>)dst_mem;

		MTLSize threadsPerGroup = { [metal_dataP->transform_pipeline threadExecutionWidth] , 16, 1 };
		MTLSize numThreadgroups = { DivideRoundUp(transform_params.mWidth, threadsPerGroup.width), DivideRoundUp(transform_params.mHeight, threadsPerGroup.height), 1 };

		[computeEncoder setComputePipelineState : metal_dataP->transform_pipeline] ;
		[computeEncoder setBuffer : src_metal_buffer offset : 0 atIndex : 0] ;
		[computeEncoder setBuffer : dst_metal_buffer offset : 0 atIndex : 1] ;
		[computeEncoder setBuffer : transform_param_buffer offset : 0 atIndex : 2] ;
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
	PF_SmartRenderExtra* extraP,
	TransformParams* infoP)
{
	PF_Err err = PF_Err_NONE;
	AEGP_SuiteHandler suites(in_data->pica_basicP);

	TransformInfo tiS;
	tiS.x_pos = static_cast<PF_Fixed>(infoP->x_pos * 65536.0f);
	tiS.y_pos = static_cast<PF_Fixed>(infoP->y_pos * 65536.0f);
	tiS.rotation = static_cast<PF_Fixed>(infoP->rotation * 65536.0f);
	tiS.scale = infoP->scale;
	tiS.x_tiles = infoP->x_tiles;
	tiS.y_tiles = infoP->y_tiles;
	tiS.mirror = infoP->mirror;
	tiS.input_worldP = input_worldP;

	switch (pixel_format) {
	case PF_PixelFormat_ARGB128:
		ERR(suites.IterateFloatSuite1()->iterate(
			in_data,
			0,
			output_worldP->height,
			input_worldP,
			NULL,
			(void*)&tiS,
			(PF_IteratePixelFloatFunc)TransformFunc<PF_PixelFloat>,
			output_worldP));
		break;

	case PF_PixelFormat_ARGB64:
		ERR(suites.Iterate16Suite1()->iterate(
			in_data,
			0,
			output_worldP->height,
			input_worldP,
			NULL,
			(void*)&tiS,
			TransformFunc<PF_Pixel16>,
			output_worldP));
		break;

	case PF_PixelFormat_ARGB32:
	default:
		ERR(suites.Iterate8Suite1()->iterate(
			in_data,
			0,
			output_worldP->height,
			input_worldP,
			NULL,
			(void*)&tiS,
			TransformFunc<PF_Pixel8>,
			output_worldP));
		break;
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

	TransformParams* infoP = reinterpret_cast<TransformParams*>(extraP->input->pre_render_data);

	if (infoP) {
		ERR((extraP->cb->checkout_layer_pixels(in_data->effect_ref, TRANSFORMS_INPUT, &input_worldP)));
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
		ERR2(extraP->cb->checkin_layer_pixels(in_data->effect_ref, TRANSFORMS_INPUT));
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
		"DKT Transform",
		"DKT Transform",
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