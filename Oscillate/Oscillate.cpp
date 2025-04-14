/**
 * Oscillate.cpp
 * After Effects plugin that simulates oscillating movement
 * Created by DKT with help from Unknown
 */

 #include "Oscillate.h"
 #include "AE_EffectCB.h"
 #include <cmath>
 #include <mutex>
 
  /**
   * Generates a triangle wave based on time input
   * @param t Time value (normalized)
   * @return Wave amplitude between -1 and 1
   */
 static PF_FpLong TriangleWave(PF_FpLong t)
 {
     // Shift phase by 0.75 and normalize to [0,1]
     t = fmod(t + 0.75, 1.0);
 
     // Handle negative values
     if (t < 0)
         t += 1.0;
 
     // Transform to triangle wave [-1,1]
     return (fabs(t - 0.5) - 0.25) * 4.0;
 }
 
 /**
  * About command handler - displays plugin information
  */
 static PF_Err About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
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
 
 /**
  * Global setup - registers plugin capabilities
  */
 static PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
 {
     // Set version information
     out_data->my_version = PF_VERSION(MAJOR_VERSION,
         MINOR_VERSION,
         BUG_VERSION,
         STAGE_VERSION,
         BUILD_VERSION);
 
     // Set plugin flags
     out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE |
         PF_OutFlag_PIX_INDEPENDENT |
         PF_OutFlag_I_EXPAND_BUFFER |
         PF_OutFlag_SEND_UPDATE_PARAMS_UI |
         PF_OutFlag_WIDE_TIME_INPUT |
         PF_OutFlag_NON_PARAM_VARY |
         PF_OutFlag_FORCE_RERENDER |
         PF_OutFlag_I_HAVE_EXTERNAL_DEPENDENCIES;
 
     out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER |
         PF_OutFlag2_FLOAT_COLOR_AWARE |
         PF_OutFlag2_SUPPORTS_THREADED_RENDERING |
         PF_OutFlag2_REVEALS_ZERO_ALPHA |
         PF_OutFlag2_PRESERVES_FULLY_OPAQUE_PIXELS |
         PF_OutFlag2_AUTOMATIC_WIDE_TIME_INPUT |
         PF_OutFlag2_I_MIX_GUID_DEPENDENCIES;
 
     return PF_Err_NONE;
 }
 
 /**
  * Updates UI state based on parameter values
  * Enables/disables the angle parameter when appropriate
  */
 static PF_Err UpdateParameterUI(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[])
 {
     PF_Err err = PF_Err_NONE;
     AEGP_SuiteHandler suites(in_data->pica_basicP);
 
     // Get the direction value (subtract 1 to convert from 1-based to 0-based)
     A_long direction = params[DIRECTION_SLIDER]->u.pd.value - 1;
 
     // Create a copy of the angle parameter to modify
     PF_ParamDef param_copy = *params[ANGLE_SLIDER];
 
     // Disable angle parameter when "Depth" (direction = 1) is selected
     if (direction == 1) {  // Depth selected
         param_copy.ui_flags |= PF_PUI_DISABLED;
     }
     else {
         param_copy.ui_flags &= ~PF_PUI_DISABLED;
     }
 
     // Update the UI
     ERR(suites.ParamUtilsSuite3()->PF_UpdateParamUI(in_data->effect_ref,
         ANGLE_SLIDER,
         &param_copy));
 
     return err;
 }
 
 /**
  * Sets up the parameters for the effect
  */
 static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
 {
     PF_Err err = PF_Err_NONE;
     PF_ParamDef def;
 
     // Clear parameter definition structure
     AEFX_CLR_STRUCT(def);
 
     // Direction parameter (Angle, Depth, or Orbit)
     PF_ADD_POPUP("Direction",
         3,                        // Number of choices
         1,                        // Default choice (1-based)
         "Angle|Depth|Orbit",      // Choices
         PF_ParamFlag_SUPERVISE,   // Enable parameter supervision
         DIRECTION_SLIDER);
 
     AEFX_CLR_STRUCT(def);
 
     // Angle parameter (in degrees)
     PF_ADD_FLOAT_SLIDERX("Angle",
         -3600.0,           // Min value
         3600.0,           // Max value
         0.0,              // Valid min
         360.0,            // Valid max
         45.0,             // Default value
         PF_Precision_TENTHS,
         PF_ParamFlag_SUPERVISE,
         0,
         ANGLE_SLIDER);
 
     AEFX_CLR_STRUCT(def);
 
     // Frequency parameter (oscillation speed)
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
 
     // Magnitude parameter (oscillation amount)
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
 
     // Wave type parameter (Sine or Triangle)
     PF_ADD_POPUP("Wave",
         2,                     // Number of choices
         1,                     // Default choice (1-based)
         "Sine|Triangle",       // Choices
         0,                     // Flags
         WAVE_TYPE_SLIDER);
 
     AEFX_CLR_STRUCT(def);
 
     // Phase parameter (wave offset)
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
 
     // Set total number of parameters
     out_data->num_params = RANDOMMOVE_NUM_PARAMS;
 
     return err;
 }
 
 // Check if there are any keyframes on the frequency parameter
 bool HasAnyFrequencyKeyframes(PF_InData* in_data)
 {
     PF_Err err = PF_Err_NONE;
     bool has_keyframes = false;
 
     AEGP_SuiteHandler suites(in_data->pica_basicP);
 
     // Get the effect reference
     AEGP_EffectRefH effect_ref = NULL;
     AEGP_StreamRefH stream_ref = NULL;
     A_long num_keyframes = 0;
 
     // Get the effect reference
     if (suites.PFInterfaceSuite1() && in_data->effect_ref) {
         AEGP_EffectRefH aegp_effect_ref = NULL;
         err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(NULL, in_data->effect_ref, &aegp_effect_ref);
 
         if (!err && aegp_effect_ref) {
             // Get the stream for the frequency parameter
             err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL,
                 aegp_effect_ref,
                 FREQUENCY_SLIDER,
                 &stream_ref);
 
             if (!err && stream_ref) {
                 // Check how many keyframes are on this stream
                 err = suites.KeyframeSuite3()->AEGP_GetStreamNumKFs(stream_ref, &num_keyframes);
 
                 // If there are any keyframes, set the flag
                 if (!err && num_keyframes > 0) {
                     has_keyframes = true;
                 }
 
                 // Dispose of the stream reference
                 suites.StreamSuite5()->AEGP_DisposeStream(stream_ref);
             }
 
             // Dispose of the effect reference
             suites.EffectSuite4()->AEGP_DisposeEffect(aegp_effect_ref);
         }
     }
 
     return has_keyframes;
 }
 
 
 /**
  * Legacy rendering function for older AE versions
  */
 static PF_Err LegacyRender(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
 {
     PF_Err err = PF_Err_NONE;
     AEGP_SuiteHandler suites(in_data->pica_basicP);
 
     // Get parameter values
     RandomMoveInfo info;
     AEFX_CLR_STRUCT(info);
 
     info.direction = params[DIRECTION_SLIDER]->u.pd.value - 1;
     info.angle = params[ANGLE_SLIDER]->u.fs_d.value;
     info.frequency = params[FREQUENCY_SLIDER]->u.fs_d.value;
     info.magnitude = params[MAGNITUDE_SLIDER]->u.fs_d.value;
     info.wave_type = params[WAVE_TYPE_SLIDER]->u.pd.value - 1;
     info.phase = params[PHASE_SLIDER]->u.fs_d.value;
 
     // Check if there are any frequency keyframes
     bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);
 
     // Convert current time to seconds for wave calculations
     PF_FpLong current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;
 
     // If there are any frequency keyframes, always advance time by half a frame
     if (has_frequency_keyframes) {
         // Shift by half a frame in time units
         A_long time_shift = in_data->time_step / 2;
 
         // Create a new time value with the shift applied
         A_Time shifted_time;
         shifted_time.value = in_data->current_time + time_shift;
         shifted_time.scale = in_data->time_scale;
 
         // Convert to seconds for the calculation
         current_time = (PF_FpLong)shifted_time.value / (PF_FpLong)shifted_time.scale;
     }
 
     // Calculate angle in radians for direction vector
     PF_FpLong angleRad = info.angle * PF_PI / 180.0;
     PF_FpLong dx = -cos(angleRad);
     PF_FpLong dy = -sin(angleRad);
 
     // Calculate wave value based on time, frequency and phase
     PF_FpLong X;
     PF_FpLong m;
 
     // Calculate wave value (sine or triangle)
     if (info.wave_type == 0) {
         // Sine wave
         X = (info.frequency * 2.0 * current_time) + (info.phase * 2.0);
         m = sin(X * 3.14159);
     }
     else {
         // Triangle wave
         X = ((info.frequency * 2.0 * current_time) + (info.phase * 2.0)) / 2.0 + info.phase;
         m = TriangleWave(X);
     }
 
     // Initialize transformation values
     PF_FpLong offsetX = 0, offsetY = 0;
     PF_FpLong scale = 100.0;
 
     // Apply effect based on direction mode
     switch (info.direction) {
     case 0: // Angle mode - position offset only
         offsetX = dx * info.magnitude * m;
         offsetY = dy * info.magnitude * m;
         break;
 
     case 1: // Depth mode - scale only
         scale = 100.0 + (info.magnitude * m * 0.1);
         break;
 
     case 2: { // Orbit mode - position offset and scale
         offsetX = dx * info.magnitude * m;
         offsetY = dy * info.magnitude * m;
 
         // Calculate second wave with phase shift for scale
         PF_FpLong phaseShift = info.wave_type == 0 ? 0.25 : 0.125;
         PF_FpLong X2;
 
         if (info.wave_type == 0) {
             X2 = (info.frequency * 2.0 * current_time) + ((info.phase + phaseShift) * 2.0);
             m = sin(X2 * 3.14159);
         }
         else {
             X2 = ((info.frequency * 2.0 * current_time) + ((info.phase + phaseShift) * 2.0)) / 2.0 + (info.phase + phaseShift);
             m = TriangleWave(X2);
         }
         scale = 100.0 + (info.magnitude * m * 0.1);
         break;
     }
     }
 
     // Get source layer origin offset from in_data
     PF_FpLong source_origin_x = in_data->pre_effect_source_origin_x;
     PF_FpLong source_origin_y = in_data->pre_effect_source_origin_y;
 
     // Account for downsample factors if present
     PF_FpLong downsample_x = in_data->downsample_x.num / (PF_FpLong)in_data->downsample_x.den;
     PF_FpLong downsample_y = in_data->downsample_y.num / (PF_FpLong)in_data->downsample_y.den;
 
     // Account for output origin
     PF_FpLong output_origin_x = in_data->output_origin_x;
     PF_FpLong output_origin_y = in_data->output_origin_y;
 
     // Apply downsample factors to offsets if needed
     offsetX *= downsample_x;
     offsetY *= downsample_y;
 
     // For legacy rendering, we need to manually transform and copy pixels
     PF_Rect src_rect;
     PF_Rect dest_rect = output->extent_hint;
 
     // Calculate center point adjusted for source origin and output origin
     PF_FpLong centerX = (output->width / 2.0) + source_origin_x - output_origin_x;
     PF_FpLong centerY = (output->height / 2.0) + source_origin_y - output_origin_y;
 
     // Calculate source rectangle
     if (scale != 100.0) {
         // For scaled content
         PF_FpLong scaledWidth = (params[RANDOMMOVE_INPUT]->u.ld.width * 100.0) / scale;
         PF_FpLong scaledHeight = (params[RANDOMMOVE_INPUT]->u.ld.height * 100.0) / scale;
 
         src_rect.left = centerX - (scaledWidth / 2.0) - offsetX;
         src_rect.top = centerY - (scaledHeight / 2.0) - offsetY;
         src_rect.right = src_rect.left + scaledWidth;
         src_rect.bottom = src_rect.top + scaledHeight;
     }
     else {
         // For non-scaled content
         PF_FpLong halfWidth = params[RANDOMMOVE_INPUT]->u.ld.width / 2.0;
         PF_FpLong halfHeight = params[RANDOMMOVE_INPUT]->u.ld.height / 2.0;
 
         src_rect.left = centerX - halfWidth - offsetX;
         src_rect.top = centerY - halfHeight - offsetY;
         src_rect.right = src_rect.left + params[RANDOMMOVE_INPUT]->u.ld.width;
         src_rect.bottom = src_rect.top + params[RANDOMMOVE_INPUT]->u.ld.height;
     }
 
     // Clear output buffer
     PF_Pixel empty_pixel = { 0, 0, 0, 0 };
     ERR(PF_FILL(&empty_pixel, &dest_rect, output));
 
     // Use legacy transform to copy pixels
     if (!err) {
         ERR(suites.WorldTransformSuite1()->copy_hq(
             in_data->effect_ref,
             &params[RANDOMMOVE_INPUT]->u.ld,
             output,
             NULL,
             &src_rect));
     }
 
     return err;
 }
 
 static PF_Err SmartPreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
 {
     PF_Err err = PF_Err_NONE;
 
     // Initialize effect info structure
     RandomMoveInfo info;
     AEFX_CLR_STRUCT(info);
 
     PF_ParamDef param_copy;
     AEFX_CLR_STRUCT(param_copy);
 
     // Initialize max_result_rect to the current output request rect
     extra->output->max_result_rect = extra->input->output_request.rect;
 
     // Set result_rect to match the request rect exactly - no expansion
     extra->output->result_rect = extra->input->output_request.rect;
 
     // Get parameters (we still need these for rendering)
     ERR(PF_CHECKOUT_PARAM(in_data, DIRECTION_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
     if (!err) info.direction = param_copy.u.pd.value - 1;
 
     ERR(PF_CHECKOUT_PARAM(in_data, ANGLE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
     if (!err) info.angle = param_copy.u.fs_d.value;
 
     ERR(PF_CHECKOUT_PARAM(in_data, FREQUENCY_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
     if (!err) info.frequency = param_copy.u.fs_d.value;
 
     ERR(PF_CHECKOUT_PARAM(in_data, MAGNITUDE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
     if (!err) info.magnitude = param_copy.u.fs_d.value;
 
     ERR(PF_CHECKOUT_PARAM(in_data, WAVE_TYPE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
     if (!err) info.wave_type = param_copy.u.pd.value - 1;
 
     ERR(PF_CHECKOUT_PARAM(in_data, PHASE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
     if (!err) info.phase = param_copy.u.fs_d.value;
 
     // Check if there are any frequency keyframes
     bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);
 
     // Calculate time offset for animation
     A_long time_offset = 0;
     if (has_frequency_keyframes) {
         // Always use a half-frame time offset if there are any frequency keyframes
         time_offset = in_data->time_step / 2; // 0.5 frames in time units
     }
 
     // Checkout the input layer with the EXACT same request - no expansion
     PF_RenderRequest req = extra->input->output_request;
     req.preserve_rgb_of_zero_alpha = TRUE;
 
     // Checkout the input layer with our request (no expansion)
     PF_CheckoutResult checkout;
     ERR(extra->cb->checkout_layer(in_data->effect_ref,
         RANDOMMOVE_INPUT,
         RANDOMMOVE_INPUT,
         &req,
         in_data->current_time,
         in_data->time_step,
         in_data->time_scale,
         &checkout));
 
     if (!err) {
         // Create a structure similar to the one in the Motion Blur example
         struct {
             A_u_char has_frequency_keyframes;
             A_long time_offset;
             RandomMoveInfo info;
         } detection_data;
 
         detection_data.has_frequency_keyframes = has_frequency_keyframes ? 1 : 0;
         detection_data.time_offset = time_offset;
         detection_data.info = info;
 
         // Mix in the data for caching
         ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(detection_data), &detection_data));
 
         // Set output flags
         extra->output->solid = FALSE;
         extra->output->pre_render_data = NULL;
         extra->output->flags = 0; // Not using PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS
     }
 
     return err;
 }
 
 
 /**
  * Calculate wave value (sine or triangle)
  */
 static PF_FpLong CalculateWaveValue(int wave_type, PF_FpLong frequency, PF_FpLong current_time, PF_FpLong phase)
 {
     PF_FpLong X, m;
 
     if (wave_type == 0) {
         // Sine wave
         X = (frequency * 2.0 * current_time) + (phase * 2.0);
         m = sin(X * 3.14159);
     }
     else {
         // Triangle wave
         X = ((frequency * 2.0 * current_time) + (phase * 2.0)) / 2.0 + phase;
         m = TriangleWave(X);
     }
 
     return m;
 }
 
 /**
  * Smart Render function - performs the actual effect rendering
  */
 static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra)
 {
     PF_Err err = PF_Err_NONE;
     AEGP_SuiteHandler suites(in_data->pica_basicP);
 
     // Use thread_local to ensure each thread has its own render data
     thread_local ThreadRenderData render_data;
     AEFX_CLR_STRUCT(render_data);
     render_data.in_data = in_data;
 
     // Checkout input layer pixels
     ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, RANDOMMOVE_INPUT, &render_data.input_worldP));
     if (!err) {
         // Checkout output buffer
         ERR(extra->cb->checkout_output(in_data->effect_ref, &render_data.output_worldP));
     }
 
     if (!err && render_data.input_worldP && render_data.output_worldP) {
         PF_ParamDef param_copy;
         AEFX_CLR_STRUCT(param_copy);
 
         // Get all effect parameters
         // Direction parameter (Angle, Depth, or Orbit)
         ERR(PF_CHECKOUT_PARAM(in_data, DIRECTION_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
         if (!err) render_data.info.direction = param_copy.u.pd.value - 1;
 
         // Angle parameter (in degrees)
         ERR(PF_CHECKOUT_PARAM(in_data, ANGLE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
         if (!err) render_data.info.angle = param_copy.u.fs_d.value;
 
         // Frequency parameter
         ERR(PF_CHECKOUT_PARAM(in_data, FREQUENCY_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
         if (!err) render_data.info.frequency = param_copy.u.fs_d.value;
 
         // Magnitude parameter
         ERR(PF_CHECKOUT_PARAM(in_data, MAGNITUDE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
         if (!err) render_data.info.magnitude = param_copy.u.fs_d.value;
 
         // Wave type parameter (Sine or Triangle)
         ERR(PF_CHECKOUT_PARAM(in_data, WAVE_TYPE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
         if (!err) render_data.info.wave_type = param_copy.u.pd.value - 1;
 
         // Phase parameter
         ERR(PF_CHECKOUT_PARAM(in_data, PHASE_SLIDER, in_data->current_time, in_data->time_step, in_data->time_scale, &param_copy));
         if (!err) render_data.info.phase = param_copy.u.fs_d.value;
 
         if (!err) {
             // Check if there are any frequency keyframes
             bool has_frequency_keyframes = HasAnyFrequencyKeyframes(in_data);
 
             // Convert current time to seconds for wave calculations
             render_data.current_time = (PF_FpLong)in_data->current_time / (PF_FpLong)in_data->time_scale;
 
             // If there are any frequency keyframes, always advance time by half a frame
             if (has_frequency_keyframes) {
                 // Shift by half a frame in time units
                 A_long time_shift = in_data->time_step / 2;
 
                 // Create a new time value with the shift applied
                 A_Time shifted_time;
                 shifted_time.value = in_data->current_time + time_shift;
                 shifted_time.scale = in_data->time_scale;
 
                 // Convert to seconds for the calculation
                 render_data.current_time = (PF_FpLong)shifted_time.value / (PF_FpLong)shifted_time.scale;
             }
 
             // Calculate angle in radians for direction vector
             PF_FpLong angleRad = render_data.info.angle * PF_PI / 180.0;
             PF_FpLong dx = -cos(angleRad);
             PF_FpLong dy = -sin(angleRad);
 
             // Calculate wave value
             PF_FpLong m = CalculateWaveValue(render_data.info.wave_type,
                 render_data.info.frequency,
                 render_data.current_time,
                 render_data.info.phase);
 
             // Initialize transformation values
             PF_FpLong offsetX = 0, offsetY = 0;
             PF_FpLong scale = 100.0;
 
             // Apply effect based on direction mode
             switch (render_data.info.direction) {
             case 0: // Angle mode - position offset only
                 offsetX = dx * render_data.info.magnitude * m;
                 offsetY = dy * render_data.info.magnitude * m;
                 break;
 
             case 1: // Depth mode - scale only
                 scale = 100.0 + (render_data.info.magnitude * m * 0.1);
                 break;
 
             case 2: { // Orbit mode - position offset and scale
                 offsetX = dx * render_data.info.magnitude * m;
                 offsetY = dy * render_data.info.magnitude * m;
 
                 // Calculate second wave with phase shift for scale
                 PF_FpLong phaseShift = render_data.info.wave_type == 0 ? 0.25 : 0.125;
                 m = CalculateWaveValue(render_data.info.wave_type,
                     render_data.info.frequency,
                     render_data.current_time,
                     render_data.info.phase + phaseShift);
 
                 scale = 100.0 + (render_data.info.magnitude * m * 0.1);
                 break;
             }
             }
 
             // Clear output buffer before copying
             PF_Pixel empty_pixel = { 0, 0, 0, 0 };
             ERR(suites.FillMatteSuite2()->fill(
                 in_data->effect_ref,
                 &empty_pixel,
                 NULL,
                 render_data.output_worldP));
 
             if (!err) {
                 // Calculate the source rectangle based on scale and offset
                 PF_Rect src_rect;
 
                 // Get source layer origin offset from in_data
                 PF_FpLong source_origin_x = in_data->pre_effect_source_origin_x;
                 PF_FpLong source_origin_y = in_data->pre_effect_source_origin_y;
 
                 // Account for downsample factors if present
                 PF_FpLong downsample_x = in_data->downsample_x.num / (PF_FpLong)in_data->downsample_x.den;
                 PF_FpLong downsample_y = in_data->downsample_y.num / (PF_FpLong)in_data->downsample_y.den;
 
                 // Account for output origin
                 PF_FpLong output_origin_x = in_data->output_origin_x;
                 PF_FpLong output_origin_y = in_data->output_origin_y;
 
                 // Apply downsample factors to offsets if needed
                 offsetX *= downsample_x;
                 offsetY *= downsample_y;
 
                 // Calculate center point adjusted for source origin and output origin
                 PF_FpLong centerX = (render_data.output_worldP->width / 2.0) + source_origin_x - output_origin_x;
                 PF_FpLong centerY = (render_data.output_worldP->height / 2.0) + source_origin_y - output_origin_y;
 
                 if (scale != 100.0) {
                     // For scaled content
                     PF_FpLong scaledWidth = (render_data.input_worldP->width * 100.0) / scale;
                     PF_FpLong scaledHeight = (render_data.input_worldP->height * 100.0) / scale;
 
                     src_rect.left = centerX - (scaledWidth / 2.0) - offsetX;
                     src_rect.top = centerY - (scaledHeight / 2.0) - offsetY;
                     src_rect.right = src_rect.left + scaledWidth;
                     src_rect.bottom = src_rect.top + scaledHeight;
                 }
                 else {
                     // For non-scaled content
                     PF_FpLong halfWidth = render_data.input_worldP->width / 2.0;
                     PF_FpLong halfHeight = render_data.input_worldP->height / 2.0;
 
                     src_rect.left = centerX - halfWidth - offsetX;
                     src_rect.top = centerY - halfHeight - offsetY;
                     src_rect.right = src_rect.left + render_data.input_worldP->width;
                     src_rect.bottom = src_rect.top + render_data.input_worldP->height;
                 }
 
                 // Use high-quality copy for transformation
                 ERR(suites.WorldTransformSuite1()->copy_hq(
                     in_data->effect_ref,
                     render_data.input_worldP,
                     render_data.output_worldP,
                     NULL,  // No destination rect - use entire output buffer
                     &src_rect));
             }
         }
     }
 
     // Check in the input layer pixels
     if (render_data.input_worldP) {
         ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, RANDOMMOVE_INPUT));
     }
 
     return err;
 }
 
 
 /**
  * Plugin registration function
  */
 extern "C" DllExport
 PF_Err PluginDataEntryFunction(
     PF_PluginDataPtr inPtr,
     PF_PluginDataCB inPluginDataCallBackPtr,
     SPBasicSuite* inSPBasicSuitePtr,
     const char* inHostName,
     const char* inHostVersion)
 {
     PF_Err result = PF_Err_INVALID_CALLBACK;
 
     // Register the effect with After Effects
     result = PF_REGISTER_EFFECT(
         inPtr,
         inPluginDataCallBackPtr,
         "Oscillate",          // Effect name
         "DKT Oscillate",      // Match name - make sure this is unique
         "DKT Effects",    // Category
         AE_RESERVED_INFO
     );
 
     return result;
 }
 
 /**
  * Main entry point for the effect
  * Handles all command dispatching
  */
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
             err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra);
             break;
         case PF_Cmd_RENDER:
             // Fallback for older versions that don't support Smart Render
             err = LegacyRender(in_data, out_data, params, output);
             break;
         case PF_Cmd_USER_CHANGED_PARAM:
         case PF_Cmd_UPDATE_PARAMS_UI:
             err = UpdateParameterUI(in_data, out_data, params);
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