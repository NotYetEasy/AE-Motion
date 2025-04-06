#include "MotionBlur.h"
#include "AE_Macros.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectSuites.h"
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    bool has_motion_prev_curr;
    bool has_scale_change_prev_curr;
    bool has_rotation_prev_curr;
    bool has_motion_curr_next;
    bool has_scale_change_curr_next;
    bool has_rotation_curr_next;
    double motion_x_prev_curr;
    double motion_y_prev_curr;
    double motion_x_curr_next;
    double motion_y_curr_next;
    double scale_x_prev_curr;
    double scale_y_prev_curr;
    double scale_x_curr_next;
    double scale_y_curr_next;
    double rotation_prev_curr;
    double rotation_curr_next;
    bool position_enabled;
    bool scale_enabled;
    bool angle_enabled;
    double tune_value;
    PF_EffectWorld* input_world;
    float scale_velocity;
    AEGP_TwoDVal anchor_point;
} DetectionData;

static PF_Err GetLayerPosition(PF_InData* in_data, AEGP_LayerH layerH, const A_Time* time, AEGP_TwoDVal* position) {
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamSuite5* streamSuite = suites.StreamSuite5();
    if (!streamSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_StreamRefH posStreamH = NULL;
    err = streamSuite->AEGP_GetNewLayerStream(
        NULL,
        layerH,
        AEGP_LayerStream_POSITION,
        &posStreamH);

    if (err || !posStreamH) {
        return err;
    }

    AEGP_StreamValue2 streamValue;
    err = streamSuite->AEGP_GetNewStreamValue(
        NULL,
        posStreamH,
        AEGP_LTimeMode_LayerTime,
        time,
        false,
        &streamValue);

    if (!err) {
        AEGP_StreamType type;
        err = streamSuite->AEGP_GetStreamType(posStreamH, &type);

        if (!err) {
            if (type == AEGP_StreamType_ThreeD || type == AEGP_StreamType_ThreeD_SPATIAL) {
                position->x = streamValue.val.three_d.x;
                position->y = streamValue.val.three_d.y;
            }
            else {
                position->x = streamValue.val.two_d.x;
                position->y = streamValue.val.two_d.y;
            }
        }

        streamSuite->AEGP_DisposeStreamValue(&streamValue);
    }

    streamSuite->AEGP_DisposeStream(posStreamH);

    return err;
}

static PF_Err GetLayerScale(PF_InData* in_data, AEGP_LayerH layerH, const A_Time* time, AEGP_TwoDVal* scale) {
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamSuite5* streamSuite = suites.StreamSuite5();
    if (!streamSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_StreamRefH scaleStreamH = NULL;
    err = streamSuite->AEGP_GetNewLayerStream(
        NULL,
        layerH,
        AEGP_LayerStream_SCALE,
        &scaleStreamH);

    if (err || !scaleStreamH) {
        return err;
    }

    AEGP_StreamValue2 streamValue;
    err = streamSuite->AEGP_GetNewStreamValue(
        NULL,
        scaleStreamH,
        AEGP_LTimeMode_LayerTime,
        time,
        false,
        &streamValue);

    if (!err) {
        scale->x = streamValue.val.two_d.x;
        scale->y = streamValue.val.two_d.y;

        streamSuite->AEGP_DisposeStreamValue(&streamValue);
    }

    streamSuite->AEGP_DisposeStream(scaleStreamH);

    return err;
}

static PF_Err GetLayerRotation(PF_InData* in_data, AEGP_LayerH layerH, const A_Time* time, double* rotation) {
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamSuite5* streamSuite = suites.StreamSuite5();
    if (!streamSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_StreamRefH rotationStreamH = NULL;
    err = streamSuite->AEGP_GetNewLayerStream(
        NULL,
        layerH,
        AEGP_LayerStream_ROTATION,
        &rotationStreamH);

    if (err || !rotationStreamH) {
        return err;
    }

    AEGP_StreamValue2 streamValue;
    err = streamSuite->AEGP_GetNewStreamValue(
        NULL,
        rotationStreamH,
        AEGP_LTimeMode_LayerTime,
        time,
        false,
        &streamValue);

    if (!err) {
        *rotation = streamValue.val.one_d;
        streamSuite->AEGP_DisposeStreamValue(&streamValue);
    }

    streamSuite->AEGP_DisposeStream(rotationStreamH);

    return err;
}

static PF_Err GetLayerAnchorPoint(PF_InData* in_data, AEGP_LayerH layerH, const A_Time* time, AEGP_TwoDVal* anchor) {
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    AEGP_StreamSuite5* streamSuite = suites.StreamSuite5();
    if (!streamSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_StreamRefH anchorStreamH = NULL;
    err = streamSuite->AEGP_GetNewLayerStream(
        NULL,
        layerH,
        AEGP_LayerStream_ANCHORPOINT,
        &anchorStreamH);

    if (err || !anchorStreamH) {
        return err;
    }

    AEGP_StreamValue2 streamValue;
    err = streamSuite->AEGP_GetNewStreamValue(
        NULL,
        anchorStreamH,
        AEGP_LTimeMode_LayerTime,
        time,
        false,
        &streamValue);

    if (!err) {
        AEGP_StreamType type;
        err = streamSuite->AEGP_GetStreamType(anchorStreamH, &type);

        if (!err) {
            if (type == AEGP_StreamType_ThreeD || type == AEGP_StreamType_ThreeD_SPATIAL) {
                anchor->x = streamValue.val.three_d.x;
                anchor->y = streamValue.val.three_d.y;
            }
            else {
                anchor->x = streamValue.val.two_d.x;
                anchor->y = streamValue.val.two_d.y;
            }
        }

        streamSuite->AEGP_DisposeStreamValue(&streamValue);
    }

    streamSuite->AEGP_DisposeStream(anchorStreamH);

    return err;
}

static bool DetectMotionFromOtherEffects(PF_InData* in_data, double* motion_x, double* motion_y, double* rotation_angle, double* scale_x, double* scale_y, float* scale_velocity_out) {
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Initialize motion values
    *motion_x = 0;
    *motion_y = 0;
    *rotation_angle = 0;
    *scale_x = 0;
    *scale_y = 0;
    *scale_velocity_out = 0;

    // Get the layer handle for the current effect
    AEGP_PFInterfaceSuite1* pfInterfaceSuite = suites.PFInterfaceSuite1();
    if (!pfInterfaceSuite) {
        return false;
    }

    AEGP_LayerH layerH = NULL;
    PF_Err err = pfInterfaceSuite->AEGP_GetEffectLayer(in_data->effect_ref, &layerH);
    if (err || !layerH) {
        return false;
    }

    // Get our own effect reference in AEGP format
    AEGP_EffectRefH our_effectH = NULL;
    err = suites.PFInterfaceSuite1()->AEGP_GetNewEffectForEffect(NULL, in_data->effect_ref, &our_effectH);
    if (err || !our_effectH) {
        return false;
    }

    // Get installed key from our effect to compare with other effects
    AEGP_InstalledEffectKey our_installed_key;
    err = suites.EffectSuite4()->AEGP_GetInstalledKeyFromLayerEffect(our_effectH, &our_installed_key);

    // Get match name for our effect
    A_char our_match_name[AEGP_MAX_EFFECT_MATCH_NAME_SIZE + 1];
    if (!err) {
        err = suites.EffectSuite4()->AEGP_GetEffectMatchName(our_installed_key, our_match_name);
    }

    // Dispose of our effect reference as we don't need it anymore
    if (our_effectH) {
        suites.EffectSuite4()->AEGP_DisposeEffect(our_effectH);
        our_effectH = NULL;
    }

    if (err) {
        return false; // Couldn't get our effect information
    }

    // Get number of effects on this layer
    A_long num_effects = 0;
    err = suites.EffectSuite4()->AEGP_GetLayerNumEffects(layerH, &num_effects);
    if (err || num_effects <= 0) {
        return false;
    }

    // Time values for previous, current, and next frames
    A_Time prev_time, current_time, next_time;

    current_time.value = in_data->current_time;
    current_time.scale = in_data->time_scale;

    prev_time.scale = current_time.scale;
    prev_time.value = current_time.value - in_data->time_step;

    next_time.scale = current_time.scale;
    next_time.value = current_time.value + in_data->time_step;

    bool found_motion = false;

    // Loop through all effects on the layer
    for (A_long i = 0; i < num_effects; i++) {
        // Get the effect reference
        AEGP_EffectRefH effectH = NULL;
        err = suites.EffectSuite4()->AEGP_GetLayerEffectByIndex(NULL, layerH, i, &effectH);
        if (err || !effectH) {
            continue;
        }

        // Get installed key and match name for this effect
        AEGP_InstalledEffectKey installed_key;
        err = suites.EffectSuite4()->AEGP_GetInstalledKeyFromLayerEffect(effectH, &installed_key);

        A_char match_name[AEGP_MAX_EFFECT_MATCH_NAME_SIZE + 1];
        if (!err) {
            err = suites.EffectSuite4()->AEGP_GetEffectMatchName(installed_key, match_name);
        }

        // Skip if this is our current effect (comparing by match name)
        // This ensures we don't detect our own effect
        if (!err && strcmp(match_name, our_match_name) == 0) {
            suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
            continue;
        }


        // Check if this is the DKT Oscillate effect
        if (strstr(match_name, "DKT Oscillate")) {
            // Variables to store Oscillate parameters
            A_long direction = 0;  // 0=Angle, 1=Depth, 2=Orbit
            double angle = 45.0;   // Default is 45 degrees
            double frequency = 1.0;
            double magnitude = 10.0;
            A_long wave_type = 0;  // 0=Sine, 1=Triangle
            double phase = 0.0;

            // Get number of parameters
            A_long num_params = 0;
            err = suites.StreamSuite5()->AEGP_GetEffectNumParamStreams(effectH, &num_params);
            if (err) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            // Parameter indices for Oscillate effect
            // Skip parameter 0 which is usually the input layer
            // Parameters start from index 1
            const int DIRECTION_PARAM = 1;
            const int ANGLE_PARAM = 2;
            const int FREQUENCY_PARAM = 3;
            const int MAGNITUDE_PARAM = 4;
            const int WAVE_TYPE_PARAM = 5;
            const int PHASE_PARAM = 6;

            // Check if we have enough parameters
            if (num_params < 7) {
                suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
                continue;
            }

            // Get Direction parameter
            AEGP_StreamRefH streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, DIRECTION_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_CompTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    direction = (A_long)value.val.one_d - 1;  // Convert from 1-based to 0-based
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            // Get Angle parameter
            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, ANGLE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_CompTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    angle = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            // Get Frequency parameter
            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, FREQUENCY_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_CompTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    frequency = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            // Get Magnitude parameter
            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, MAGNITUDE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_CompTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    magnitude = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            // Get Wave Type parameter
            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, WAVE_TYPE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_CompTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    wave_type = (A_long)value.val.one_d - 1;  // Convert from 1-based to 0-based
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            // Get Phase parameter
            streamH = NULL;
            err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, PHASE_PARAM, &streamH);
            if (!err && streamH) {
                AEGP_StreamValue2 value;
                err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, streamH, AEGP_LTimeMode_CompTime,
                    &current_time, FALSE, &value);
                if (!err) {
                    phase = value.val.one_d;
                    suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                }
                suites.StreamSuite5()->AEGP_DisposeStream(streamH);
            }

            // Convert times to seconds
            double prev_time_secs = (double)prev_time.value / (double)prev_time.scale;
            double current_time_secs = (double)current_time.value / (double)current_time.scale;
            double next_time_secs = (double)next_time.value / (double)next_time.scale;

            // Calculate angle in radians for direction vector
            double angleRad = angle * M_PI / 180.0;
            double dx = -cos(angleRad);
            double dy = -sin(angleRad);

            // Function to calculate wave value
            auto calculateWaveValue = [](int wave_type, double frequency, double time, double phase) -> double {
                double X, m;

                if (wave_type == 0) {
                    // Sine wave
                    X = (frequency * 2.0 * time) + (phase * 2.0);
                    m = sin(X * M_PI);
                }
                else {
                    // Triangle wave
                    X = ((frequency * 2.0 * time) + (phase * 2.0)) / 2.0 + phase;

                    // Triangle wave calculation
                    double t = fmod(X + 0.75, 1.0);
                    if (t < 0) t += 1.0;
                    m = (fabs(t - 0.5) - 0.25) * 4.0;
                }

                return m;
                };

            // Calculate wave values for all three frames
            double prev_wave = calculateWaveValue(wave_type, frequency, prev_time_secs, phase);
            double current_wave = calculateWaveValue(wave_type, frequency, current_time_secs, phase);
            double next_wave = calculateWaveValue(wave_type, frequency, next_time_secs, phase);

            // Variables to store positions and scales for all frames
            double prev_offsetX = 0, prev_offsetY = 0, prev_scale = 100.0;
            double current_offsetX = 0, current_offsetY = 0, current_scale = 100.0;
            double next_offsetX = 0, next_offsetY = 0, next_scale = 100.0;

            // Calculate previous frame values
            switch (direction) {
            case 0: // Angle mode - position offset only
                prev_offsetX = dx * magnitude * prev_wave;
                prev_offsetY = dy * magnitude * prev_wave;
                break;

            case 1: // Depth mode - scale only
                prev_scale = 100.0 + (magnitude * prev_wave * 0.1);
                break;

            case 2: { // Orbit mode - position offset and scale
                prev_offsetX = dx * magnitude * prev_wave;
                prev_offsetY = dy * magnitude * prev_wave;

                // Calculate second wave with phase shift for scale
                double phaseShift = wave_type == 0 ? 0.25 : 0.125;
                double m_scale = calculateWaveValue(wave_type, frequency, prev_time_secs, phase + phaseShift);
                prev_scale = 100.0 + (magnitude * m_scale * 0.1);
                break;
            }
            }

            // Calculate current frame values
            switch (direction) {
            case 0: // Angle mode - position offset only
                current_offsetX = dx * magnitude * current_wave;
                current_offsetY = dy * magnitude * current_wave;
                break;

            case 1: // Depth mode - scale only
                current_scale = 100.0 + (magnitude * current_wave * 0.1);
                break;

            case 2: { // Orbit mode - position offset and scale
                current_offsetX = dx * magnitude * current_wave;
                current_offsetY = dy * magnitude * current_wave;

                // Calculate second wave with phase shift for scale
                double phaseShift = wave_type == 0 ? 0.25 : 0.125;
                double m_scale = calculateWaveValue(wave_type, frequency, current_time_secs, phase + phaseShift);
                current_scale = 100.0 + (magnitude * m_scale * 0.1);
                break;
            }
            }

            // Calculate next frame values
            switch (direction) {
            case 0: // Angle mode - position offset only
                next_offsetX = dx * magnitude * next_wave;
                next_offsetY = dy * magnitude * next_wave;
                break;

            case 1: // Depth mode - scale only
                next_scale = 100.0 + (magnitude * next_wave * 0.1);
                break;

            case 2: { // Orbit mode - position offset and scale
                next_offsetX = dx * magnitude * next_wave;
                next_offsetY = dy * magnitude * next_wave;

                // Calculate second wave with phase shift for scale
                double phaseShift = wave_type == 0 ? 0.25 : 0.125;
                double m_scale = calculateWaveValue(wave_type, frequency, next_time_secs, phase + phaseShift);
                next_scale = 100.0 + (magnitude * m_scale * 0.1);
                break;
            }
            }

            // Calculate motion as the maximum of (current-prev) and (next-current)
            double prev_to_current_x = current_offsetX - prev_offsetX;
            double prev_to_current_y = current_offsetY - prev_offsetY;
            double current_to_next_x = next_offsetX - current_offsetX;
            double current_to_next_y = next_offsetY - current_offsetY;

            // Calculate scale changes
            double prev_to_current_scale = current_scale - prev_scale;
            double current_to_next_scale = next_scale - current_scale;

            // Use the larger motion vector (by magnitude)
            double prev_to_current_magnitude = sqrt(prev_to_current_x * prev_to_current_x + prev_to_current_y * prev_to_current_y);
            double current_to_next_magnitude = sqrt(current_to_next_x * current_to_next_x + current_to_next_y * current_to_next_y);

            // Handle different modes appropriately
            if (direction == 0) { // Angle mode - position offset only
                // Select the larger motion
                if (current_to_next_magnitude > prev_to_current_magnitude) {
                    *motion_x += current_to_next_x;
                    *motion_y += current_to_next_y;
                }
                else {
                    *motion_x += prev_to_current_x;
                    *motion_y += prev_to_current_y;
                }

                // If there's any motion, set found_motion to true
                if (fabs(*motion_x) > 0.01 || fabs(*motion_y) > 0.01) {
                    found_motion = true;
                }
            }
            else if (direction == 1 || direction == 2) { // Depth mode or Orbit mode
                // For Depth and Orbit modes, we need to handle scale differently

                // If it's Orbit mode, also handle position
                if (direction == 2) {
                    // Select the larger motion
                    if (current_to_next_magnitude > prev_to_current_magnitude) {
                        *motion_x += current_to_next_x;
                        *motion_y += current_to_next_y;
                    }
                    else {
                        *motion_x += prev_to_current_x;
                        *motion_y += prev_to_current_y;
                    }

                    // If there's any motion, set found_motion to true
                    if (fabs(*motion_x) > 0.01 || fabs(*motion_y) > 0.01) {
                        found_motion = true;
                    }
                }

                // For scale, we need to calculate scale_velocity directly
                // Get the comp dimensions
                AEGP_CompH compH = NULL;
                err = suites.LayerSuite9()->AEGP_GetLayerParentComp(layerH, &compH);
                if (!err && compH) {
                    AEGP_ItemH itemH = NULL;
                    err = suites.CompSuite11()->AEGP_GetItemFromComp(compH, &itemH);

                    A_long width = 0, height = 0;
                    if (!err && itemH) {
                        err = suites.ItemSuite9()->AEGP_GetItemDimensions(itemH, &width, &height);

                        if (!err) {
                            // Select the larger scale change
                            double scaleChange = fabs(current_to_next_scale) > fabs(prev_to_current_scale) ?
                                current_to_next_scale : prev_to_current_scale;

                            // Convert percentage scale change to pixel dimensions
                            float layer_width = (float)width;
                            float layer_height = (float)height;

                            // Calculate sizes at different scales
                            float current_size = sqrt(pow(layer_width * (current_scale / 100.0f), 2) +
                                pow(layer_height * (current_scale / 100.0f), 2));

                            float prev_size = sqrt(pow(layer_width * (prev_scale / 100.0f), 2) +
                                pow(layer_height * (prev_scale / 100.0f), 2));

                            float next_size = sqrt(pow(layer_width * (next_scale / 100.0f), 2) +
                                pow(layer_height * (next_scale / 100.0f), 2));

                            // Calculate scale velocity based on the larger change
                            if (fabs(next_size - current_size) > fabs(current_size - prev_size)) {
                                *scale_velocity_out = next_size - current_size;
                            }
                            else {
                                *scale_velocity_out = current_size - prev_size;
                            }

                            // Apply the same multiplier as in DetectChanges
                            *scale_velocity_out *= 1.6f;

                            // If there's any scale velocity, set found_motion to true
                            if (fabs(*scale_velocity_out) > 0.01f) {
                                found_motion = true;
                            }
                        }
                    }
                }
            }
        }


        // Check if this is the DKT Random Displacement effect
        else if (strstr(match_name, "DKT Random Displacement")) {
            // Get the comp dimensions to use as a reference point
            AEGP_CompH compH = NULL;
            err = suites.LayerSuite9()->AEGP_GetLayerParentComp(layerH, &compH);
            if (!err && compH) {
                // Get the item from the comp
                AEGP_ItemH itemH = NULL;
                err = suites.CompSuite11()->AEGP_GetItemFromComp(compH, &itemH);

                // Use ItemSuite9 to get dimensions
                A_long width = 0, height = 0;
                if (!err && itemH) {
                    err = suites.ItemSuite9()->AEGP_GetItemDimensions(itemH, &width, &height);

                    if (!err) {
                        // Get the parameter streams
                        AEGP_StreamRefH magnitude_streamH = NULL;
                        AEGP_StreamRefH evolution_streamH = NULL;
                        AEGP_StreamRefH seed_streamH = NULL;
                        AEGP_StreamRefH scatter_streamH = NULL;

                        // Get parameter streams
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 1, &magnitude_streamH); // MAGNITUDE_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 2, &evolution_streamH); // EVOLUTION_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 3, &seed_streamH);      // SEED_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 4, &scatter_streamH);   // SCATTER_PARAM

                        // Get the layer's position
                        AEGP_TwoDVal layer_position = { 0, 0 };
                        A_Time current_time, next_time;
                        current_time.value = in_data->current_time;
                        current_time.scale = in_data->time_scale;
                        next_time.value = in_data->current_time + in_data->time_step;
                        next_time.scale = in_data->time_scale;

                        // Get the layer's position
                        err = GetLayerPosition(in_data, layerH, &current_time, &layer_position);
                        if (err) {
                            // Fall back to center of comp if position can't be retrieved
                            layer_position.x = width / 2.0;
                            layer_position.y = height / 2.0;
                        }

                        // Get values for current frame
                        double magnitude = 50.0, evolution = 0.0, seed = 0.0, scatter = 0.5;

                        if (magnitude_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, magnitude_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                magnitude = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (evolution_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, evolution_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                evolution = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (seed_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, seed_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                seed = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (scatter_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, scatter_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                scatter = value.val.one_d; // Use directly, don't scale
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        // Get values for next frame
                        double next_magnitude = magnitude, next_evolution = evolution,
                            next_seed = seed, next_scatter = scatter;

                        // Check if evolution has keyframes and get value at next frame
                        if (evolution_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, evolution_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_evolution = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        // Check for other parameters at next frame if needed
                        if (magnitude_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, magnitude_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_magnitude = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (seed_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, seed_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_seed = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (scatter_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, scatter_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_scatter = value.val.one_d; // Use directly, don't scale
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        // Calculate displacement for current frame
                        double dx_curr = 0, dy_curr = 0;

                        // Directly implement the noise calculation as in ComputeDisplacement
                        dx_curr = SimplexNoise::noise(
                            layer_position.x * scatter / 50.0 + seed * 54623.245,
                            -layer_position.y * scatter / 500.0,
                            evolution + seed * 49235.319798
                        );

                        dy_curr = SimplexNoise::noise(
                            -layer_position.x * scatter / 50.0,
                            -layer_position.y * scatter / 500.0 + seed * 8723.5647,
                            evolution + 7468.329 + seed * 19337.940385
                        );

                        // Apply magnitude (positive for both to match the implementation)
                        dx_curr *= magnitude;
                        dy_curr *= magnitude;

                        // Calculate displacement for next frame
                        double dx_next = 0, dy_next = 0;

                        // Directly implement the noise calculation for next frame
                        dx_next = SimplexNoise::noise(
                            layer_position.x * next_scatter / 50.0 + next_seed * 54623.245,
                            -layer_position.y * next_scatter / 500.0,
                            next_evolution + next_seed * 49235.319798
                        );

                        dy_next = SimplexNoise::noise(
                            -layer_position.x * next_scatter / 50.0,
                            -layer_position.y * next_scatter / 500.0 + next_seed * 8723.5647,
                            next_evolution + 7468.329 + next_seed * 19337.940385
                        );

                        // Apply magnitude (positive for both to match the implementation)
                        dx_next *= next_magnitude;
                        dy_next *= next_magnitude;

                        // Calculate the motion vector between frames
                        double motion_dx = dx_next - dx_curr;
                        double motion_dy = dy_next - dy_curr;

                        // Add to total motion
                        *motion_x += motion_dx;
                        *motion_y += motion_dy;

                        // If there's any motion, set found_motion to true
                        if (fabs(motion_dx) > 0.01 || fabs(motion_dy) > 0.01) {
                            found_motion = true;
                        }

                        // Dispose of streams
                        if (magnitude_streamH) suites.StreamSuite5()->AEGP_DisposeStream(magnitude_streamH);
                        if (evolution_streamH) suites.StreamSuite5()->AEGP_DisposeStream(evolution_streamH);
                        if (seed_streamH) suites.StreamSuite5()->AEGP_DisposeStream(seed_streamH);
                        if (scatter_streamH) suites.StreamSuite5()->AEGP_DisposeStream(scatter_streamH);
                    }
                }
            }
        }

        // Check if this is the DKT Auto-Shake effect
        else if (strstr(match_name, "DKT Auto-Shake")) {
            // Get the comp dimensions to use as a reference point
            AEGP_CompH compH = NULL;
            err = suites.LayerSuite9()->AEGP_GetLayerParentComp(layerH, &compH);
            if (!err && compH) {
                // Get the item from the comp
                AEGP_ItemH itemH = NULL;
                err = suites.CompSuite11()->AEGP_GetItemFromComp(compH, &itemH);

                // Use ItemSuite9 to get dimensions
                A_long width = 0, height = 0;
                if (!err && itemH) {
                    err = suites.ItemSuite9()->AEGP_GetItemDimensions(itemH, &width, &height);

                    if (!err) {
                        // Get the parameter streams
                        AEGP_StreamRefH magnitude_streamH = NULL;
                        AEGP_StreamRefH frequency_streamH = NULL;
                        AEGP_StreamRefH evolution_streamH = NULL;
                        AEGP_StreamRefH seed_streamH = NULL;
                        AEGP_StreamRefH angle_streamH = NULL;
                        AEGP_StreamRefH slack_streamH = NULL;
                        AEGP_StreamRefH zshake_streamH = NULL;

                        // Get parameter streams - based on the parameter indices from AutoShake.cpp
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 1, &magnitude_streamH);  // MAGNITUDE_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 2, &frequency_streamH);  // FREQUENCY_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 3, &evolution_streamH);  // EVOLUTION_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 4, &seed_streamH);       // SEED_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 5, &angle_streamH);      // ANGLE_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 6, &slack_streamH);      // SLACK_PARAM
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 7, &zshake_streamH);     // ZSHAKE_PARAM

                        // Get values for current frame
                        double magnitude = 50.0, frequency = 2.0, evolution = 0.0, seed = 0.0;
                        double angle = 45.0, slack = 0.25, zshake = 0.0;

                        if (magnitude_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, magnitude_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                magnitude = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (frequency_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, frequency_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                frequency = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (evolution_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, evolution_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                evolution = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (seed_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, seed_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                seed = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (angle_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, angle_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                angle = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (slack_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, slack_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                slack = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (zshake_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, zshake_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                zshake = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        // Get values for next frame
                        double next_magnitude = magnitude, next_frequency = frequency;
                        double next_evolution = evolution, next_seed = seed;
                        double next_angle = angle, next_slack = slack, next_zshake = zshake;

                        // Check if parameters have keyframes and get values at next frame
                        if (evolution_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, evolution_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_evolution = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (magnitude_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, magnitude_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_magnitude = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (frequency_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, frequency_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_frequency = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (seed_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, seed_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_seed = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (angle_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, angle_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_angle = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (slack_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, slack_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_slack = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (zshake_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, zshake_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_zshake = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        // Convert current time to seconds for both frames
                        double current_time_secs = (double)current_time.value / (double)current_time.scale;
                        double next_time_secs = (double)next_time.value / (double)next_time.scale;

                        // Calculate shake offsets for current frame - exactly as in CalculateShakeOffsets
                        double rx_curr = 0, ry_curr = 0, dz_curr = 0;

                        // Calculate angle in radians for direction vector
                        double angleRad = angle * (M_PI / 180.0);
                        double s = -sin(angleRad);
                        double c = -cos(angleRad);

                        // Calculate evolution value
                        double evolutionValue = evolution + frequency * current_time_secs;

                        // Generate noise values using SimplexNoise
                        double dx_curr = SimplexNoise::noise(evolutionValue, seed * 49235.319798);
                        double dy_curr = SimplexNoise::noise(evolutionValue + 7468.329, seed * 19337.940385);
                        dz_curr = SimplexNoise::noise(evolutionValue + 14192.277, seed * 71401.168533);

                        // Scale noise by parameters
                        dx_curr *= magnitude;
                        dy_curr *= magnitude * slack;
                        dz_curr *= zshake;

                        // Apply rotation to get final offset
                        rx_curr = dx_curr * c + dy_curr * s;
                        ry_curr = dx_curr * s - dy_curr * c;

                        // Calculate shake offsets for next frame
                        double rx_next = 0, ry_next = 0, dz_next = 0;

                        // Calculate angle in radians for direction vector (next frame)
                        double nextAngleRad = next_angle * (M_PI / 180.0);
                        double next_s = -sin(nextAngleRad);
                        double next_c = -cos(nextAngleRad);

                        // Calculate evolution value for next frame
                        double nextEvolutionValue = next_evolution + next_frequency * next_time_secs;

                        // Generate noise values using SimplexNoise for next frame
                        double dx_next = SimplexNoise::noise(nextEvolutionValue, next_seed * 49235.319798);
                        double dy_next = SimplexNoise::noise(nextEvolutionValue + 7468.329, next_seed * 19337.940385);
                        dz_next = SimplexNoise::noise(nextEvolutionValue + 14192.277, next_seed * 71401.168533);

                        // Scale noise by parameters for next frame
                        dx_next *= next_magnitude;
                        dy_next *= next_magnitude * next_slack;
                        dz_next *= next_zshake;

                        // Apply rotation to get final offset for next frame
                        rx_next = dx_next * next_c + dy_next * next_s;
                        ry_next = dx_next * next_s - dy_next * next_c;

                        // Calculate the motion vector between frames
                        double motion_dx = rx_next - rx_curr;
                        double motion_dy = ry_next - ry_curr;

                        // Add to total motion
                        *motion_x += motion_dx;
                        *motion_y += motion_dy;

                        // If there's any motion, set found_motion to true
                        if (fabs(motion_dx) > 0.01 || fabs(motion_dy) > 0.01) {
                            found_motion = true;
                        }

                        // Dispose of streams
                        if (magnitude_streamH) suites.StreamSuite5()->AEGP_DisposeStream(magnitude_streamH);
                        if (frequency_streamH) suites.StreamSuite5()->AEGP_DisposeStream(frequency_streamH);
                        if (evolution_streamH) suites.StreamSuite5()->AEGP_DisposeStream(evolution_streamH);
                        if (seed_streamH) suites.StreamSuite5()->AEGP_DisposeStream(seed_streamH);
                        if (angle_streamH) suites.StreamSuite5()->AEGP_DisposeStream(angle_streamH);
                        if (slack_streamH) suites.StreamSuite5()->AEGP_DisposeStream(slack_streamH);
                        if (zshake_streamH) suites.StreamSuite5()->AEGP_DisposeStream(zshake_streamH);
                    }
                }
            }
        }


        // Check if this is the DKT Swing effect
        else if (strstr(match_name, "DKT Swing")) {
            // Get the comp dimensions to use as a reference point
            AEGP_CompH compH = NULL;
            err = suites.LayerSuite9()->AEGP_GetLayerParentComp(layerH, &compH);
            if (!err && compH) {
                // Get the item from the comp
                AEGP_ItemH itemH = NULL;
                err = suites.CompSuite11()->AEGP_GetItemFromComp(compH, &itemH);

                // Use ItemSuite9 to get dimensions
                A_long width = 0, height = 0;
                if (!err && itemH) {
                    err = suites.ItemSuite9()->AEGP_GetItemDimensions(itemH, &width, &height);

                    if (!err) {
                        // Get the parameter streams based on the parameter indices from Swing.cpp
                        AEGP_StreamRefH frequency_streamH = NULL;
                        AEGP_StreamRefH angle1_streamH = NULL;
                        AEGP_StreamRefH angle2_streamH = NULL;
                        AEGP_StreamRefH phase_streamH = NULL;
                        AEGP_StreamRefH wave_type_streamH = NULL;

                        // Get parameter streams
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 1, &frequency_streamH);  // SWING_FREQ
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 2, &angle1_streamH);     // SWING_ANGLE1
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 3, &angle2_streamH);     // SWING_ANGLE2
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 4, &phase_streamH);      // SWING_PHASE
                        err = suites.StreamSuite5()->AEGP_GetNewEffectStreamByIndex(NULL, effectH, 5, &wave_type_streamH);  // SWING_WAVE_TYPE

                        // Get values for current frame
                        double frequency = 2.0, angle1 = -30.0, angle2 = 30.0, phase = 0.0;
                        A_long wave_type = 0;

                        if (frequency_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, frequency_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                frequency = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (angle1_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, angle1_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                angle1 = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (angle2_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, angle2_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                angle2 = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (phase_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, phase_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                phase = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (wave_type_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, wave_type_streamH,
                                AEGP_LTimeMode_CompTime, &current_time, FALSE, &value);
                            if (!err) {
                                wave_type = (A_long)value.val.one_d - 1; // Convert from 1-based to 0-based
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        // Get values for next frame
                        double next_frequency = frequency, next_angle1 = angle1,
                            next_angle2 = angle2, next_phase = phase;
                        A_long next_wave_type = wave_type;

                        // Check if parameters have keyframes and get values at next frame
                        if (frequency_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, frequency_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_frequency = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (angle1_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, angle1_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_angle1 = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (angle2_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, angle2_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_angle2 = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (phase_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, phase_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_phase = value.val.one_d;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        if (wave_type_streamH) {
                            AEGP_StreamValue2 value;
                            err = suites.StreamSuite5()->AEGP_GetNewStreamValue(NULL, wave_type_streamH,
                                AEGP_LTimeMode_CompTime, &next_time, FALSE, &value);
                            if (!err) {
                                next_wave_type = (A_long)value.val.one_d - 1;
                                suites.StreamSuite5()->AEGP_DisposeStreamValue(&value);
                            }
                        }

                        // Convert current and next time to seconds
                        double current_time_secs = (double)current_time.value / (double)current_time.scale;
                        double next_time_secs = (double)next_time.value / (double)next_time.scale;

                        // Calculate the swing angle for current frame using the same algorithm as in Swing.cpp
                        // Calculate effective phase for current frame
                        double effective_phase = phase + (current_time_secs * frequency);

                        // Calculate modulation value based on wave type (0 = Sine, 1 = Triangle)
                        double m_curr;
                        if (wave_type == 0) {
                            // Sine wave
                            m_curr = sin(effective_phase * M_PI);
                        }
                        else {
                            // Triangle wave - using the same TriangleWave function as in Swing.cpp
                            double t = fmod(effective_phase / 2.0 + 0.75, 1.0);
                            if (t < 0) t += 1.0;
                            m_curr = (fabs(t - 0.5) - 0.25) * 4.0;
                        }

                        // Map modulation from -1...1 to 0...1
                        double t_curr = (m_curr + 1.0) / 2.0;

                        // Calculate final angle by interpolating between angle1 and angle2
                        double finalAngle_curr = angle1 + t_curr * (angle2 - angle1);

                        // Calculate the swing angle for next frame
                        // Calculate effective phase for next frame
                        double next_effective_phase = next_phase + (next_time_secs * next_frequency);

                        // Calculate modulation value based on wave type (0 = Sine, 1 = Triangle)
                        double m_next;
                        if (next_wave_type == 0) {
                            // Sine wave
                            m_next = sin(next_effective_phase * M_PI);
                        }
                        else {
                            // Triangle wave
                            double t = fmod(next_effective_phase / 2.0 + 0.75, 1.0);
                            if (t < 0) t += 1.0;
                            m_next = (fabs(t - 0.5) - 0.25) * 4.0;
                        }

                        // Map modulation from -1...1 to 0...1
                        double t_next = (m_next + 1.0) / 2.0;

                        // Calculate final angle by interpolating between angle1 and angle2
                        double finalAngle_next = next_angle1 + t_next * (next_angle2 - next_angle1);

                        // Calculate the angle difference between frames
                        double angle_diff = finalAngle_next - finalAngle_curr;

                        // For Swing, we want to report rotation, not position
                        *rotation_angle += angle_diff;

                        // Set position motion to zero to avoid double-counting
                        *motion_x = 0;
                        *motion_y = 0;

                        // If there's any rotation, set found_motion to true
                        if (fabs(angle_diff) > 0.1) {
                            found_motion = true;
                        }

                        // Dispose of streams
                        if (frequency_streamH) suites.StreamSuite5()->AEGP_DisposeStream(frequency_streamH);
                        if (angle1_streamH) suites.StreamSuite5()->AEGP_DisposeStream(angle1_streamH);
                        if (angle2_streamH) suites.StreamSuite5()->AEGP_DisposeStream(angle2_streamH);
                        if (phase_streamH) suites.StreamSuite5()->AEGP_DisposeStream(phase_streamH);
                        if (wave_type_streamH) suites.StreamSuite5()->AEGP_DisposeStream(wave_type_streamH);
                    }
                }
            }
        }




        // Dispose of effect reference
        suites.EffectSuite4()->AEGP_DisposeEffect(effectH);
    }

    return found_motion;
}



static PF_Err DetectChanges(
    PF_InData* in_data,
    double* motion_x_prev_curr, double* motion_y_prev_curr,
    double* motion_x_curr_next, double* motion_y_curr_next,
    double* scale_x_prev_curr, double* scale_y_prev_curr,
    double* scale_x_curr_next, double* scale_y_curr_next,
    double* rotation_prev_curr, double* rotation_curr_next,
    bool* has_motion_prev_curr, bool* has_motion_curr_next,
    bool* has_scale_change_prev_curr, bool* has_scale_change_curr_next,
    bool* has_rotation_prev_curr, bool* has_rotation_curr_next,
    float* scale_velocity,
    AEGP_TwoDVal* anchor_point
)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    // Initialize values
    *motion_x_prev_curr = *motion_y_prev_curr = 0;
    *motion_x_curr_next = *motion_y_curr_next = 0;
    *scale_x_prev_curr = *scale_y_prev_curr = 0;
    *scale_x_curr_next = *scale_y_curr_next = 0;
    *rotation_prev_curr = *rotation_curr_next = 0;
    *has_motion_prev_curr = *has_motion_curr_next = false;
    *has_scale_change_prev_curr = *has_scale_change_curr_next = false;
    *has_rotation_prev_curr = *has_rotation_curr_next = false;
    *scale_velocity = 0.0f;
    anchor_point->x = anchor_point->y = 0;

    // Get the layer handle for the current effect
    AEGP_PFInterfaceSuite1* pfInterfaceSuite = suites.PFInterfaceSuite1();
    if (!pfInterfaceSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    AEGP_LayerH layerH = NULL;
    PF_Err err = pfInterfaceSuite->AEGP_GetEffectLayer(in_data->effect_ref, &layerH);
    if (err || !layerH) {
        return err;
    }

    // Get the comp handle for this layer
    AEGP_CompH compH = NULL;
    AEGP_LayerSuite9* layerSuite = suites.LayerSuite9();
    if (!layerSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    err = layerSuite->AEGP_GetLayerParentComp(layerH, &compH);
    if (err || !compH) {
        return err;
    }

    // Get the source item for this layer
    AEGP_ItemH itemH = NULL;
    err = layerSuite->AEGP_GetLayerSourceItem(layerH, &itemH);
    if (err || !itemH) {
        return err;
    }

    // Get the item dimensions
    A_long width = 0, height = 0;
    AEGP_ItemSuite9* itemSuite = suites.ItemSuite9();
    if (!itemSuite) {
        return PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    err = itemSuite->AEGP_GetItemDimensions(itemH, &width, &height);
    if (err) {
        return err;
    }

    // Define previous, current, and next times
    A_Time prev_time, current_time, next_time;

    current_time.value = in_data->current_time;
    current_time.scale = in_data->time_scale;

    prev_time.scale = in_data->time_scale;
    prev_time.value = in_data->current_time - in_data->time_step;

    next_time.scale = in_data->time_scale;
    next_time.value = in_data->current_time + in_data->time_step;

    // Get position values for all three frames
    AEGP_TwoDVal prev_pos, curr_pos, next_pos;
    err = GetLayerPosition(in_data, layerH, &prev_time, &prev_pos);
    if (!err) {
        err = GetLayerPosition(in_data, layerH, &current_time, &curr_pos);
    }
    if (!err) {
        err = GetLayerPosition(in_data, layerH, &next_time, &next_pos);
    }

    // Get scale values for all three frames
    AEGP_TwoDVal prev_scale, curr_scale, next_scale;
    err = GetLayerScale(in_data, layerH, &prev_time, &prev_scale);
    if (!err) {
        err = GetLayerScale(in_data, layerH, &current_time, &curr_scale);
    }
    if (!err) {
        err = GetLayerScale(in_data, layerH, &next_time, &next_scale);
    }

    // Get rotation values for all three frames
    double prev_rotation = 0, curr_rotation = 0, next_rotation = 0;
    err = GetLayerRotation(in_data, layerH, &prev_time, &prev_rotation);
    if (!err) {
        err = GetLayerRotation(in_data, layerH, &current_time, &curr_rotation);
    }
    if (!err) {
        err = GetLayerRotation(in_data, layerH, &next_time, &next_rotation);
    }

    // Get anchor point for current frame
    err = GetLayerAnchorPoint(in_data, layerH, &current_time, anchor_point);

    if (err) {
        return err;
    }

    // Calculate position changes (prev to current, current to next)
    *motion_x_prev_curr = curr_pos.x - prev_pos.x;
    *motion_y_prev_curr = curr_pos.y - prev_pos.y;
    *motion_x_curr_next = next_pos.x - curr_pos.x;
    *motion_y_curr_next = next_pos.y - curr_pos.y;

    // Calculate scale changes (prev to current, current to next)
    *scale_x_prev_curr = curr_scale.x - prev_scale.x;
    *scale_y_prev_curr = curr_scale.y - prev_scale.y;
    *scale_x_curr_next = next_scale.x - curr_scale.x;
    *scale_y_curr_next = next_scale.y - curr_scale.y;

    // Calculate rotation changes (prev to current, current to next)
    *rotation_prev_curr = curr_rotation - prev_rotation;
    *rotation_curr_next = next_rotation - curr_rotation;

    // Normalize rotation changes to -180 to +180 degrees
    if (*rotation_prev_curr > 180) {
        *rotation_prev_curr -= 360;
    }
    else if (*rotation_prev_curr < -180) {
        *rotation_prev_curr += 360;
    }

    if (*rotation_curr_next > 180) {
        *rotation_curr_next -= 360;
    }
    else if (*rotation_curr_next < -180) {
        *rotation_curr_next -= 360;
    }

    // Use thresholds to determine if there are changes
    double scale_change_threshold = 0.1; // 0.1% change in scale
    double motion_threshold = 0.5;
    double rotation_threshold = 0.1; // 0.1 degrees change in rotation

    // Determine if there are changes between previous and current frames
    *has_motion_prev_curr = (sqrt((*motion_x_prev_curr) * (*motion_x_prev_curr) +
        (*motion_y_prev_curr) * (*motion_y_prev_curr)) > motion_threshold);
    *has_rotation_prev_curr = (fabs(*rotation_prev_curr) > rotation_threshold);

    // Determine if there are changes between current and next frames
    *has_motion_curr_next = (sqrt((*motion_x_curr_next) * (*motion_x_curr_next) +
        (*motion_y_curr_next) * (*motion_y_curr_next)) > motion_threshold);
    *has_rotation_curr_next = (fabs(*rotation_curr_next) > rotation_threshold);

    // Calculate scale velocity
    float layer_width = (float)width;
    float layer_height = (float)height;

    float current_size_x = layer_width * (curr_scale.x / 100.0f);
    float current_size_y = layer_height * (curr_scale.y / 100.0f);

    float prev_size_x, prev_size_y;

    bool prev_curr_scale_change = (fabs(*scale_x_prev_curr) > scale_change_threshold ||
        fabs(*scale_y_prev_curr) > scale_change_threshold);
    bool curr_next_scale_change = (fabs(*scale_x_curr_next) > scale_change_threshold ||
        fabs(*scale_y_curr_next) > scale_change_threshold);

    if (prev_curr_scale_change) {
        prev_size_x = layer_width * (prev_scale.x / 100.0f);
        prev_size_y = layer_height * (prev_scale.y / 100.0f);
    }
    else if (curr_next_scale_change) {
        prev_size_x = layer_width * (next_scale.x / 100.0f);
        prev_size_y = layer_height * (next_scale.y / 100.0f);
    }
    else {
        prev_size_x = current_size_x;
        prev_size_y = current_size_y;
    }

    float current_length = sqrt(current_size_x * current_size_x + current_size_y * current_size_y);
    float prev_length = sqrt(prev_size_x * prev_size_x + prev_size_y * prev_size_y);

    *scale_velocity = current_length - prev_length;
    *scale_velocity *= 1.6f;

    float scale_velocity_threshold = 0.01f;
    *has_scale_change_prev_curr = fabs(*scale_velocity) > scale_velocity_threshold;
    *has_scale_change_curr_next = fabs(*scale_velocity) > scale_velocity_threshold;

    return PF_Err_NONE;
}

template <typename PixelType>
static PF_Err ApplyMotionBlurGeneric(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    DetectionData* data = (DetectionData*)refcon;

    if (!data->position_enabled ||
        (!data->has_motion_prev_curr && !data->has_motion_curr_next)) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    double motion_x = 0, motion_y = 0;
    if (data->has_motion_prev_curr) {
        motion_x = data->motion_x_prev_curr;
        motion_y = data->motion_y_prev_curr;
    }
    else {
        motion_x = data->motion_x_curr_next;
        motion_y = data->motion_y_curr_next;
    }

    float max_dimension = fmax(data->input_world->width, data->input_world->height);
    float texel_size_x = 1.0f / max_dimension;
    float texel_size_y = 1.0f / max_dimension;

    float velocity_x = motion_x * data->tune_value * 0.7f;
    float velocity_y = motion_y * data->tune_value * 0.7f;

    float normalized_velocity_x = velocity_x / data->input_world->width;
    float normalized_velocity_y = velocity_y / data->input_world->height;

    float speed = sqrt((normalized_velocity_x * normalized_velocity_x) +
        (normalized_velocity_y * normalized_velocity_y)) / texel_size_x;

    int nSamples = (int)fmin(100, fmax(2, speed));

    if (nSamples <= 1) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    float aspect_ratio = (float)data->input_world->width / (float)data->input_world->height;

    float accumR, accumG, accumB, accumA;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else { // PF_Pixel8
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }

    for (int i = 1; i < nSamples; i++) {
        float offset_factor = ((float)i / (float)(nSamples - 1)) - 0.5f;
        int offset_x = (int)(velocity_x * offset_factor);
        int offset_y = (int)(velocity_y * offset_factor);

        offset_y = (int)(offset_y * aspect_ratio);

        int sample_x = xL - offset_x;
        int sample_y = yL - offset_y;

        sample_x = fmax(0, fmin(data->input_world->width - 1, sample_x));
        sample_y = fmax(0, fmin(data->input_world->height - 1, sample_y));

        if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
            PF_PixelFloat* input_pixels = (PF_PixelFloat*)data->input_world->data;
            PF_PixelFloat sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_PixelFloat) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
            PF_Pixel16* input_pixels = (PF_Pixel16*)data->input_world->data;
            PF_Pixel16 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel16) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else { // PF_Pixel8
            PF_Pixel8* input_pixels = (PF_Pixel8*)data->input_world->data;
            PF_Pixel8 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel8) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
    }

    float inv_nSamples = 1.0f / nSamples;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        outP->red = accumR * inv_nSamples;
        outP->green = accumG * inv_nSamples;
        outP->blue = accumB * inv_nSamples;
        outP->alpha = accumA * inv_nSamples;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        outP->red = static_cast<A_u_short>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_short>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_short>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_short>(accumA * inv_nSamples + 0.5f);
    }
    else { // PF_Pixel8
        outP->red = static_cast<A_u_char>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_char>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_char>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_char>(accumA * inv_nSamples + 0.5f);
    }

    return PF_Err_NONE;
}

static PF_Err RenderFunc8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return ApplyMotionBlurGeneric<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err RenderFunc16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return ApplyMotionBlurGeneric<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err RenderFuncFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return ApplyMotionBlurGeneric<PF_PixelFloat>(refcon, xL, yL, inP, outP);
}

template <typename PixelType>
static PF_Err ApplyScaleBlurGeneric(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    DetectionData* data = (DetectionData*)refcon;
    float scale_velocity = data->scale_velocity;

    if (!data->scale_enabled || fabs(scale_velocity) < 0.01f) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    float max_dimension = fmax(data->input_world->width, data->input_world->height);
    float texel_size_x = 1.0f / max_dimension;
    float texel_size_y = 1.0f / max_dimension;

    float cx = data->anchor_point.x / data->input_world->width;
    float cy = data->anchor_point.y / data->input_world->height;

    float norm_x = (float)xL / (float)data->input_world->width;
    float norm_y = (float)yL / (float)data->input_world->height;

    float v_x = norm_x - cx;
    float v_y = norm_y - cy;

    float aspect_ratio = (float)data->input_world->width / (float)data->input_world->height;
    v_y *= aspect_ratio;

    float speed = fabs(scale_velocity * data->tune_value) / 2.0f;
    speed *= sqrt(v_x * v_x + v_y * v_y);

    int nSamples = (int)fmin(100, fmax(2, speed));

    if (nSamples <= 1) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    float length_v = sqrt(v_x * v_x + v_y * v_y);
    float vnorm_x = 0, vnorm_y = 0;
    if (length_v > 0.0001f) {
        vnorm_x = v_x / length_v;
        vnorm_y = v_y / length_v;
    }

    vnorm_x *= texel_size_x * speed;
    vnorm_y *= texel_size_y * speed;

    float accumR, accumG, accumB, accumA;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else { // PF_Pixel8
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }

    for (int i = 1; i < nSamples; i++) {
        float offset_factor = ((float)i / (float)(nSamples - 1)) - 0.5f;
        float offset_x = vnorm_x * offset_factor;
        float offset_y = vnorm_y * offset_factor;

        offset_y *= aspect_ratio;

        int sample_x = xL - (int)(offset_x * data->input_world->width);
        int sample_y = yL - (int)(offset_y * data->input_world->height);

        sample_x = fmax(0, fmin(data->input_world->width - 1, sample_x));
        sample_y = fmax(0, fmin(data->input_world->height - 1, sample_y));

        if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
            PF_PixelFloat* input_pixels = (PF_PixelFloat*)data->input_world->data;
            PF_PixelFloat sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_PixelFloat) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
            PF_Pixel16* input_pixels = (PF_Pixel16*)data->input_world->data;
            PF_Pixel16 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel16) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else { // PF_Pixel8
            PF_Pixel8* input_pixels = (PF_Pixel8*)data->input_world->data;
            PF_Pixel8 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel8) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
    }

    float inv_nSamples = 1.0f / nSamples;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        outP->red = accumR * inv_nSamples;
        outP->green = accumG * inv_nSamples;
        outP->blue = accumB * inv_nSamples;
        outP->alpha = accumA * inv_nSamples;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        outP->red = static_cast<A_u_short>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_short>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_short>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_short>(accumA * inv_nSamples + 0.5f);
    }
    else { // PF_Pixel8
        outP->red = static_cast<A_u_char>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_char>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_char>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_char>(accumA * inv_nSamples + 0.5f);
    }

    return PF_Err_NONE;
}

static PF_Err ScaleBlurFunc8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return ApplyScaleBlurGeneric<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err ScaleBlurFunc16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return ApplyScaleBlurGeneric<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err ScaleBlurFuncFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return ApplyScaleBlurGeneric<PF_PixelFloat>(refcon, xL, yL, inP, outP);
}

template <typename PixelType>
static PF_Err ApplyAngleBlurGeneric(
    void* refcon,
    A_long xL,
    A_long yL,
    PixelType* inP,
    PixelType* outP)
{
    DetectionData* data = (DetectionData*)refcon;

    if (!data->angle_enabled ||
        (!data->has_rotation_prev_curr && !data->has_rotation_curr_next)) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    double rotation = 0;
    if (data->has_rotation_prev_curr) {
        rotation = data->rotation_prev_curr;
    }
    else {
        rotation = data->rotation_curr_next;
    }

    float angle_rad = fabs(rotation) * (float)M_PI / 180.0f;
    angle_rad *= data->tune_value * 0.7f;

    float max_dimension = fmax(data->input_world->width, data->input_world->height);
    float texel_size_x = 1.0f / max_dimension;
    float texel_size_y = 1.0f / max_dimension;

    float cx = data->anchor_point.x / data->input_world->width;
    float cy = data->anchor_point.y / data->input_world->height;

    float norm_x = (float)xL / (float)data->input_world->width;
    float norm_y = (float)yL / (float)data->input_world->height;

    float v_x = norm_x - cx;
    float v_y = norm_y - cy;

    float aspect_ratio = (float)data->input_world->width / (float)data->input_world->height;
    v_y *= aspect_ratio;

    float r = sqrt(v_x * v_x + v_y * v_y);
    float d = 2.0f * (float)M_PI * r;
    float l = d * angle_rad / (2.0f * (float)M_PI);

    float speed = l / texel_size_x;

    int nSamples = (int)fmin(100, fmax(2, speed));

    if (nSamples <= 1) {
        *outP = *inP;
        return PF_Err_NONE;
    }

    float accumR, accumG, accumB, accumA;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }
    else { // PF_Pixel8
        accumR = inP->red;
        accumG = inP->green;
        accumB = inP->blue;
        accumA = inP->alpha;
    }

    float amin = angle_rad / 2.0f;

    for (int i = 1; i < nSamples; i++) {
        float sampleAngle = -amin + angle_rad * (float)i / (float)nSamples;

        float s = sin(sampleAngle);
        float c = cos(sampleAngle);

        float rot_xx = c;
        float rot_xy = -s;
        float rot_yx = s;
        float rot_yy = c;

        float temp_x = v_x * rot_xx + v_y * rot_xy;
        float temp_y = v_x * rot_yx + v_y * rot_yy;

        temp_y = temp_y / aspect_ratio;

        float sample_norm_x = cx + temp_x;
        float sample_norm_y = cy + temp_y;

        int sample_x = (int)(sample_norm_x * data->input_world->width);
        int sample_y = (int)(sample_norm_y * data->input_world->height);

        sample_x = fmax(0, fmin(data->input_world->width - 1, sample_x));
        sample_y = fmax(0, fmin(data->input_world->height - 1, sample_y));

        if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
            PF_PixelFloat* input_pixels = (PF_PixelFloat*)data->input_world->data;
            PF_PixelFloat sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_PixelFloat) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
            PF_Pixel16* input_pixels = (PF_Pixel16*)data->input_world->data;
            PF_Pixel16 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel16) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
        else { // PF_Pixel8
            PF_Pixel8* input_pixels = (PF_Pixel8*)data->input_world->data;
            PF_Pixel8 sample = input_pixels[sample_y * data->input_world->rowbytes / sizeof(PF_Pixel8) + sample_x];

            accumR += sample.red;
            accumG += sample.green;
            accumB += sample.blue;
            accumA += sample.alpha;
        }
    }

    float inv_nSamples = 1.0f / nSamples;

    if constexpr (std::is_same_v<PixelType, PF_PixelFloat>) {
        outP->red = accumR * inv_nSamples;
        outP->green = accumG * inv_nSamples;
        outP->blue = accumB * inv_nSamples;
        outP->alpha = accumA * inv_nSamples;
    }
    else if constexpr (std::is_same_v<PixelType, PF_Pixel16>) {
        outP->red = static_cast<A_u_short>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_short>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_short>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_short>(accumA * inv_nSamples + 0.5f);
    }
    else { // PF_Pixel8
        outP->red = static_cast<A_u_char>(accumR * inv_nSamples + 0.5f);
        outP->green = static_cast<A_u_char>(accumG * inv_nSamples + 0.5f);
        outP->blue = static_cast<A_u_char>(accumB * inv_nSamples + 0.5f);
        outP->alpha = static_cast<A_u_char>(accumA * inv_nSamples + 0.5f);
    }

    return PF_Err_NONE;
}

static PF_Err AngleBlurFunc8(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel8* inP,
    PF_Pixel8* outP)
{
    return ApplyAngleBlurGeneric<PF_Pixel8>(refcon, xL, yL, inP, outP);
}

static PF_Err AngleBlurFunc16(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_Pixel16* inP,
    PF_Pixel16* outP)
{
    return ApplyAngleBlurGeneric<PF_Pixel16>(refcon, xL, yL, inP, outP);
}

static PF_Err AngleBlurFuncFloat(
    void* refcon,
    A_long xL,
    A_long yL,
    PF_PixelFloat* inP,
    PF_PixelFloat* outP)
{
    return ApplyAngleBlurGeneric<PF_PixelFloat>(refcon, xL, yL, inP, outP);
}

static PF_Err
About(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
        "%s v%d.%d\r%s",
        STR_NAME,
        MAJOR_VERSION,
        MINOR_VERSION,
        STR_DESCRIPTION);
    return PF_Err_NONE;
}

static PF_Err
GlobalSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    out_data->my_version = 524288; // Version 1.0
    out_data->out_flags = PF_OutFlag_DEEP_COLOR_AWARE;
    out_data->out_flags |= PF_OutFlag_WIDE_TIME_INPUT;
    out_data->out_flags |= PF_OutFlag_NON_PARAM_VARY;
    out_data->out_flags |= PF_OutFlag_USE_OUTPUT_EXTENT;
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_SMART_RENDER;
    out_data->out_flags2 |= PF_OutFlag2_FLOAT_COLOR_AWARE;
    out_data->out_flags2 |= PF_OutFlag2_I_MIX_GUID_DEPENDENCIES;
    out_data->out_flags2 |= PF_OutFlag2_REVEALS_ZERO_ALPHA;

    return PF_Err_NONE;
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

    PF_ADD_FLOAT_SLIDERX(STR_TUNE_NAME,
        0.00,
        4.00,
        0.00,
        4.00,
        1.00,
        PF_Precision_HUNDREDTHS,
        0,
        0,
        TUNE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Position",
        "Position",
        TRUE,
        0,
        POSITION_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Scale",
        "Scale",
        TRUE,
        0,
        SCALE_DISK_ID);

    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOX("Angle",
        "Angle",
        TRUE,
        0,
        ANGLE_DISK_ID);

    out_data->num_params = MOTIONBLUR_ANGLE + 1;

    return err;
}

static PF_Err
SequenceSetup(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    return PF_Err_NONE;
}

static PF_Err
SequenceSetdown(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_ParamDef* params[],
    PF_LayerDef* output)
{
    return PF_Err_NONE;
}

static PF_Err
SmartPreRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;

    PF_ParamDef position_param, scale_param, angle_param, tune_param;
    AEFX_CLR_STRUCT(position_param);
    AEFX_CLR_STRUCT(scale_param);
    AEFX_CLR_STRUCT(angle_param);
    AEFX_CLR_STRUCT(tune_param);

    ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_POSITION, in_data->current_time, in_data->time_step, in_data->time_scale, &position_param));
    ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_SCALE, in_data->current_time, in_data->time_step, in_data->time_scale, &scale_param));
    ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &angle_param));
    ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_TUNE, in_data->current_time, in_data->time_step, in_data->time_scale, &tune_param));

    // Calculate blur radius based on tune parameter
    A_long blur_radius = (A_long)ceil(tune_param.u.fs_d.value * 50.0);

    // Keep the original request - DO NOT MODIFY THIS
    PF_RenderRequest req = extra->input->output_request;
    req.preserve_rgb_of_zero_alpha = TRUE;

    // Checkout the input layer with the ORIGINAL request
    PF_CheckoutResult checkout_result;
    ERR(extra->cb->checkout_layer(in_data->effect_ref,
        MOTIONBLUR_INPUT,
        MOTIONBLUR_INPUT,
        &req,
        in_data->current_time,
        in_data->time_step,
        in_data->time_scale,
        &checkout_result));

    if (!err) {
        // IMPORTANT: Keep the result_rect exactly as what was requested
        // DO NOT MODIFY THIS - this is the actual visible area of the layer
        extra->output->result_rect = checkout_result.result_rect;

        // Set max_result_rect to be larger - this indicates the maximum area
        // the effect could possibly render, but doesn't change the layer size
        extra->output->max_result_rect.left = checkout_result.result_rect.left - blur_radius;
        extra->output->max_result_rect.top = checkout_result.result_rect.top - blur_radius;
        extra->output->max_result_rect.right = checkout_result.result_rect.right + blur_radius;
        extra->output->max_result_rect.bottom = checkout_result.result_rect.bottom + blur_radius;

        // Set the flag to indicate we return extra pixels
        extra->output->flags = PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS;

        // We're not fully opaque
        extra->output->solid = FALSE;
        extra->output->pre_render_data = NULL;

        double motion_x_prev_curr = 0, motion_y_prev_curr = 0;
        double motion_x_curr_next = 0, motion_y_curr_next = 0;
        double scale_x_prev_curr = 0, scale_y_prev_curr = 0;
        double scale_x_curr_next = 0, scale_y_curr_next = 0;
        double rotation_prev_curr = 0, rotation_curr_next = 0;
        bool has_motion_prev_curr = false, has_motion_curr_next = false;
        bool has_scale_change_prev_curr = false, has_scale_change_curr_next = false;
        bool has_rotation_prev_curr = false, has_rotation_curr_next = false;
        float scale_velocity = 0.0f;
        AEGP_TwoDVal anchor_point;

        DetectChanges(in_data,
            &motion_x_prev_curr, &motion_y_prev_curr,
            &motion_x_curr_next, &motion_y_curr_next,
            &scale_x_prev_curr, &scale_y_prev_curr,
            &scale_x_curr_next, &scale_y_curr_next,
            &rotation_prev_curr, &rotation_curr_next,
            &has_motion_prev_curr, &has_motion_curr_next,
            &has_scale_change_prev_curr, &has_scale_change_curr_next,
            &has_rotation_prev_curr, &has_rotation_curr_next,
            &scale_velocity, &anchor_point);

        // Detect additional motion from other effects
        double effect_motion_x = 0, effect_motion_y = 0;
        double effect_rotation = 0;
        double effect_scale_x = 0, effect_scale_y = 0;
        float effect_scale_velocity = 0.0f;
        bool has_effect_motion = DetectMotionFromOtherEffects(
            in_data,
            &effect_motion_x, &effect_motion_y,
            &effect_rotation,
            &effect_scale_x, &effect_scale_y,
            &effect_scale_velocity);

        // Add effect motion to our existing motion
        if (has_effect_motion) {
            motion_x_curr_next += effect_motion_x;
            motion_y_curr_next += effect_motion_y;
            rotation_curr_next += effect_rotation;
            scale_x_curr_next += effect_scale_x;
            scale_y_curr_next += effect_scale_y;

            if (fabs(effect_scale_velocity) > 0.01f) {
                scale_velocity = effect_scale_velocity;
            }

            if (!has_motion_curr_next) {
                double motion_magnitude = sqrt(motion_x_curr_next * motion_x_curr_next +
                    motion_y_curr_next * motion_y_curr_next);
                has_motion_curr_next = (motion_magnitude > 0.5);
            }

            if (!has_rotation_curr_next) {
                has_rotation_curr_next = (fabs(rotation_curr_next) > 0.1);
            }

            if (!has_scale_change_curr_next) {
                has_scale_change_curr_next = (fabs(scale_x_curr_next) > 0.1 ||
                    fabs(scale_y_curr_next) > 0.1 ||
                    fabs(scale_velocity) > 0.01f);
            }
        }

        // Store detection data for use in rendering
        struct {
            A_u_char has_motion_prev_curr;
            A_u_char has_scale_change_prev_curr;
            A_u_char has_rotation_prev_curr;
            A_u_char has_motion_curr_next;
            A_u_char has_scale_change_curr_next;
            A_u_char has_rotation_curr_next;
            A_u_char position_enabled;
            A_u_char scale_enabled;
            A_u_char angle_enabled;
            float motion_x_prev_curr;
            float motion_y_prev_curr;
            float motion_x_curr_next;
            float motion_y_curr_next;
            float scale_velocity;
            float anchor_x;
            float anchor_y;
            float tune_value;
            A_long blur_radius;
        } detection_data;

        detection_data.has_motion_prev_curr = has_motion_prev_curr ? 1 : 0;
        detection_data.has_scale_change_prev_curr = has_scale_change_prev_curr ? 1 : 0;
        detection_data.has_rotation_prev_curr = has_rotation_prev_curr ? 1 : 0;
        detection_data.has_motion_curr_next = has_motion_curr_next ? 1 : 0;
        detection_data.has_scale_change_curr_next = has_scale_change_curr_next ? 1 : 0;
        detection_data.has_rotation_curr_next = has_rotation_curr_next ? 1 : 0;
        detection_data.position_enabled = position_param.u.bd.value ? 1 : 0;
        detection_data.scale_enabled = scale_param.u.bd.value ? 1 : 0;
        detection_data.angle_enabled = angle_param.u.bd.value ? 1 : 0;
        detection_data.motion_x_prev_curr = motion_x_prev_curr;
        detection_data.motion_y_prev_curr = motion_y_prev_curr;
        detection_data.motion_x_curr_next = motion_x_curr_next;
        detection_data.motion_y_curr_next = motion_y_curr_next;
        detection_data.scale_velocity = scale_velocity;
        detection_data.anchor_x = anchor_point.x;
        detection_data.anchor_y = anchor_point.y;
        detection_data.tune_value = tune_param.u.fs_d.value;
        detection_data.blur_radius = blur_radius;

        ERR(extra->cb->GuidMixInPtr(in_data->effect_ref, sizeof(detection_data), &detection_data));
    }

    ERR(PF_CHECKIN_PARAM(in_data, &position_param));
    ERR(PF_CHECKIN_PARAM(in_data, &scale_param));
    ERR(PF_CHECKIN_PARAM(in_data, &angle_param));
    ERR(PF_CHECKIN_PARAM(in_data, &tune_param));

    return err;
}


static PF_Err
SmartRender(
    PF_InData* in_data,
    PF_OutData* out_data,
    PF_SmartRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    PF_EffectWorld* input_worldP = NULL;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, MOTIONBLUR_INPUT, &input_worldP));

    if (!err && input_worldP) {
        PF_EffectWorld* output_worldP = NULL;
        ERR(extra->cb->checkout_output(in_data->effect_ref, &output_worldP));

        if (!err && output_worldP) {
            output_worldP->origin_x = input_worldP->origin_x;
            output_worldP->origin_y = input_worldP->origin_y;

            PF_ParamDef position_param, scale_param, angle_param, tune_param;
            AEFX_CLR_STRUCT(position_param);
            AEFX_CLR_STRUCT(scale_param);
            AEFX_CLR_STRUCT(angle_param);
            AEFX_CLR_STRUCT(tune_param);

            ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_POSITION, in_data->current_time, in_data->time_step, in_data->time_scale, &position_param));
            ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_SCALE, in_data->current_time, in_data->time_step, in_data->time_scale, &scale_param));
            ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_ANGLE, in_data->current_time, in_data->time_step, in_data->time_scale, &angle_param));
            ERR(PF_CHECKOUT_PARAM(in_data, MOTIONBLUR_TUNE, in_data->current_time, in_data->time_step, in_data->time_scale, &tune_param));

            double motion_x_prev_curr = 0, motion_y_prev_curr = 0;
            double motion_x_curr_next = 0, motion_y_curr_next = 0;
            double scale_x_prev_curr = 0, scale_y_prev_curr = 0;
            double scale_x_curr_next = 0, scale_y_curr_next = 0;
            double rotation_prev_curr = 0, rotation_curr_next = 0;
            bool has_motion_prev_curr = false, has_motion_curr_next = false;
            bool has_scale_change_prev_curr = false, has_scale_change_curr_next = false;
            bool has_rotation_prev_curr = false, has_rotation_curr_next = false;
            float scale_velocity = 0.0f;
            AEGP_TwoDVal anchor_point;

            // Detect changes from transform properties
            DetectChanges(in_data,
                &motion_x_prev_curr, &motion_y_prev_curr,
                &motion_x_curr_next, &motion_y_curr_next,
                &scale_x_prev_curr, &scale_y_prev_curr,
                &scale_x_curr_next, &scale_y_curr_next,
                &rotation_prev_curr, &rotation_curr_next,
                &has_motion_prev_curr, &has_motion_curr_next,
                &has_scale_change_prev_curr, &has_scale_change_curr_next,
                &has_rotation_prev_curr, &has_rotation_curr_next,
                &scale_velocity, &anchor_point);

            // Detect additional motion from other effects
            double effect_motion_x = 0, effect_motion_y = 0;
            double effect_rotation = 0;
            double effect_scale_x = 0, effect_scale_y = 0;
            float effect_scale_velocity = 0.0f;
            bool has_effect_motion = DetectMotionFromOtherEffects(
                in_data,
                &effect_motion_x, &effect_motion_y,
                &effect_rotation,
                &effect_scale_x, &effect_scale_y,
                &effect_scale_velocity);

            // Add effect motion to our existing motion
            if (has_effect_motion) {
                // Add to the current-to-next motion since that's the most common case
                motion_x_curr_next += effect_motion_x;
                motion_y_curr_next += effect_motion_y;
                rotation_curr_next += effect_rotation;
                scale_x_curr_next += effect_scale_x;
                scale_y_curr_next += effect_scale_y;

                // This is the key change - directly update scale_velocity if it's significant
                if (fabs(effect_scale_velocity) > 0.01f) {
                    scale_velocity = effect_scale_velocity;
                }

                // Update has_motion flag if we now have motion
                if (!has_motion_curr_next) {
                    double motion_magnitude = sqrt(motion_x_curr_next * motion_x_curr_next +
                        motion_y_curr_next * motion_y_curr_next);
                    has_motion_curr_next = (motion_magnitude > 0.5); // Using the same threshold as in DetectChanges
                }

                // Update has_rotation flag if we now have rotation
                if (!has_rotation_curr_next) {
                    has_rotation_curr_next = (fabs(rotation_curr_next) > 0.1); // Using the same threshold as in DetectChanges
                }

                // Update has_scale_change flag if we now have scale change
                if (!has_scale_change_curr_next) {
                    has_scale_change_curr_next = (fabs(scale_x_curr_next) > 0.1 ||
                        fabs(scale_y_curr_next) > 0.1 ||
                        fabs(scale_velocity) > 0.01f);
                }
            }

            DetectionData data;
            data.has_motion_prev_curr = has_motion_prev_curr;
            data.has_scale_change_prev_curr = has_scale_change_prev_curr;
            data.has_rotation_prev_curr = has_rotation_prev_curr;
            data.has_motion_curr_next = has_motion_curr_next;
            data.has_scale_change_curr_next = has_scale_change_curr_next;
            data.has_rotation_curr_next = has_rotation_curr_next;
            data.motion_x_prev_curr = motion_x_prev_curr;
            data.motion_y_prev_curr = motion_y_prev_curr;
            data.motion_x_curr_next = motion_x_curr_next;
            data.motion_y_curr_next = motion_y_curr_next;
            data.scale_x_prev_curr = scale_x_prev_curr;
            data.scale_y_prev_curr = scale_y_prev_curr;
            data.scale_x_curr_next = scale_x_curr_next;
            data.scale_y_curr_next = scale_y_curr_next;
            data.rotation_prev_curr = rotation_prev_curr;
            data.rotation_curr_next = rotation_curr_next;
            data.position_enabled = position_param.u.bd.value;
            data.scale_enabled = scale_param.u.bd.value;
            data.angle_enabled = angle_param.u.bd.value;
            data.tune_value = tune_param.u.fs_d.value;
            data.input_world = input_worldP;
            data.scale_velocity = scale_velocity;
            data.anchor_point = anchor_point;

            // Copy the input to the output first, to ensure we have all pixels from the input
            ERR(suites.WorldTransformSuite1()->copy(
                in_data->effect_ref,
                input_worldP,
                output_worldP,
                NULL,
                NULL));

            PF_EffectWorld temp_world1, temp_world2;

            PF_WorldSuite2* world_suite = NULL;
            SPBasicSuite* basic_suite = in_data->pica_basicP;

            if (basic_suite) {
                basic_suite->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&world_suite);
            }

            if (world_suite) {
                PF_PixelFormat pixel_format;
                ERR(world_suite->PF_GetPixelFormat(input_worldP, &pixel_format));

                // Create temporary worlds with the same dimensions as the output world
                ERR(world_suite->PF_NewWorld(
                    in_data->effect_ref,
                    output_worldP->width,
                    output_worldP->height,
                    TRUE,
                    pixel_format,
                    &temp_world1));

                ERR(world_suite->PF_NewWorld(
                    in_data->effect_ref,
                    output_worldP->width,
                    output_worldP->height,
                    TRUE,
                    pixel_format,
                    &temp_world2));

                // Copy input to first temp world
                ERR(suites.WorldTransformSuite1()->copy(
                    in_data->effect_ref,
                    input_worldP,
                    &temp_world1,
                    NULL,
                    NULL));

                const double bytesPerPixel = static_cast<double>(input_worldP->rowbytes) /
                    static_cast<double>(input_worldP->width);

                PF_EffectWorld* current_result = &temp_world1;
                PF_EffectWorld* next_result = &temp_world2;

                if (data.position_enabled &&
                    (data.has_motion_prev_curr || data.has_motion_curr_next)) {

                    data.input_world = current_result;

                    if (bytesPerPixel >= 16.0) {
                        ERR(suites.IterateFloatSuite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            RenderFuncFloat,
                            next_result));
                    }
                    else if (bytesPerPixel >= 8.0) {
                        ERR(suites.Iterate16Suite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            RenderFunc16,
                            next_result));
                    }
                    else {
                        ERR(suites.Iterate8Suite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            RenderFunc8,
                            next_result));
                    }

                    PF_EffectWorld* temp = current_result;
                    current_result = next_result;
                    next_result = temp;
                }

                if (data.scale_enabled && (fabs(data.scale_velocity) > 0.01f)) {
                    data.input_world = current_result;

                    if (bytesPerPixel >= 16.0) {
                        ERR(suites.IterateFloatSuite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            ScaleBlurFuncFloat,
                            next_result));
                    }
                    else if (bytesPerPixel >= 8.0) {
                        ERR(suites.Iterate16Suite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            ScaleBlurFunc16,
                            next_result));
                    }
                    else {
                        ERR(suites.Iterate8Suite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            ScaleBlurFunc8,
                            next_result));
                    }

                    PF_EffectWorld* temp = current_result;
                    current_result = next_result;
                    next_result = temp;
                }

                if (data.angle_enabled &&
                    (data.has_rotation_prev_curr || data.has_rotation_curr_next)) {

                    data.input_world = current_result;

                    if (bytesPerPixel >= 16.0) {
                        ERR(suites.IterateFloatSuite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            AngleBlurFuncFloat,
                            next_result));
                    }
                    else if (bytesPerPixel >= 8.0) {
                        ERR(suites.Iterate16Suite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            AngleBlurFunc16,
                            next_result));
                    }
                    else {
                        ERR(suites.Iterate8Suite1()->iterate(
                            in_data,
                            0,
                            output_worldP->height,
                            current_result,
                            NULL, // Important: Use NULL instead of a specific area to allow rendering outside bounds
                            &data,
                            AngleBlurFunc8,
                            next_result));
                    }

                    PF_EffectWorld* temp = current_result;
                    current_result = next_result;
                    next_result = temp;
                }

                // Copy the final result to the output
                ERR(suites.WorldTransformSuite1()->copy(
                    in_data->effect_ref,
                    current_result,
                    output_worldP,
                    NULL,
                    NULL));

                ERR(world_suite->PF_DisposeWorld(in_data->effect_ref, &temp_world1));
                ERR(world_suite->PF_DisposeWorld(in_data->effect_ref, &temp_world2));

                if (basic_suite) {
                    basic_suite->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
                }
            }

            ERR(PF_CHECKIN_PARAM(in_data, &position_param));
            ERR(PF_CHECKIN_PARAM(in_data, &scale_param));
            ERR(PF_CHECKIN_PARAM(in_data, &angle_param));
            ERR(PF_CHECKIN_PARAM(in_data, &tune_param));
        }
    }

    if (input_worldP) {
        ERR(extra->cb->checkin_layer_pixels(in_data->effect_ref, MOTIONBLUR_INPUT));
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
    AEGP_SuiteHandler suites(in_data->pica_basicP);

    double motion_x_prev_curr = 0, motion_y_prev_curr = 0;
    double motion_x_curr_next = 0, motion_y_curr_next = 0;
    double scale_x_prev_curr = 0, scale_y_prev_curr = 0;
    double scale_x_curr_next = 0, scale_y_curr_next = 0;
    double rotation_prev_curr = 0, rotation_curr_next = 0;
    bool has_motion_prev_curr = false, has_motion_curr_next = false;
    bool has_scale_change_prev_curr = false, has_scale_change_curr_next = false;
    bool has_rotation_prev_curr = false, has_rotation_curr_next = false;
    float scale_velocity = 0.0f;
    AEGP_TwoDVal anchor_point;

    DetectChanges(in_data,
        &motion_x_prev_curr, &motion_y_prev_curr,
        &motion_x_curr_next, &motion_y_curr_next,
        &scale_x_prev_curr, &scale_y_prev_curr,
        &scale_x_curr_next, &scale_y_curr_next,
        &rotation_prev_curr, &rotation_curr_next,
        &has_motion_prev_curr, &has_motion_curr_next,
        &has_scale_change_prev_curr, &has_scale_change_curr_next,
        &has_rotation_prev_curr, &has_rotation_curr_next,
        &scale_velocity, &anchor_point);

    DetectionData data;
    data.has_motion_prev_curr = has_motion_prev_curr;
    data.has_scale_change_prev_curr = has_scale_change_prev_curr;
    data.has_rotation_prev_curr = has_rotation_prev_curr;
    data.has_motion_curr_next = has_motion_curr_next;
    data.has_scale_change_curr_next = has_scale_change_curr_next;
    data.has_rotation_curr_next = has_rotation_curr_next;
    data.motion_x_prev_curr = motion_x_prev_curr;
    data.motion_y_prev_curr = motion_y_prev_curr;
    data.motion_x_curr_next = motion_x_curr_next;
    data.motion_y_curr_next = motion_y_curr_next;
    data.rotation_prev_curr = rotation_prev_curr;
    data.rotation_curr_next = rotation_curr_next;
    data.position_enabled = params[MOTIONBLUR_POSITION]->u.bd.value;
    data.scale_enabled = params[MOTIONBLUR_SCALE]->u.bd.value;
    data.angle_enabled = params[MOTIONBLUR_ANGLE]->u.bd.value;
    data.tune_value = params[MOTIONBLUR_TUNE]->u.fs_d.value;
    data.scale_velocity = scale_velocity;
    data.anchor_point = anchor_point;

    PF_EffectWorld input_world;
    input_world.width = params[MOTIONBLUR_INPUT]->u.ld.width;
    input_world.height = params[MOTIONBLUR_INPUT]->u.ld.height;
    input_world.data = params[MOTIONBLUR_INPUT]->u.ld.data;
    input_world.rowbytes = params[MOTIONBLUR_INPUT]->u.ld.rowbytes;
    data.input_world = &input_world;

    ERR(suites.WorldTransformSuite1()->copy_hq(
        in_data->effect_ref,
        &params[MOTIONBLUR_INPUT]->u.ld,
        output,
        NULL,
        NULL));

    PF_EffectWorld temp_world1, temp_world2;

    PF_WorldSuite2* world_suite = NULL;
    SPBasicSuite* basic_suite = in_data->pica_basicP;

    if (basic_suite) {
        basic_suite->AcquireSuite(kPFWorldSuite, kPFWorldSuiteVersion2, (const void**)&world_suite);
    }

    if (world_suite) {
        PF_PixelFormat pixel_format;
        ERR(world_suite->PF_GetPixelFormat(&input_world, &pixel_format));

        ERR(world_suite->PF_NewWorld(
            in_data->effect_ref,
            output->width,
            output->height,
            TRUE,
            pixel_format,
            &temp_world1));

        ERR(world_suite->PF_NewWorld(
            in_data->effect_ref,
            output->width,
            output->height,
            TRUE,
            pixel_format,
            &temp_world2));

        ERR(suites.WorldTransformSuite1()->copy(
            in_data->effect_ref,
            &input_world,
            &temp_world1,
            NULL,
            NULL));

        A_long linesL = output->height;
        PF_Rect area = { 0, 0, output->width, output->height };

        const double bytesPerPixel = static_cast<double>(input_world.rowbytes) /
            static_cast<double>(input_world.width);

        PF_EffectWorld* current_result = &temp_world1;
        PF_EffectWorld* next_result = &temp_world2;

        if (data.position_enabled &&
            (data.has_motion_prev_curr || data.has_motion_curr_next)) {

            data.input_world = current_result;

            if (bytesPerPixel >= 16.0) {
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    RenderFuncFloat,
                    next_result));
            }
            else if (bytesPerPixel >= 8.0) {
                ERR(suites.Iterate16Suite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    RenderFunc16,
                    next_result));
            }
            else {
                ERR(suites.Iterate8Suite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    RenderFunc8,
                    next_result));
            }

            PF_EffectWorld* temp = current_result;
            current_result = next_result;
            next_result = temp;
        }

        if (data.scale_enabled && (fabs(data.scale_velocity) > 0.01f)) {
            data.input_world = current_result;

            if (bytesPerPixel >= 16.0) {
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    ScaleBlurFuncFloat,
                    next_result));
            }
            else if (bytesPerPixel >= 8.0) {
                ERR(suites.Iterate16Suite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    ScaleBlurFunc16,
                    next_result));
            }
            else {
                ERR(suites.Iterate8Suite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    ScaleBlurFunc8,
                    next_result));
            }

            PF_EffectWorld* temp = current_result;
            current_result = next_result;
            next_result = temp;
        }

        if (data.angle_enabled &&
            (data.has_rotation_prev_curr || data.has_rotation_curr_next)) {

            data.input_world = current_result;

            if (bytesPerPixel >= 16.0) {
                ERR(suites.IterateFloatSuite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    AngleBlurFuncFloat,
                    next_result));
            }
            else if (bytesPerPixel >= 8.0) {
                ERR(suites.Iterate16Suite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    AngleBlurFunc16,
                    next_result));
            }
            else {
                ERR(suites.Iterate8Suite1()->iterate(
                    in_data,
                    0,
                    linesL,
                    current_result,
                    &area,
                    &data,
                    AngleBlurFunc8,
                    next_result));
            }

            PF_EffectWorld* temp = current_result;
            current_result = next_result;
            next_result = temp;
        }

        ERR(suites.WorldTransformSuite1()->copy(
            in_data->effect_ref,
            current_result,
            output,
            NULL,
            NULL));

        ERR(world_suite->PF_DisposeWorld(in_data->effect_ref, &temp_world1));
        ERR(world_suite->PF_DisposeWorld(in_data->effect_ref, &temp_world2));

        if (basic_suite) {
            basic_suite->ReleaseSuite(kPFWorldSuite, kPFWorldSuiteVersion2);
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
        "Motion Blur",
        "DKT Motion Blur",
        "DKT Effects",
        AE_RESERVED_INFO);

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

        case PF_Cmd_SEQUENCE_SETUP:
            err = SequenceSetup(in_data, out_data, params, output);
            break;

        case PF_Cmd_SEQUENCE_SETDOWN:
            err = SequenceSetdown(in_data, out_data, params, output);
            break;

        case PF_Cmd_SMART_PRE_RENDER:
            err = SmartPreRender(in_data, out_data, (PF_PreRenderExtra*)extra);
            break;

        case PF_Cmd_SMART_RENDER:
            err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra);
            break;

        case PF_Cmd_RENDER:
            err = Render(in_data, out_data, params, output);
            break;
        }
    }
    catch (PF_Err& thrown_err) {
        err = thrown_err;
    }
    return err;
}


