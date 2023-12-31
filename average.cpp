extern "C"
{
#include "raylib.h"
#define RAYMATH_STATIC_INLINE
#include "raymath.h"
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
}
#if defined(PLATFORM_WEB)
#include <emscripten/emscripten.h>
#endif

#include "common.h"
#include "vec.h"
#include "mat.h"
#include "quat.h"
#include "array.h"

#include <functional>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
}

//--------------------------------------

float orbit_camera_update_azimuth(
    const float azimuth, 
    const float mouse_dx,
    const float dt)
{
    return azimuth + 1.0f * dt * -mouse_dx;
}

float orbit_camera_update_altitude(
    const float altitude, 
    const float mouse_dy,
    const float dt)
{
    return clampf(altitude + 1.0f * dt * mouse_dy, 0.0, 0.4f * PIf);
}

float orbit_camera_update_distance(
    const float distance, 
    const float dt)
{
    return clampf(distance +  200.0f * dt * -GetMouseWheelMove(), 0.1f, 100.0f);
}

void orbit_camera_update(
    Camera3D& cam, 
    float& camera_azimuth,
    float& camera_altitude,
    float& camera_distance,
    const vec3 target,
    const float mouse_dx,
    const float mouse_dy,
    const float dt)
{
    camera_azimuth = orbit_camera_update_azimuth(camera_azimuth, mouse_dx, dt);
    camera_altitude = orbit_camera_update_altitude(camera_altitude, mouse_dy, dt);
    camera_distance = orbit_camera_update_distance(camera_distance, dt);
    
    quat rotation_azimuth = quat_from_angle_axis(camera_azimuth, vec3(0, 1, 0));
    vec3 position = quat_mul_vec3(rotation_azimuth, vec3(0, 0, camera_distance));
    vec3 axis = normalize(cross(position, vec3(0, 1, 0)));
    
    quat rotation_altitude = quat_from_angle_axis(camera_altitude, axis);
    
    vec3 eye = target + quat_mul_vec3(rotation_altitude, position);

    cam.target = (Vector3){ target.x, target.y, target.z };
    cam.position = (Vector3){ eye.x, eye.y, eye.z };
}

//--------------------------------------

static inline quat quat_average_basic(const slice1d<quat> rotations, const slice1d<float> weights)
{
    quat accum = quat(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Loop over rotations
    for (int i = 0; i < rotations.size; i++)
    {
        // If more than 180 degrees away from current running average...
        if (quat_dot(accum, rotations(i)) < 0.0f)
        {
            // Add opposite-hemisphere version of the quaternion
            accum = accum - weights(i) * rotations(i);
        }
        else
        {
            // Add the quaternion
            accum = accum + weights(i) * rotations(i);
        }
    }
    
    // Normalize the final result
    return quat_normalize(quat_abs(accum));
}

static inline quat quat_average_accurate(const slice1d<quat> rotations, const slice1d<float> weights)
{
    mat4 accum = mat4_zero();
    
    // Loop over rotations
    for (int i = 0; i < rotations.size; i++)
    {
        quat q = rotations(i);
      
        // Compute the outer-product of the quaternion 
        // multiplied by the weight and add to the accumulator 
        accum = accum + weights(i) * mat4(
            q.w*q.w, q.w*q.x, q.w*q.y, q.w*q.z,
            q.x*q.w, q.x*q.x, q.x*q.y, q.x*q.z,
            q.y*q.w, q.y*q.x, q.y*q.y, q.y*q.z,
            q.z*q.w, q.z*q.x, q.z*q.y, q.z*q.z);
    }
    
    // Initial guess at eigen vector is identity quaternion
    vec4 guess = vec4(1, 0, 0, 0);
    
    // Compute first eigen vector
    vec4 u = mat4_svd_dominant_eigen(accum, guess, 64, 1e-5f);
    vec4 v = normalize(mat4_transpose_mul_vec4(accum, u));
    
    // Average quaternion is first eigen vector
    return quat_abs(quat(v.x, v.y, v.z, v.w));
}

//--------------------------------------

void draw_axis(const vec3 pos, const quat rot, const float scale = 1.0f)
{
    vec3 axis0 = pos + quat_mul_vec3(rot, scale * vec3(1.0f, 0.0f, 0.0f));
    vec3 axis1 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 1.0f, 0.0f));
    vec3 axis2 = pos + quat_mul_vec3(rot, scale * vec3(0.0f, 0.0f, 1.0f));
    
    DrawLine3D(to_Vector3(pos), to_Vector3(axis0), RED);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis1), GREEN);
    DrawLine3D(to_Vector3(pos), to_Vector3(axis2), BLUE);
}

//--------------------------------------

void update_callback(void* args)
{
    ((std::function<void()>*)args)->operator()();
}

//--------------------------------------

int main(void)
{
    // Init Window
    
    const int screen_width = 1280;
    const int screen_height = 720;
    // const int screen_width = 640;
    // const int screen_height = 480;
    
    SetConfigFlags(FLAG_VSYNC_HINT);
    SetConfigFlags(FLAG_MSAA_4X_HINT);
    InitWindow(screen_width, screen_height, "raylib [quaternion average]");
    SetTargetFPS(60);
    
    Camera camera = { 0 };
    camera.position = (Vector3){ 5.0f, 5.0f, 5.0f };
    camera.target = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.6f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    
    float camera_azimuth = 0.0f;
    float camera_altitude = 0.4f;
    float camera_distance = 4.0f;
    
    Model duck = LoadModel("resources/duck.obj");
    Shader duck_shader = LoadShader("./resources/duck.vs", "./resources/duck.fs");
    duck.materials[0].shader = duck_shader;
    
    quat duck_rotation = quat();
    vec3 duck_position = vec3(0.5f, 1.0f, 0.0f);
    vec3 duck_scale = vec3(0.02f, 0.02f, 0.02f);
    
    int duck_rotation_num = 6;
    
    array1d<quat> duck_rotations(duck_rotation_num);
    array1d<quat> duck_rotations_reversed(duck_rotation_num);
    array1d<vec3> duck_angle_axis_rotations(duck_rotation_num);
    array1d<float> duck_weights(duck_rotation_num);
    array1d<float> duck_weights_normalized(duck_rotation_num);
    array1d<float> duck_weights_normalized_reversed(duck_rotation_num);
    
    duck_rotations.set(quat());
    duck_rotations_reversed.set(quat());
    duck_angle_axis_rotations.set(vec3());
    duck_weights.set(1.0f);
    duck_weights_normalized.set(1.0f / duck_rotation_num);
    duck_weights_normalized_reversed.set(1.0f / duck_rotation_num);
    
    int duck_selected = 0;
    bool use_accurate = false;
    bool reverse_order = false;
    
    // Go
    
    srand(0x71512732);
    
    const float dt = 1.0 / 60.0f;
  
    auto update_func = [&]()
    {
        // Update Camera
        
        orbit_camera_update(
            camera, 
            camera_azimuth,
            camera_altitude,
            camera_distance,
            vec3(0, 1.0f, 0),
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().x : 0.0f,
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().y : 0.0f,
            dt);
        
        // Normalize Weights
        
        float total = 0.0f;
        for (int i = 0; i < duck_rotation_num; i++)
        {
            total += duck_weights(i);
        }
        total = maxf(total, 1e-8f);

        for (int i = 0; i < duck_rotation_num; i++)
        {
            duck_weights_normalized(i) = duck_weights(i) / total;
        }
        
        // Compute Reverse
        
        if (reverse_order)
        {
            for (int i = 0; i < duck_rotation_num; i++)
            {
                duck_rotations_reversed(i) = duck_rotations(duck_rotation_num - 1 - i);
                duck_weights_normalized_reversed(i) = duck_weights_normalized(duck_rotation_num - 1 - i);
            }
        }
        
        // Compute Average
        
        if (use_accurate)
        {
            if (reverse_order)
            {
                duck_rotation = quat_average_accurate(duck_rotations_reversed, duck_weights_normalized_reversed);              
            }
            else
            {
                duck_rotation = quat_average_accurate(duck_rotations, duck_weights_normalized);              
            }
        }
        else
        {
            if (reverse_order)
            {
                duck_rotation = quat_average_basic(duck_rotations_reversed, duck_weights_normalized_reversed);              
            }
            else
            {
                duck_rotation = quat_average_basic(duck_rotations, duck_weights_normalized);
            }
        }
      
        // Update Duck
      
        Matrix xform_rotation = QuaternionToMatrix((Quaternion){
            duck_rotation.x,
            duck_rotation.y,
            duck_rotation.z,
            duck_rotation.w});
        Matrix xform_scale = MatrixScale(duck_scale.x, duck_scale.y, duck_scale.z);
        Matrix xform_position = MatrixTranslate(duck_position.x, duck_position.y, duck_position.z);
        
        duck.transform = MatrixMultiply(MatrixMultiply(xform_scale, xform_rotation), xform_position);
      
        BeginDrawing();

        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);
    
        DrawGrid(20, 1.0f);
        
        // Draw Main Duck
        
        DrawModel(duck, (Vector3){ 0.0f, 0.0f, 0.0f }, 1.0f, ORANGE);
        draw_axis(duck_position, duck_rotation, 0.5f);
        
        // Draw Sub-Rotations
        
        for (int i = 0; i < duck_rotation_num; i++)
        {
            float x = (i / 3) * 0.75f - 1.5f;
            float z = (i % 3) * 0.75f - 1.0f;
            
            Matrix xform_rotation = QuaternionToMatrix((Quaternion){
                duck_rotations(i).x,
                duck_rotations(i).y,
                duck_rotations(i).z,
                duck_rotations(i).w});
                
            Matrix xform_scale = MatrixScale(0.01f * duck_weights(i), 0.01f * duck_weights(i), 0.01f * duck_weights(i));
            Matrix xform_position = MatrixTranslate(x, 1.0f, z);
            
            duck.transform = MatrixMultiply(MatrixMultiply(xform_scale, xform_rotation), xform_position);

            DrawModel(duck, (Vector3){ 0.0f, 0.0f, 0.0f }, 1.0f, i == duck_selected ? RED : ORANGE);
            draw_axis(vec3(x, 1.0f, z), duck_rotations(i), duck_weights(i) * 0.25f);
        }
        
        EndMode3D();
        
        float ui_settings_left = screen_width - 150;
        
        GuiGroupBox((Rectangle){ ui_settings_left - 10, 20, 150, 350 }, "settings");

        GuiCheckBox(
            (Rectangle){ ui_settings_left, 40, 20, 20 }, 
            "accurate method", 
            &use_accurate);

        GuiCheckBox(
            (Rectangle){ ui_settings_left, 70, 20, 20 }, 
            "reverse order", 
            &reverse_order);

        if (GuiButton((Rectangle){ ui_settings_left + 10, 100, 110, 20 }, "random rotations"))
        {
            for (int i = 0; i < duck_rotation_num; i++)
            {
                duck_rotations(i) = quat_normalize(quat(
                    2.0f * ((float)rand()) / RAND_MAX - 1.0f,
                    2.0f * ((float)rand()) / RAND_MAX - 1.0f,
                    2.0f * ((float)rand()) / RAND_MAX - 1.0f,
                    2.0f * ((float)rand()) / RAND_MAX - 1.0f));
                    
                duck_angle_axis_rotations(i) = quat_to_scaled_angle_axis(duck_rotations(i));
            }
        }

        if (GuiButton((Rectangle){ ui_settings_left + 10, 130, 110, 20 }, "random weights"))
        {
            for (int i = 0; i < duck_rotation_num; i++)
            {
                duck_weights(i) = ((float)rand()) / RAND_MAX;
            }
        }

        if (GuiButton((Rectangle){ ui_settings_left + 10, 160, 110, 20 }, "reset rotations"))
        {
            duck_rotations.set(quat());
            duck_angle_axis_rotations.set(vec3());
        }

        if (GuiButton((Rectangle){ ui_settings_left + 10, 190, 110, 20 }, "reset weights"))
        {
            duck_weights.set(1.0f);
        }
        
        float duck_selected_float = duck_selected;
        
        GuiSliderBar(
            (Rectangle){ ui_settings_left + 30, 220, 75, 20 }, 
            "index", 
            TextFormat("%i", duck_selected),
            &duck_selected_float,
            0, duck_rotation_num - 1);
        
        duck_selected = (int)roundf(duck_selected_float);
        
        GuiSliderBar(
            (Rectangle){ ui_settings_left + 30, 250, 75, 20 }, 
            "rot x", 
            TextFormat("%3.2f", duck_angle_axis_rotations(duck_selected).x),
            &duck_angle_axis_rotations(duck_selected).x,
            -2*PIf, 2*PIf);  

        GuiSliderBar(
            (Rectangle){ ui_settings_left + 30, 280, 75, 20 }, 
            "rot y", 
            TextFormat("%3.2f", duck_angle_axis_rotations(duck_selected).y),
            &duck_angle_axis_rotations(duck_selected).y,
            -2*PIf, 2*PIf);  
            
        GuiSliderBar(
            (Rectangle){ ui_settings_left + 30, 310, 75, 20 }, 
            "rot z", 
            TextFormat("%3.2f", duck_angle_axis_rotations(duck_selected).z),
            &duck_angle_axis_rotations(duck_selected).z,
            -2*PIf, 2*PIf);  

        GuiSliderBar(
            (Rectangle){ ui_settings_left + 30, 340, 75, 20 }, 
            "weight", 
            TextFormat("%3.2f", duck_weights(duck_selected)),
            &duck_weights(duck_selected),
            0.0f, 1.0f);  

        duck_rotations(duck_selected) = quat_from_scaled_angle_axis(duck_angle_axis_rotations(duck_selected));

        float err_total = 0.0f;
        for (int i = 0; i < duck_rotation_num; i++)
        {
            err_total += duck_weights(i) * squaref(quat_frobenius_distance(duck_rotation, duck_rotations(i)));
        }
        
        GuiLabel((Rectangle){ 10, 10, 200, 20 }, TextFormat("Sum Squared Sin Half-Angle: %6.4f", err_total));

        EndDrawing();
    };

#if defined(PLATFORM_WEB)
    std::function<void()> u{update_func};
    emscripten_set_main_loop_arg(update_callback, &u, 0, 1);
#else
    while (!WindowShouldClose())
    {
        update_func();
    }
#endif
    
    UnloadModel(duck);
    UnloadShader(duck_shader);
    
    CloseWindow();

    return 0;
}