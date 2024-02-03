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
#include "character.h"
#include "database.h"

#include <functional>

//--------------------------------------

static inline Vector3 to_Vector3(vec3 v)
{
    return (Vector3){ v.x, v.y, v.z };
}

//--------------------------------------

// Perform linear blend skinning and copy 
// result into mesh data. Update and upload 
// deformed vertex positions and normals to GPU
void deform_character_mesh(
  Mesh& mesh, 
  const character& c,
  const slice1d<vec3> bone_anim_positions,
  const slice1d<quat> bone_anim_rotations,
  const slice1d<int> bone_parents)
{
    linear_blend_skinning_positions(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.vertices),
        c.positions,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_positions,
        c.bone_rest_rotations,
        bone_anim_positions,
        bone_anim_rotations);
    
    linear_blend_skinning_normals(
        slice1d<vec3>(mesh.vertexCount, (vec3*)mesh.normals),
        c.normals,
        c.bone_weights,
        c.bone_indices,
        c.bone_rest_rotations,
        bone_anim_rotations);
    
    UpdateMeshBuffer(mesh, 0, mesh.vertices, mesh.vertexCount * 3 * sizeof(float), 0);
    UpdateMeshBuffer(mesh, 2, mesh.normals, mesh.vertexCount * 3 * sizeof(float), 0);
}

Mesh make_character_mesh(character& c)
{
    Mesh mesh = { 0 };
    
    mesh.vertexCount = c.positions.size;
    mesh.triangleCount = c.triangles.size / 3;
    mesh.vertices = (float*)MemAlloc(c.positions.size * 3 * sizeof(float));
    mesh.texcoords = (float*)MemAlloc(c.texcoords.size * 2 * sizeof(float));
    mesh.normals = (float*)MemAlloc(c.normals.size * 3 * sizeof(float));
    mesh.indices = (unsigned short*)MemAlloc(c.triangles.size * sizeof(unsigned short));
    
    memcpy(mesh.vertices, c.positions.data, c.positions.size * 3 * sizeof(float));
    memcpy(mesh.texcoords, c.texcoords.data, c.texcoords.size * 2 * sizeof(float));
    memcpy(mesh.normals, c.normals.data, c.normals.size * 3 * sizeof(float));
    memcpy(mesh.indices, c.triangles.data, c.triangles.size * sizeof(unsigned short));
    
    UploadMesh(&mesh, true);
    
    return mesh;
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

void bone_positions_weighted_average(
    slice1d<vec3> blended_positions, 
    const slice2d<vec3> sample_positions, 
    const slice1d<float> sample_weights)
{
    assert(sample_positions.rows == sample_weights.size);
    assert(sample_positions.cols == blended_positions.size);
    
    blended_positions.zero();
    for (int i = 0; i < sample_positions.rows; i++)
    {
        for (int j = 0; j < sample_positions.cols; j++)
        {
            blended_positions(j) += sample_weights(i) * sample_positions(i, j);
        }
    }
}

void bone_rotations_weighted_average(
    slice1d<quat> blended_rotations,
    slice1d<mat4> accum_rotations,
    const slice2d<quat> sample_rotations, 
    const slice1d<float> sample_weights)
{
    assert(sample_rotations.rows == sample_weights.size);
    assert(sample_rotations.cols == blended_rotations.size);
    assert(sample_rotations.cols == accum_rotations.size);
    
    accum_rotations.zero();
    
    for (int i = 0; i < sample_rotations.rows; i++)
    {
        for (int j = 0; j < sample_rotations.cols; j++)
        {
            quat q = sample_rotations(i, j);
          
            accum_rotations(j) = accum_rotations(j) + sample_weights(i) * mat4(
                q.w*q.w, q.w*q.x, q.w*q.y, q.w*q.z,
                q.x*q.w, q.x*q.x, q.x*q.y, q.x*q.z,
                q.y*q.w, q.y*q.x, q.y*q.y, q.y*q.z,
                q.z*q.w, q.z*q.x, q.z*q.y, q.z*q.z);
        }
    }
    
    for (int j = 0; j < sample_rotations.cols; j++)
    {
        vec4 guess = vec4(1, 0, 0, 0);
        vec4 u = mat4_svd_dominant_eigen(accum_rotations(j), guess, 64, 1e-5f);
        vec4 v = normalize(mat4_transpose_mul_vec4(accum_rotations(j), u));
        
        blended_rotations(j) = quat_abs(quat(v.x, v.y, v.z, v.w));
    }
}

void bone_rotations_weighted_average_ref(
    slice1d<quat> blended_rotations,
    slice1d<mat4> accum_rotations,
    const slice1d<quat> reference_rotations,
    const slice2d<quat> sample_rotations, 
    const slice1d<float> sample_weights)
{
    assert(sample_rotations.rows == sample_weights.size);
    assert(sample_rotations.cols == blended_rotations.size);
    assert(sample_rotations.cols == accum_rotations.size);
    
    accum_rotations.zero();
    
    for (int i = 0; i < sample_rotations.rows; i++)
    {
        for (int j = 0; j < sample_rotations.cols; j++)
        {
            quat q = quat_abs(quat_inv_mul(reference_rotations(j), sample_rotations(i, j)));
          
            accum_rotations(j) = accum_rotations(j) + sample_weights(i) * mat4(
                q.w*q.w, q.w*q.x, q.w*q.y, q.w*q.z,
                q.x*q.w, q.x*q.x, q.x*q.y, q.x*q.z,
                q.y*q.w, q.y*q.x, q.y*q.y, q.y*q.z,
                q.z*q.w, q.z*q.x, q.z*q.y, q.z*q.z);
        }
    }
    
    for (int j = 0; j < sample_rotations.cols; j++)
    {
        vec4 guess = vec4(1, 0, 0, 0);
        vec4 u = mat4_svd_dominant_eigen(accum_rotations(j), guess, 64, 1e-5f);
        vec4 v = normalize(mat4_transpose_mul_vec4(accum_rotations(j), u));
        
        blended_rotations(j) = quat_abs(quat_mul(reference_rotations(j), quat(v.x, v.y, v.z, v.w)));
    }
}

void bone_rotations_weighted_average_raw(
    slice1d<quat> blended_rotations, 
    const slice2d<quat> sample_rotations, 
    const slice1d<float> sample_weights)
{
    assert(sample_rotations.rows == sample_weights.size);
    assert(sample_rotations.cols == blended_rotations.size);
    
    blended_rotations.zero();
    
    for (int i = 0; i < sample_rotations.rows; i++)
    {
        for (int j = 0; j < sample_rotations.cols; j++)
        {
            blended_rotations(j) = blended_rotations(j) + 
                sample_weights(i) * quat_abs(sample_rotations(i, j));
        }
    }
    
    for (int j = 0; j < sample_rotations.cols; j++)
    {
        blended_rotations(j) = quat_normalize(blended_rotations(j));
    }
}

void bone_rotations_weighted_average_raw_ref(
    slice1d<quat> blended_rotations, 
    const slice1d<quat> reference_rotations,
    const slice2d<quat> sample_rotations, 
    const slice1d<float> sample_weights)
{
    assert(sample_rotations.rows == sample_weights.size);
    assert(sample_rotations.cols == blended_rotations.size);
    assert(sample_rotations.cols == reference_rotations.size);
    
    blended_rotations.zero();
    
    for (int i = 0; i < sample_rotations.rows; i++)
    {
        for (int j = 0; j < sample_rotations.cols; j++)
        {
            blended_rotations(j) = blended_rotations(j) + sample_weights(i) * 
                quat_abs(quat_inv_mul(
                    reference_rotations(j), sample_rotations(i, j)));
        }
    }
    
    for (int j = 0; j < sample_rotations.cols; j++)
    {
        blended_rotations(j) = quat_abs(quat_mul(
            reference_rotations(j), quat_normalize(blended_rotations(j))));
    }
}

void bone_rotations_weighted_average_log(
    slice1d<quat> blended_rotations, 
    slice1d<vec3> accum_rotations, 
    const slice2d<quat> sample_rotations, 
    const slice1d<float> sample_weights)
{
    assert(sample_rotations.rows == sample_weights.size);
    assert(sample_rotations.cols == blended_rotations.size);
    assert(sample_rotations.cols == accum_rotations.size);
    
    accum_rotations.zero();
    
    for (int i = 0; i < sample_rotations.rows; i++)
    {
        for (int j = 0; j < sample_rotations.cols; j++)
        {
            accum_rotations(j) += sample_weights(i) * 
                quat_log(quat_abs(sample_rotations(i, j)));
        }
    }
    
    for (int j = 0; j < sample_rotations.cols; j++)
    {
        blended_rotations(j) = quat_exp(accum_rotations(j));
    }
}

void bone_rotations_weighted_average_log_ref(
    slice1d<quat> blended_rotations, 
    slice1d<vec3> accum_rotations, 
    const slice1d<quat> reference_rotations,
    const slice2d<quat> sample_rotations, 
    const slice1d<float> sample_weights)
{
    assert(sample_rotations.rows == sample_weights.size);
    assert(sample_rotations.cols == blended_rotations.size);
    assert(sample_rotations.cols == accum_rotations.size);
    assert(sample_rotations.cols == reference_rotations.size);
    
    accum_rotations.zero();
    
    for (int i = 0; i < sample_rotations.rows; i++)
    {
        for (int j = 0; j < sample_rotations.cols; j++)
        {
            accum_rotations(j) += sample_weights(i) * 
                quat_log(quat_abs(quat_inv_mul(
                    reference_rotations(j), sample_rotations(i, j))));
        }
    }
    
    for (int j = 0; j < sample_rotations.cols; j++)
    {
        blended_rotations(j) = quat_abs(quat_mul(
            reference_rotations(j), quat_exp(accum_rotations(j))));
    }
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
    
    // Character
    
    character character_data;
    character_load(character_data, "./resources/character.bin");
    
    Shader character_shader = LoadShader("./resources/character.vs", "./resources/character.fs");
    Mesh character_mesh = make_character_mesh(character_data);
    Model character_model = LoadModelFromMesh(character_mesh);
    character_model.materials[0].shader = character_shader;
        
    // Load Animation Data and build Matching Database
    
    database db;
    database_load(db, "./resources/database.bin");
    
    array1d<vec3> reference_positions(db.nbones());
    array1d<quat> reference_rotations(db.nbones());
    
    backward_kinematics_full(
        reference_positions,
        reference_rotations,
        character_data.bone_rest_positions,
        character_data.bone_rest_rotations,
        db.bone_parents);
    
    array1d<vec3> global_bone_positions(db.nbones());
    array1d<quat> global_bone_rotations(db.nbones());
    
    array2d<vec3> sample_positions(3, db.nbones());
    array2d<quat> sample_rotations(3, db.nbones());
    array1d<float> sample_weights(3);
    sample_weights(0) = 0.33f;
    sample_weights(1) = 0.33f;
    sample_weights(2) = 0.33f;
    
    array1d<vec3> blended_positions(db.nbones());
    array1d<quat> blended_rotations(db.nbones());
    array1d<mat4> blended_accum_rotations_mat(db.nbones());
    array1d<vec3> blended_accum_rotations_log(db.nbones());
    
    int pose0 = 10243;
    int pose1 = 8536;
    int pose2 = 11089;
    
    sample_positions(0) = db.bone_positions(pose0);
    sample_rotations(0) = db.bone_rotations(pose0);
    sample_positions(1) = db.bone_positions(pose1);
    sample_rotations(1) = db.bone_rotations(pose1);
    sample_positions(2) = db.bone_positions(pose2);
    sample_rotations(2) = db.bone_rotations(pose2);
    
    blended_positions = reference_positions;
    blended_rotations = reference_rotations;
    
    bool use_reference_pose = false;
    int averaging_method = 0;
    bool averaging_method_active = false;
    
    bool appendix = false;

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
            appendix ? vec3(0, 1.75f, 0) : vec3(0, 1.0f, 0),
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().x : 0.0f,
            (IsKeyDown(KEY_LEFT_CONTROL) && IsMouseButtonDown(0)) ? GetMouseDelta().y : 0.0f,
            dt);
        
        if (!appendix)
        {       
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
        }
        else
        {
            // Normalize Weights
            
            float total = sample_weights(0) + sample_weights(1) + sample_weights(2);
            sample_weights(0) /= total;
            sample_weights(1) /= total;
            sample_weights(2) /= total;
            
            // Compute Position Weighted Average
            
            bone_positions_weighted_average(
                blended_positions, 
                sample_positions, 
                sample_weights);
            
            // Compute Rotation Weighted Average
            
            if (averaging_method == 0)
            {
                if (use_reference_pose)
                {
                    bone_rotations_weighted_average_ref(
                        blended_rotations, 
                        blended_accum_rotations_mat,
                        reference_rotations,
                        sample_rotations, 
                        sample_weights);
                }
                else
                {
                    bone_rotations_weighted_average(
                        blended_rotations, 
                        blended_accum_rotations_mat,
                        sample_rotations, 
                        sample_weights);
                }
            }
            else if (averaging_method == 1)
            {
                if (use_reference_pose)
                {
                    bone_rotations_weighted_average_raw_ref(
                        blended_rotations, 
                        reference_rotations,
                        sample_rotations, 
                        sample_weights);
                }
                else
                {
                    bone_rotations_weighted_average_raw(
                        blended_rotations, 
                        sample_rotations, 
                        sample_weights);
                }                  
            }
            else if (averaging_method == 2)
            {
                if (use_reference_pose)
                {
                    bone_rotations_weighted_average_log_ref(
                        blended_rotations, 
                        blended_accum_rotations_log,
                        reference_rotations,
                        sample_rotations, 
                        sample_weights);
                }
                else
                {
                    bone_rotations_weighted_average_log(
                        blended_rotations, 
                        blended_accum_rotations_log,
                        sample_rotations, 
                        sample_weights);
                }  
            }
        }
        
        BeginDrawing();

        ClearBackground(RAYWHITE);
        
        BeginMode3D(camera);
    
        DrawGrid(20, 1.0f);
        
        if (!appendix)
        {  
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
        }
        else
        {          
            // Draw Sample 0
          
            forward_kinematics_full(
                global_bone_positions,
                global_bone_rotations,
                sample_positions(0),
                sample_rotations(0),
                db.bone_parents);
          
            deform_character_mesh(
                character_mesh, 
                character_data, 
                global_bone_positions, 
                global_bone_rotations,
                db.bone_parents);
            
            DrawModel(character_model, (Vector3){-1.0f, 0.0f, 0.0f}, 1.0f, RAYWHITE);
          
            // Draw Sample 1
            
            forward_kinematics_full(
                global_bone_positions,
                global_bone_rotations,
                sample_positions(1),
                sample_rotations(1),
                db.bone_parents);
          
            deform_character_mesh(
                character_mesh, 
                character_data, 
                global_bone_positions, 
                global_bone_rotations,
                db.bone_parents);
            
            DrawModel(character_model, (Vector3){0.0f, 0.0f, 0.0f}, 1.0f, RAYWHITE);
          
            // Draw Sample 2
          
            forward_kinematics_full(
                global_bone_positions,
                global_bone_rotations,
                sample_positions(2),
                sample_rotations(2),
                db.bone_parents);
          
            deform_character_mesh(
                character_mesh, 
                character_data, 
                global_bone_positions, 
                global_bone_rotations,
                db.bone_parents);
            
            DrawModel(character_model, (Vector3){1.0f, 0.0f, 0.0f}, 1.0f, RAYWHITE);
          
            // Draw Result
          
            forward_kinematics_full(
                global_bone_positions,
                global_bone_rotations,
                blended_positions,
                blended_rotations,
                db.bone_parents);
          
            deform_character_mesh(
                character_mesh, 
                character_data, 
                global_bone_positions, 
                global_bone_rotations,
                db.bone_parents);
            
            DrawModel(character_model, (Vector3){0.0f, 1.75f, 0.0f}, 1.0f, RED);
        }
        
        EndMode3D();
        
        if (!appendix)
        {  
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
        }
        else
        {
            // Selector for method on the right
            
            GuiSliderBar(
                (Rectangle){ 50, 20, 120, 20 }, 
                "pose 0", 
                TextFormat("%3.2f", sample_weights(0)),
                &sample_weights(0),
                0.0f, 1.0f); 
            
            GuiSliderBar(
                (Rectangle){ 50, 50, 120, 20 }, 
                "pose 1", 
                TextFormat("%3.2f", sample_weights(1)),
                &sample_weights(1),
                0.0f, 1.0f); 
            
            GuiSliderBar(
                (Rectangle){ 50, 80, 120, 20 }, 
                "pose 2", 
                TextFormat("%3.2f", sample_weights(2)),
                &sample_weights(2),
                0.0f, 1.0f); 
            
            // Selector for weight on the left
          
            float ui_settings_left = screen_width - 150;

            GuiGroupBox((Rectangle){ ui_settings_left - 10, 20, 150, 150 }, "settings");

            GuiCheckBox(
                (Rectangle){ ui_settings_left, 40, 20, 20 }, 
                "use reference pose", 
                &use_reference_pose);
          
            if (GuiDropdownBox(
                (Rectangle){ ui_settings_left, 70, 130, 20 }, 
                "Accurate;Raw;Log", 
                &averaging_method,
                averaging_method_active))
            {
                averaging_method_active = !averaging_method_active;
            }
        }
        
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
    
    UnloadModel(character_model);
    UnloadShader(character_shader);    
    UnloadModel(duck);
    UnloadShader(duck_shader);
    
    CloseWindow();

    return 0;
}