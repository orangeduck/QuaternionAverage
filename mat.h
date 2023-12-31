#pragma once
#include "vec.h"

struct mat3
{
    mat3() : 
        xx(1.0f), xy(0.0f), xz(0.0f),
        yx(0.0f), yy(1.0f), yz(0.0f),
        zx(0.0f), zy(0.0f), zz(1.0f) {}
    
    mat3(
        float _xx, float _xy, float _xz, 
        float _yx, float _yy, float _yz, 
        float _zx, float _zy, float _zz) : 
        xx(_xx), xy(_xy), xz(_xz),
        yx(_yx), yy(_yy), yz(_yz),
        zx(_zx), zy(_zy), zz(_zz) {}
    
    mat3(vec3 r0, vec3 r1, vec3 r2) : 
        xx(r0.x), xy(r0.y), xz(r0.z),
        yx(r1.x), yy(r1.y), yz(r1.z),
        zx(r2.x), zy(r2.y), zz(r2.z) {}
    
    vec3 r0() const { return vec3(xx, xy, xz); }
    vec3 r1() const { return vec3(yx, yy, yz); }
    vec3 r2() const { return vec3(zx, zy, zz); }

    vec3 c0() const { return vec3(xx, yx, zx); }
    vec3 c1() const { return vec3(xy, yy, zy); }
    vec3 c2() const { return vec3(xz, yz, zz); }

    float xx, xy, xz,
          yx, yy, yz,
          zx, zy, zz;
};

static inline mat3 operator+(mat3 m, mat3 n)
{
    return mat3(
        m.xx + n.xx, m.xy + n.xy, m.xz + n.xz,
        m.yx + n.yx, m.yy + n.yy, m.yz + n.yz,
        m.zx + n.zx, m.zy + n.zy, m.zz + n.zz);
}

static inline mat3 operator-(mat3 m, mat3 n)
{
    return mat3(
        m.xx - n.xx, m.xy - n.xy, m.xz - n.xz,
        m.yx - n.yx, m.yy - n.yy, m.yz - n.yz,
        m.zx - n.zx, m.zy - n.zy, m.zz - n.zz);
}

static inline mat3 operator/(mat3 m, float v)
{
    return mat3(
        m.xx / v, m.xy / v, m.xz / v,
        m.yx / v, m.yy / v, m.yz / v,
        m.zx / v, m.zy / v, m.zz / v);
}

static inline mat3 operator*(float v, mat3 m)
{
    return mat3(
        v * m.xx, v * m.xy, v * m.xz,
        v * m.yx, v * m.yy, v * m.yz,
        v * m.zx, v * m.zy, v * m.zz);
}

static inline mat3 mat3_zero()
{
    return mat3(
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f);
}

static inline mat3 mat3_transpose(mat3 m)
{
    return mat3(
        m.xx, m.yx, m.zx,
        m.xy, m.yy, m.zy,
        m.xz, m.yz, m.zz);
}

static inline mat3 mat3_mul(mat3 m, mat3 n)
{
  return mat3(
      dot(m.r0(), n.c0()), dot(m.r0(), n.c1()), dot(m.r0(), n.c2()),
      dot(m.r1(), n.c0()), dot(m.r1(), n.c1()), dot(m.r1(), n.c2()),
      dot(m.r2(), n.c0()), dot(m.r2(), n.c1()), dot(m.r2(), n.c2()));
}

static inline mat3 mat3_transpose_mul(mat3 m, mat3 n)
{
  return mat3(
      dot(m.c0(), n.c0()), dot(m.c0(), n.c1()), dot(m.c0(), n.c2()),
      dot(m.c1(), n.c0()), dot(m.c1(), n.c1()), dot(m.c1(), n.c2()),
      dot(m.c2(), n.c0()), dot(m.c2(), n.c1()), dot(m.c2(), n.c2()));
}

static inline vec3 mat3_mul_vec3(mat3 m, vec3 v)
{
    return vec3(
        dot(m.r0(), v),
        dot(m.r1(), v),
        dot(m.r2(), v));
}

static inline vec3 mat3_transpose_mul_vec3(mat3 m, vec3 v)
{
    return vec3(
        dot(m.c0(), v),
        dot(m.c1(), v),
        dot(m.c2(), v));
}

static inline mat3 mat3_from_angle_axis(float angle, vec3 axis)
{
    float a0 = axis.x, a1 = axis.y, a2 = axis.z; 
    float c = cosf(angle), s = sinf(angle), t = 1.0f - cosf(angle);
    
    return mat3(
        c+a0*a0*t, a0*a1*t-a2*s, a0*a2*t+a1*s,
        a0*a1*t+a2*s, c+a1*a1*t, a1*a2*t-a0*s,
        a0*a2*t-a1*s, a1*a2*t+a0*s, c+a2*a2*t);
}

static inline mat3 mat3_outer(vec3 v, vec3 w)
{
    return mat3(
        v.x * w.x, v.x * w.y, v.x * w.z,
        v.y * w.x, v.y * w.y, v.y * w.z,
        v.z * w.x, v.z * w.y, v.z * w.z);
}

static inline vec3 mat3_svd_dominant_eigen(
    const mat3 A, 
    const vec3 v0,
    const int iterations, 
    const float eps)
{
    // Initial Guess at Eigen Vector & Value
    vec3 v = v0;
    float ev = (mat3_mul_vec3(A, v) / v).x;
    
    for (int i = 0; i < iterations; i++)
    {
        // Power Iteration
        vec3 Av = mat3_mul_vec3(A, v);
        
        // Next Guess at Eigen Vector & Value
        vec3 v_new = normalize(Av);
        float ev_new = (mat3_mul_vec3(A, v_new) / v_new).x;
        
        // Break if converged
        if (fabs(ev - ev_new) < eps)
        {
            break;
        }
        
        // Update best guess
        v = v_new;
        ev = ev_new;
    }
    
    return v;
}

static inline void mat3_svd_piter(
    mat3& U,
    vec3& s,
    mat3& V,
    const mat3 A, 
    const int iterations = 64,
    const float eps = 1e-5f)
{
    // First Eigen Vector
    vec3 g0 = vec3(1, 0, 0);
    mat3 B0 = A;
    vec3 u0 = mat3_svd_dominant_eigen(B0, g0, iterations, eps);
    vec3 v0_unnormalized = mat3_transpose_mul_vec3(A, u0);
    float s0 = length(v0_unnormalized);
    vec3 v0 = s0 < eps ? g0 : normalize(v0_unnormalized);

    // Second Eigen Vector
    mat3 B1 = A;
    vec3 g1 = normalize(cross(vec3(0, 0, 1), v0));
    B1 = B1 - s0 * mat3_outer(u0, v0);
    vec3 u1 = mat3_svd_dominant_eigen(B1, g1, iterations, eps);
    vec3 v1_unnormalized = mat3_transpose_mul_vec3(A, u1);
    float s1 = length(v1_unnormalized);
    vec3 v1 = s1 < eps ? g1 : normalize(v1_unnormalized);
    
    // Third Eigen Vector
    mat3 B2 = A;
    vec3 v2 = normalize(cross(v0, v1));
    B2 = B2 - s0 * mat3_outer(u0, v0);
    B2 = B2 - s1 * mat3_outer(u1, v1);
    vec3 u2 = mat3_svd_dominant_eigen(B2, v2, iterations, eps);
    float s2 = length(mat3_transpose_mul_vec3(A, u2));
    
    // Done
    U = mat3(u0, u1, u2);
    s = vec3(s0, s1, s2);
    V = mat3(v0, v1, v2);
}

//--------------------------------------

struct mat4
{
    mat4() : 
        xx(1.0f), xy(0.0f), xz(0.0f), xw(0.0f),
        yx(0.0f), yy(1.0f), yz(0.0f), yw(0.0f),
        zx(0.0f), zy(0.0f), zz(1.0f), zw(0.0f),
        wx(0.0f), wy(0.0f), wz(1.0f), ww(1.0f) {}
    
    mat4(
        float _xx, float _xy, float _xz, float _xw,
        float _yx, float _yy, float _yz, float _yw, 
        float _zx, float _zy, float _zz, float _zw, 
        float _wx, float _wy, float _wz, float _ww) : 
        xx(_xx), xy(_xy), xz(_xz), xw(_xw),
        yx(_yx), yy(_yy), yz(_yz), yw(_yw),
        zx(_zx), zy(_zy), zz(_zz), zw(_zw),
        wx(_wx), wy(_wy), wz(_wz), ww(_ww) {}
    
    mat4(vec4 r0, vec4 r1, vec4 r2, vec4 r3) : 
        xx(r0.x), xy(r0.y), xz(r0.z), xw(r0.w),
        yx(r1.x), yy(r1.y), yz(r1.z), yw(r1.w),
        zx(r2.x), zy(r2.y), zz(r2.z), zw(r2.w),
        wx(r3.x), wy(r3.y), wz(r3.z), ww(r3.w) {}
    
    vec4 r0() const { return vec4(xx, xy, xz, xw); }
    vec4 r1() const { return vec4(yx, yy, yz, yw); }
    vec4 r2() const { return vec4(zx, zy, zz, zw); }
    vec4 r3() const { return vec4(wx, wy, wz, ww); }

    vec4 c0() const { return vec4(xx, yx, zx, wx); }
    vec4 c1() const { return vec4(xy, yy, zy, wy); }
    vec4 c2() const { return vec4(xz, yz, zz, wz); }
    vec4 c3() const { return vec4(xw, yw, zw, ww); }

    float xx, xy, xz, xw,
          yx, yy, yz, yw,
          zx, zy, zz, zw,
          wx, wy, wz, ww;
};

static inline mat4 operator+(mat4 m, mat4 n)
{
    return mat4(
        m.xx + n.xx, m.xy + n.xy, m.xz + n.xz, m.xw + n.xw,
        m.yx + n.yx, m.yy + n.yy, m.yz + n.yz, m.yw + n.yw,
        m.zx + n.zx, m.zy + n.zy, m.zz + n.zz, m.zw + n.zw,
        m.wx + n.wx, m.wy + n.wy, m.wz + n.wz, m.ww + n.ww);
}

static inline mat4 operator-(mat4 m, mat4 n)
{
    return mat4(
        m.xx - n.xx, m.xy - n.xy, m.xz - n.xz, m.xw - n.xw,
        m.yx - n.yx, m.yy - n.yy, m.yz - n.yz, m.yw - n.yw,
        m.zx - n.zx, m.zy - n.zy, m.zz - n.zz, m.zw - n.zw,
        m.wx - n.wx, m.wy - n.wy, m.wz - n.wz, m.ww - n.ww);
}

static inline mat4 operator/(mat4 m, float v)
{
    return mat4(
        m.xx  / v, m.xy / v, m.xz / v, m.xw / v,
        m.yx  / v, m.yy / v, m.yz / v, m.yw / v,
        m.zx  / v, m.zy / v, m.zz / v, m.zw / v,
        m.wx  / v, m.wy / v, m.wz / v, m.ww / v);
}

static inline mat4 operator*(float v, mat4 m)
{
    return mat4(
        v * m.xx, v * m.xy, v * m.xz, v * m.xw,
        v * m.yx, v * m.yy, v * m.yz, v * m.yw,
        v * m.zx, v * m.zy, v * m.zz, v * m.zw,
        v * m.wx, v * m.wy, v * m.wz, v * m.ww);
}

static inline mat4 mat4_zero()
{
    return mat4(
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f);
}

static inline mat4 mat4_transpose(mat4 m)
{
    return mat4(
        m.xx, m.yx, m.zx, m.wx,
        m.xy, m.yy, m.zy, m.wy,
        m.xz, m.yz, m.zz, m.wz,
        m.xw, m.yw, m.zw, m.ww);
}

static inline mat4 mat4_mul(mat4 m, mat4 n)
{
  return mat4(
      dot(m.r0(), n.c0()), dot(m.r0(), n.c1()), dot(m.r0(), n.c2()), dot(m.r0(), n.c3()),
      dot(m.r1(), n.c0()), dot(m.r1(), n.c1()), dot(m.r1(), n.c2()), dot(m.r1(), n.c3()),
      dot(m.r2(), n.c0()), dot(m.r2(), n.c1()), dot(m.r2(), n.c2()), dot(m.r2(), n.c3()),
      dot(m.r3(), n.c0()), dot(m.r3(), n.c1()), dot(m.r3(), n.c2()), dot(m.r3(), n.c3()));
}

static inline mat4 mat4_transpose_mul(mat4 m, mat4 n)
{
  return mat4(
      dot(m.c0(), n.c0()), dot(m.c0(), n.c1()), dot(m.c0(), n.c2()), dot(m.c0(), n.c3()),
      dot(m.c1(), n.c0()), dot(m.c1(), n.c1()), dot(m.c1(), n.c2()), dot(m.c1(), n.c3()),
      dot(m.c2(), n.c0()), dot(m.c2(), n.c1()), dot(m.c2(), n.c2()), dot(m.c2(), n.c3()),
      dot(m.c3(), n.c0()), dot(m.c3(), n.c1()), dot(m.c3(), n.c2()), dot(m.c3(), n.c3()));
}

static inline vec4 mat4_mul_vec4(mat4 m, vec4 v)
{
    return vec4(
        dot(m.r0(), v),
        dot(m.r1(), v),
        dot(m.r2(), v),
        dot(m.r3(), v));
}

static inline vec4 mat4_transpose_mul_vec4(mat4 m, vec4 v)
{
    return vec4(
        dot(m.c0(), v),
        dot(m.c1(), v),
        dot(m.c2(), v),
        dot(m.c3(), v));
}

static inline mat4 mat4_outer(vec4 v, vec4 w)
{
    return mat4(
        v.x * w.x, v.x * w.y, v.x * w.z, v.x * w.w,
        v.y * w.x, v.y * w.y, v.y * w.z, v.y * w.w,
        v.z * w.x, v.z * w.y, v.z * w.z, v.z * w.w,
        v.w * w.x, v.w * w.y, v.w * w.z, v.w * w.w);
}

static inline vec4 mat4_svd_dominant_eigen(
    const mat4 A, 
    const vec4 v0,
    const int iterations, 
    const float eps)
{
    // Initial Guess at Eigen Vector & Value
    vec4 v = v0;
    float ev = (mat4_mul_vec4(A, v) / v).x;
    
    for (int i = 0; i < iterations; i++)
    {
        // Power Iteration
        vec4 Av = mat4_mul_vec4(A, v);
        
        // Next Guess at Eigen Vector & Value
        vec4 v_new = normalize(Av);
        float ev_new = (mat4_mul_vec4(A, v_new) / v_new).x;
        
        // Break if converged
        if (fabs(ev - ev_new) < eps)
        {
            break;
        }
        
        // Update best guess
        v = v_new;
        ev = ev_new;
    }
    
    return v;
}

static inline void mat4_svd_piter(
    mat4& U,
    vec4& s,
    mat4& V,
    const mat4 A, 
    const int iterations = 64,
    const float eps = 1e-5f)
{
    // First Eigen Vector
    vec4 g0 = vec4(1, 0, 0, 0);
    mat4 B0 = A;
    vec4 u0 = mat4_svd_dominant_eigen(B0, g0, iterations, eps);
    vec4 v0_unnormalized = mat4_transpose_mul_vec4(A, u0);
    float s0 = length(v0_unnormalized);
    vec4 v0 = s0 < eps ? g0 : normalize(v0_unnormalized);

    // Second Eigen Vector
    mat4 B1 = A;
    vec4 g1 = vec4(0, 1, 0, 0);
    B1 = B1 - s0 * mat4_outer(u0, v0);
    vec4 u1 = mat4_svd_dominant_eigen(B1, g1, iterations, eps);
    vec4 v1_unnormalized = mat4_transpose_mul_vec4(A, u1);
    float s1 = length(v1_unnormalized);
    vec4 v1 = s1 < eps ? g1 : normalize(v1_unnormalized);
    
    // Third Eigen Vector
    mat4 B2 = A;
    vec4 v2 = vec4(0, 0, 1, 0);
    B2 = B2 - s0 * mat4_outer(u0, v0);
    B2 = B2 - s1 * mat4_outer(u1, v1);
    vec4 u2 = mat4_svd_dominant_eigen(B2, v2, iterations, eps);
    float s2 = length(mat4_transpose_mul_vec4(A, u2));
    
    // Forth Eigen Vector
    mat4 B3 = A;
    vec4 v3 = vec4(0, 0, 0, 1);
    B3 = B3 - s0 * mat4_outer(u0, v0);
    B3 = B3 - s1 * mat4_outer(u1, v1);
    B3 = B3 - s2 * mat4_outer(u2, v2);
    vec4 u3 = mat4_svd_dominant_eigen(B3, v3, iterations, eps);
    float s3 = length(mat4_transpose_mul_vec4(A, u3));
    
    // Done
    U = mat4(u0, u1, u2, u3);
    s = vec4(s0, s1, s2, s3);
    V = mat4(v0, v1, v2, v3);
}