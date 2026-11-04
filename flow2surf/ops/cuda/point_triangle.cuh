// FP64 squared distances to closed triangles and segments.
// No absolute area cutoff or additive denominator regularization.
#pragma once
#include <cuda_runtime.h>
#include <math.h>

namespace point_mesh {

struct Vec3 { double x, y, z; };

__device__ inline Vec3 load(const double* p) {
    return {p[0], p[1], p[2]};
}

__device__ inline Vec3 subtract(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ inline Vec3 divide(Vec3 a, double scale) {
    return {a.x / scale, a.y / scale, a.z / scale};
}

__device__ inline double dot(Vec3 a, Vec3 b) {
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
}

__device__ inline double difference_of_products(double a, double b,
                                               double c, double d) {
    const double cd = c * d;
    return fma(a, b, -cd) + fma(-c, d, cd);
}

__device__ inline Vec3 cross(Vec3 a, Vec3 b) {
    return {difference_of_products(a.y, b.z, a.z, b.y),
            difference_of_products(a.z, b.x, a.x, b.z),
            difference_of_products(a.x, b.y, a.y, b.x)};
}

__device__ inline double largest_component(Vec3 a) {
    return fmax(fabs(a.x), fmax(fabs(a.y), fabs(a.z)));
}

__device__ inline double segment_distance(Vec3 p, Vec3 a, Vec3 b) {
    Vec3 edge = subtract(b, a);
    const double scale = largest_component(edge);
    if (scale == 0.0) return dot(subtract(p, a), subtract(p, a));
    edge = divide(edge, scale);
    const Vec3 relative = subtract(p, a);
    const double t = fmin(1.0, fmax(0.0,
        dot(divide(relative, scale), edge) / dot(edge, edge)));
    const Vec3 delta = {fma(-t * scale, edge.x, relative.x),
                        fma(-t * scale, edge.y, relative.y),
                        fma(-t * scale, edge.z, relative.z)};
    return dot(delta, delta);
}

__device__ inline double triangle_distance(Vec3 p, const double* triangle) {
    const Vec3 a = load(triangle);
    const Vec3 b = load(triangle + 3);
    const Vec3 c = load(triangle + 6);
    Vec3 ab = subtract(b, a), ac = subtract(c, a);
    const double scale = fmax(largest_component(ab), largest_component(ac));
    if (scale == 0.0) return dot(subtract(p, a), subtract(p, a));
    ab = divide(ab, scale);
    ac = divide(ac, scale);
    const Vec3 ap = divide(subtract(p, a), scale);
    const Vec3 normal = cross(ab, ac);
    const double normal2 = dot(normal, normal);
    if (normal2 > 0.0) {
        // Signed subareas are unchanged by projection onto the face plane.
        const double wb = dot(cross(ap, ac), normal);
        const double wc = dot(cross(ab, ap), normal);
        if (wb >= 0.0 && wc >= 0.0 && wb + wc <= normal2) {
            const double height = dot(ap, normal) / sqrt(normal2) * scale;
            return height * height;
        }
    }
    // Exterior projection or collinear vertices: minimize over closed segments.
    return fmin(segment_distance(p, a, b),
        fmin(segment_distance(p, a, c), segment_distance(p, b, c)));
}

} // namespace point_mesh
