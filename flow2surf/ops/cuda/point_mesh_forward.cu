__global__ void point_mesh_distance(@ARGS_DEF, bool PointQuery) {
    @PRECALC
    const int query = blockIdx.x;
    const int count = PointQuery ? in1_shape0 : in0_shape0;
    __shared__ double minimum[256];
    double best = INFINITY;
    for (int j = threadIdx.x; j < count; j += blockDim.x) {
        const int point = PointQuery ? query : j;
        const int face = PointQuery ? j : query;
        best = fmin(best, point_mesh::triangle_distance(
            point_mesh::load(in0_p + point * 3), in1_p + face * 9));
    }
    minimum[threadIdx.x] = best;
    __syncthreads();
    for (int offset = 128; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            minimum[threadIdx.x] = fmin(minimum[threadIdx.x],
                minimum[threadIdx.x + offset]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (PointQuery) @out0(query) = minimum[0];
        else @out1(query) = minimum[0];
    }
}

point_mesh_distance<<<in0_shape0, 256>>>(@ARGS, true);
point_mesh_distance<<<in1_shape0, 256>>>(@ARGS, false);
