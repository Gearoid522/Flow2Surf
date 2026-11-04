__global__ void nearest_distance(@ARGS_DEF, bool FirstQuery) {
    @PRECALC
    using Scalar = in0_type;
    const int M = in0_shape1, N = in1_shape1;
    const int queries = FirstQuery ? M : N;
    const int references = FirstQuery ? N : M;
    const int query_blocks = (queries + 255) / 256;
    const int b = blockIdx.x / query_blocks;
    const int q = (blockIdx.x - b * query_blocks) * 256 + threadIdx.x;
    __shared__ Scalar reference_x[256];
    __shared__ Scalar reference_y[256];
    __shared__ Scalar reference_z[256];

    Scalar x = 0, y = 0, z = 0;
    if (q < queries) {
        x = FirstQuery ? @in0(b,q,0) : @in1(b,q,0);
        y = FirstQuery ? @in0(b,q,1) : @in1(b,q,1);
        z = FirstQuery ? @in0(b,q,2) : @in1(b,q,2);
    }
    Scalar best = INFINITY;
    int best_index = references;
    for (int start = 0; start < references; start += 256) {
        const int loaded = start + threadIdx.x;
        if (loaded < references) {
            reference_x[threadIdx.x] = FirstQuery ? @in1(b,loaded,0) : @in0(b,loaded,0);
            reference_y[threadIdx.x] = FirstQuery ? @in1(b,loaded,1) : @in0(b,loaded,1);
            reference_z[threadIdx.x] = FirstQuery ? @in1(b,loaded,2) : @in0(b,loaded,2);
        }
        __syncthreads();
        if (q < queries) {
            const int count = references - start < 256 ? references - start : 256;
            for (int offset = 0; offset < count; ++offset) {
                const Scalar dx = x - reference_x[offset];
                const Scalar dy = y - reference_y[offset];
                const Scalar dz = z - reference_z[offset];
                const Scalar distance = dx * dx + dy * dy + dz * dz;
                const int reference = start + offset;
                if (distance < best || (distance == best && reference < best_index)) {
                    best = distance;
                    best_index = reference;
                }
            }
        }
        __syncthreads();
    }
    if (q < queries) {
        if (FirstQuery) {
            @out0(b,q) = best;
            @out1(b,q) = best_index;
        } else {
            @out2(b,q) = best;
            @out3(b,q) = best_index;
        }
    }
}

nearest_distance<<<in0_shape0 * ((in0_shape1 + 255) / 256), 256>>>(@ARGS, true);
nearest_distance<<<in1_shape0 * ((in1_shape1 + 255) / 256), 256>>>(@ARGS, false);
