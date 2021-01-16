__kernel void local_bitonic_begin(__global unsigned int* data, unsigned int n) {
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);
    unsigned int work_size = GROUP_SIZE * 2;
    __local unsigned int local_data[GROUP_SIZE * 2];

    unsigned int tmp = 0;

    unsigned int read_idx = work_size * group_id + local_id;

    local_data[local_id] = read_idx < n ? data[read_idx] : 0;

    read_idx += GROUP_SIZE;

    local_data[local_id + GROUP_SIZE] = read_idx < n ? data[read_idx] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int last_group = n / work_size;
    unsigned int real_array_size = (group_id == last_group && (n % work_size)) ? (n % work_size) : work_size;
    unsigned int outer = pow(2, ceil(log2((float) real_array_size)));

    unsigned int segment_length = 2;
    while (outer != 1) {
        unsigned int local_line_id = local_id % (segment_length / 2);
        unsigned int local_twin_id = segment_length - local_line_id - 1;
        unsigned int group_line_id = local_id / (segment_length / 2);
        unsigned int line_id = segment_length * group_line_id + local_line_id;
        unsigned int twin_id = segment_length * group_line_id + local_twin_id;

        if (twin_id < real_array_size && local_data[line_id] > local_data[twin_id]) {
            tmp = local_data[line_id];
            local_data[line_id] = local_data[twin_id];
            local_data[twin_id] = tmp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int j = segment_length / 2; j > 1; j >>= 1) {
            local_line_id = local_id % (j / 2);
            local_twin_id = local_line_id + (j / 2);
            group_line_id = local_id / (j / 2);
            line_id = j * group_line_id + local_line_id;
            twin_id = j * group_line_id + local_twin_id;
            if (twin_id < real_array_size && local_data[line_id] > local_data[twin_id]) {
                tmp = local_data[line_id];
                local_data[line_id] = local_data[twin_id];
                local_data[twin_id] = tmp;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        outer >>= 1;
        segment_length <<= 1;
    }

    unsigned int glob_id = get_global_id(0);

    unsigned int write_idx = work_size * group_id + local_id;
    if (write_idx < n) {
        data[write_idx] = local_data[local_id];
    }

    write_idx += GROUP_SIZE;
    if (write_idx < n) {
        data[write_idx] = local_data[local_id + GROUP_SIZE];
    }
}


__kernel void bitonic_global_step(__global unsigned int* data,
                                  unsigned int segment_length,
                                  unsigned int mirror,
                                  unsigned int n) {
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_line_id = global_id % (segment_length / 2);
    unsigned int local_twin_id = mirror ? segment_length - local_line_id - 1 : local_line_id + (segment_length / 2);
    unsigned int group_line_id = global_id / (segment_length / 2);
    unsigned int line_id = segment_length * group_line_id + local_line_id;
    unsigned int twin_id = segment_length * group_line_id + local_twin_id;

    unsigned int tmp = 0;
    if ((twin_id < n) && data[line_id] > data[twin_id]) {
        tmp = data[line_id];
        data[line_id] = data[twin_id];
        data[twin_id] = tmp;
    }
}

__kernel void bitonic_local_endings(__global unsigned int* data, unsigned int n) {
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int work_size = GROUP_SIZE * 2;

    __local unsigned int local_data[GROUP_SIZE * 2];

    unsigned int tmp = 0;

    unsigned int read_idx = work_size * group_id + local_id;

    local_data[local_id] = read_idx < n ? data[read_idx] : 0;

    read_idx += GROUP_SIZE;

    local_data[local_id + GROUP_SIZE] = read_idx < n ? data[read_idx] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int segment_length = work_size;
    unsigned int last_group = n / work_size;
    unsigned int real_array_size = (group_id == last_group && (n % work_size)) ? (n % work_size) : work_size;

    for (unsigned int j = segment_length; j > 1; j >>= 1) {
        unsigned int local_line_id = local_id % (j / 2);
        unsigned int local_twin_id = local_line_id + (j / 2);
        unsigned int group_line_id = local_id / (j / 2);
        unsigned int line_id = j * group_line_id + local_line_id;
        unsigned int twin_id = j * group_line_id + local_twin_id;

        if (twin_id < real_array_size && local_data[line_id] > local_data[twin_id]) {
            tmp = local_data[line_id];
            local_data[line_id] = local_data[twin_id];
            local_data[twin_id] = tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    unsigned int write_idx = work_size * group_id + local_id;
    if (write_idx < n) {
        data[write_idx] = local_data[local_id];
    }

    write_idx += GROUP_SIZE;
    if (write_idx < n) {
        data[write_idx] = local_data[local_id + GROUP_SIZE];
    }
}
