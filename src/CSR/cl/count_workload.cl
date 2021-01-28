__kernel void count_workload(__global unsigned int* workload,
                             __global const unsigned int* a_rows_pointers,
                             __global const unsigned int* a_cols,
                             __global const unsigned int* b_rows_pointers,
                             unsigned int a_nzr) {
    uint global_id = get_global_id(0);
    if (global_id >= a_nzr) return;

    uint start = a_rows_pointers[global_id];
    uint end = a_rows_pointers[global_id + 1];

    uint result = 0;
    for (uint col_ind = start; col_ind < end; ++col_ind) {
        uint col = a_cols[col_ind];
        result += (b_rows_pointers[col + 1] - b_rows_pointers[col]);
    }
    workload[global_id] = result;
}
