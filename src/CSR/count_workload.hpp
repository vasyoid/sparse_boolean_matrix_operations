#pragma once

#include "../cl_includes.hpp"
#include "../library_classes/controls.hpp"

void count_workload(Controls &controls,
                    cl::Buffer &workload_out,
                    cl::Buffer &a_rows_pointers,
                    const cl::Buffer &a_cols,
                    cl::Buffer &b_rows_pointers,
                    const cl::Buffer &b_cols,
                    uint32_t rows_cnt);