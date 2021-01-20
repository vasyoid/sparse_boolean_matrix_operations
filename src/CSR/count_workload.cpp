#include "count_workload.hpp"
#include "../utils.hpp"

void count_workload(Controls &controls,
                    cl::Buffer &workload_out,
                    cl::Buffer &a_rows_pointers,
                    const cl::Buffer &a_cols,
                    cl::Buffer &b_rows_pointers,
                    const cl::Buffer &b_cols,
                    uint32_t rows_cnt) {

    // буффер с распределением рабочей нагрузки, равен числу строк матрицы A
    cl::Program program;
    try {
        cl::Buffer workload(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * rows_cnt);
        program = controls.create_program_from_file("../src/CSR/cl/count_workload.cl");
        uint32_t block_size = controls.block_size;

        std::stringstream options;
        options << "-D GROUP_SIZE=" << block_size;
        program.build(options.str().c_str());


        uint32_t work_group_size = block_size;
        uint32_t global_work_size = utils::calculate_global_size(work_group_size, rows_cnt);


        cl::Kernel count_workload_kernel(program, "count_workload");
        cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> count_workload(
            count_workload_kernel);

        cl::EnqueueArgs eargs(controls.queue, cl::NDRange(global_work_size), cl::NDRange(work_group_size));

        count_workload(eargs, workload, a_rows_pointers, a_cols, b_rows_pointers);

        workload_out = std::move(workload);

    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << "\n";
        if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
            exception << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(controls.device);
        }
        throw std::runtime_error(exception.str());
    }
}
