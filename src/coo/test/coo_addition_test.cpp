#include "coo_tests.hpp"
#include "../../cl_includes.hpp"
#include "../../library_classes/matrix_coo.hpp"
#include "../coo_utils.hpp"
#include "../coo_matrix_addition.hpp"

using coo_utils::matrix_coo_cpu;
void testMatrixAddition() {
    Controls controls = utils::create_controls();

    matrix_coo_cpu matrix_res_cpu;
    // first argument is pseudo size (size before reducing duplicates after random)
    // second is the maximum possible matrix size
    matrix_coo_cpu matrix_a_cpu = coo_utils::generate_random_matrix_cpu(475726, 10756);
    matrix_coo_cpu matrix_b_cpu = coo_utils::generate_random_matrix_cpu(667312, 3526);

    matrix_coo matrix_res_gpu;
    matrix_coo matrix_a_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_a_cpu);
    matrix_coo matrix_b_gpu = coo_utils::matrix_coo_from_cpu(controls, matrix_b_cpu);

    coo_utils::matrix_addition_cpu(matrix_res_cpu, matrix_a_cpu, matrix_b_cpu);

    matrix_addition(controls, matrix_res_gpu, matrix_a_gpu, matrix_b_gpu);

    std::vector<uint32_t> rows_cpu;
    std::vector<uint32_t> cols_cpu;

    coo_utils::get_vectors_from_cpu_matrix(rows_cpu, cols_cpu, matrix_res_cpu);

    if (matrix_res_gpu.rows_indices_cpu() == rows_cpu && matrix_res_gpu.cols_indices_cpu() == cols_cpu) {
        std::cout << "correct addition" << std::endl;
    } else {
        std::cerr << "incorrect addition" << std::endl;
    }

}
