#include <iostream>
#include <vector>
#include <cstdint>

#include "csr_utils.hpp"
#include "../utils.hpp"
#include "../library_classes/matrix_csr.hpp"
#include "bitonic_sort.hpp"
#include "count_workload.hpp"

bool test_multiply_cpu(uint32_t n, uint32_t m, uint32_t k, int seed) {
    FastRandom rand(seed);

    std::vector<uint32_t> cols1;
    std::vector<uint32_t> row_inds1;
    csr_utils::generate_csr(cols1, row_inds1, n, k, rand);

    std::vector<uint32_t> cols2;
    std::vector<uint32_t> row_inds2;
    csr_utils::generate_csr(cols2, row_inds2, k, m, rand);

    std::vector<std::vector<uint8_t>> mat1;
    csr_utils::csr_to_dense(cols1, row_inds1, n, k, mat1);
    std::vector<std::vector<uint8_t>> mat2;
    csr_utils::csr_to_dense(cols2, row_inds2, k, m, mat2);

    std::vector<std::vector<uint8_t>> expected;
    csr_utils::multiply_dense(mat1, mat2, expected);

    std::vector<uint32_t> cols3;
    std::vector<uint32_t> row_inds3;
    csr_utils::multiply_csr_hash_table(cols1, row_inds1, cols2, row_inds2, cols3, row_inds3);

    std::vector<std::vector<uint8_t>> actual;
    csr_utils::csr_to_dense(cols3, row_inds3, n, m, actual);
    if (expected != actual) {
        std::cout << seed << "\n";
        csr_utils::print_dense(mat1);
        std::cout << "*\n";
        csr_utils::print_dense(mat2);
        std::cout << "=\n";
        csr_utils::print_dense(expected);
        std::cout << "\n";
        csr_utils::print_csr(cols1, row_inds1, n, k);
        std::cout << "*\n";
        csr_utils::print_csr(cols2, row_inds2, k, m);
        std::cout << "=\n";
        csr_utils::print_csr(cols3, row_inds3, n, m);
        return false;
    }
    return true;
}

bool test_add_cpu(uint32_t n, uint32_t m, int seed) {
    FastRandom rand(seed);

    std::vector<uint32_t> cols1;
    std::vector<uint32_t> row_inds1;
    csr_utils::generate_csr(cols1, row_inds1, n, m, rand);

    std::vector<uint32_t> cols2;
    std::vector<uint32_t> row_inds2;
    csr_utils::generate_csr(cols2, row_inds2, n, m, rand);

    std::vector<std::vector<uint8_t>> mat1;
    csr_utils::csr_to_dense(cols1, row_inds1, n, m, mat1);
    std::vector<std::vector<uint8_t>> mat2;
    csr_utils::csr_to_dense(cols2, row_inds2, n, m, mat2);

    std::vector<std::vector<uint8_t>> expected;
    csr_utils::add_dense(mat1, mat2, expected);

    std::vector<uint32_t> cols3;
    std::vector<uint32_t> row_inds3;
    csr_utils::add_csr_hash_table(cols1, row_inds1, cols2, row_inds2, cols3, row_inds3);

    std::vector<std::vector<uint8_t>> actual;
    csr_utils::csr_to_dense(cols3, row_inds3, n, m, actual);
    if (expected != actual) {
        std::cout << seed << "\n";
        csr_utils::print_dense(mat1);
        std::cout << "+\n";
        csr_utils::print_dense(mat2);
        std::cout << "=\n";
        csr_utils::print_dense(expected);
        std::cout << "\n";
        csr_utils::print_csr(cols1, row_inds1, n, m);
        std::cout << "+\n";
        csr_utils::print_csr(cols2, row_inds2, n, m);
        std::cout << "=\n";
        csr_utils::print_csr(cols3, row_inds3, n, m);
        return false;
    }
    return true;
}

bool test_multiply_gpu(uint32_t n, uint32_t m, uint32_t k, unsigned int seed) {
    FastRandom rand(seed);
    Controls controls = utils::create_controls();

    std::vector<uint32_t> cols1;
    std::vector<uint32_t> row_inds1;
    csr_utils::generate_csr(cols1, row_inds1, n, k, rand);
    matrix_csr mat1_gpu = matrix_csr(controls, n, k, cols1.size(), row_inds1, cols1);

    std::vector<uint32_t> cols2;
    std::vector<uint32_t> row_inds2;
    csr_utils::generate_csr(cols2, row_inds2, k, m, rand);
    matrix_csr mat2_gpu(controls, k, m, cols1.size(), row_inds2, cols2);

    std::vector<uint32_t> cols_expected;
    std::vector<uint32_t> row_inds_expected;
    csr_utils::multiply_csr_hash_table(cols1, row_inds1, cols2, row_inds2, cols_expected, row_inds_expected);

    matrix_csr actual_gpu;
    //csr_utils::multiply_gpu(cols1, row_inds1, cols2, row_inds2, cols_expected, row_inds_expected);

    const std::vector<uint32_t>& cols_actual = actual_gpu.cols_indexes_cpu();
    const std::vector<uint32_t>& row_inds_actual = actual_gpu.rows_pointers_cpu();

    if (cols_expected != cols_actual || row_inds_expected != row_inds_actual) {
        std::cout << seed << "\nexpected:\n";
        csr_utils::print_csr(cols1, row_inds1, n, k);
        std::cout << "*\n";
        csr_utils::print_csr(cols2, row_inds2, k, m);
        std::cout << "=\n";
        csr_utils::print_csr(cols_expected, row_inds_expected, n, m);
        std::cout << "actual:\n";
        csr_utils::print_csr(cols_actual, row_inds_actual, n, m);
        return false;
    }
    return true;
}

bool test_bitonic_sort(uint32_t size, int seed) {
    FastRandom rand(seed);
    Controls controls = utils::create_controls();

    std::vector<uint32_t> data_cpu(size);

    for (auto& value : data_cpu) {
        value = rand.next(0, size);
    }

    std::vector<uint32_t> expected(data_cpu.begin(), data_cpu.end());
    std::sort(expected.begin(), expected.end());

    cl::Buffer data_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * size);
    controls.queue.enqueueWriteBuffer(data_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, data_cpu.data());

    sort(controls, data_gpu, size);

    std::vector<uint32_t> actual(size);
    controls.queue.enqueueReadBuffer(data_gpu, CL_TRUE, 0, sizeof(uint32_t) * size, actual.data());

    if (expected != actual) {
        std::cout << "incorrect sort: " << std::endl;
        for (auto& value : expected) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        for (auto& value : actual) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        return false;
    }
    return true;
}

bool test_count_workload(uint32_t n, uint32_t m, uint32_t k, unsigned int seed) {
    FastRandom rand(seed);
    Controls controls = utils::create_controls();

    std::vector<uint32_t> cols1;
    std::vector<uint32_t> row_inds1;
    csr_utils::generate_csr(cols1, row_inds1, n, k, rand);
    cl::Buffer a_rows_pointers_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n);
    cl::Buffer a_cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * n);
    csr_utils::write_buffer(controls, row_inds1, a_rows_pointers_gpu);
    csr_utils::write_buffer(controls, cols1, a_cols_gpu);

    std::vector<uint32_t> cols2;
    std::vector<uint32_t> row_inds2;
    csr_utils::generate_csr(cols2, row_inds2, k, m, rand);
    cl::Buffer b_rows_pointers_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * k);
    cl::Buffer b_cols_gpu(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * k);
    csr_utils::write_buffer(controls, row_inds2, b_rows_pointers_gpu);
    csr_utils::write_buffer(controls, cols2, b_cols_gpu);

    std::vector<uint32_t> expected;

    for (uint32_t row = 0; row < row_inds1.size() - 1; ++row) {
        expected.push_back(csr_utils::count_intermediate(cols1, row_inds1, row_inds2, row));
    }

    cl::Buffer actual_gpu;
    count_workload(controls, actual_gpu, a_rows_pointers_gpu, a_cols_gpu, b_rows_pointers_gpu, b_cols_gpu, n, cols1.size());

    std::vector<uint32_t> actual(n);
    csr_utils::read_buffer(controls, actual, actual_gpu);

    if (expected != actual) {
        for (auto& value : expected) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        for (auto& value : actual) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        return false;
    }
    return true;
}

int main() {
    FastRandom rand;
    for (int i = 1; i <= 3; ++i) {
        if (!test_count_workload(i * 2, i * 3, i * 4, rand.next())) {
            exit(1);
        }
    }
    std::cout << "OK\n";
  return 0;
}