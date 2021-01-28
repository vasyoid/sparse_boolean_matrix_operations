#pragma once

#include <iostream>
#include <vector>
#include <cstdint>

#include <random>
#include <algorithm>

#include "../fast_random.h"
#include "../library_classes/matrix_csr.hpp"

namespace csr_utils {
    void generate_csr(std::vector<uint32_t> &cols, std::vector<uint32_t> &row_inds,
                      uint32_t n, uint32_t m, FastRandom rand);

    void print_csr(const std::vector<uint32_t> &cols, const std::vector<uint32_t> &row_inds, uint32_t n, uint32_t m);

    void print_dense(const std::vector<std::vector<uint8_t>>& mat);

    void multiply_dense(const std::vector<std::vector<uint8_t>>& a,
                        const std::vector<std::vector<uint8_t>>& b,
                        std::vector<std::vector<uint8_t>>& c);

    void add_dense(const std::vector<std::vector<uint8_t>>& a,
                   const std::vector<std::vector<uint8_t>>& b,
                   std::vector<std::vector<uint8_t>>& c);

    void csr_to_dense(const std::vector<uint32_t>& cols,
                      const std::vector<uint32_t>& row_inds,
                      uint32_t n, uint32_t m,
                      std::vector<std::vector<uint8_t>>& result);

    void multiply_csr_simple(const std::vector<uint32_t>& cols1,
                             const std::vector<uint32_t>& row_inds1,
                             const std::vector<uint32_t>& cols2,
                             const std::vector<uint32_t>& row_inds2,
                             std::vector<uint32_t>& cols3,
                             std::vector<uint32_t>& row_inds3);

    uint32_t hash_operation(std::vector<int32_t>& table, int32_t key, uint32_t nz);

    uint32_t count_intermediate(const std::vector<uint32_t>& cols1,
                                const std::vector<uint32_t>& row_inds1,
                                const std::vector<uint32_t>& row_inds2,
                                uint32_t row);

    void multiply_csr_hash_table(const std::vector<uint32_t>& cols1,
                                 const std::vector<uint32_t>& row_inds1,
                                 const std::vector<uint32_t>& cols2,
                                 const std::vector<uint32_t>& row_inds2,
                                 std::vector<uint32_t>& cols3,
                                 std::vector<uint32_t>& row_inds3);

    void add_csr_hash_table(const std::vector<uint32_t>& cols1,
                            const std::vector<uint32_t>& row_inds1,
                            const std::vector<uint32_t>& cols2,
                            const std::vector<uint32_t>& row_inds2,
                            std::vector<uint32_t>& cols3,
                            std::vector<uint32_t>& row_inds3);

    void write_buffer(Controls& controls, const std::vector<uint32_t>& buffer_cpu, cl::Buffer& buffer_gpu);

    void read_buffer(Controls& controls, std::vector<uint32_t>& buffer_cpu, cl::Buffer& buffer_gpu);
}