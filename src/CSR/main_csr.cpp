#include <iostream>
#include <vector>
#include <cstdint>

#include <random>
#include <algorithm>

void generate_csr(std::vector<uint32_t> &cols, std::vector<uint32_t> &row_inds,
                  uint32_t n, uint32_t m, std::mt19937 rand) {
    cols.clear();
    row_inds.clear();

    std::vector<std::pair<uint32_t, uint32_t>> mat;

    uint32_t s = n + m;
    uint32_t nz = (rand() % s) + s / 2;

    mat.reserve(nz);
    for (uint32_t i = 0; i < nz; ++i) {
        mat.emplace_back(rand() % n, rand() % m);
    }

    std::sort(mat.begin(), mat.end());

    mat.resize(std::unique(mat.begin(), mat.end()) - mat.begin());

    nz = mat.size();

    row_inds.push_back(0);
    uint32_t cur_row = 0;
    for (uint32_t i = 0; i < nz; ++i) {
        cols.push_back(mat[i].second);
        while (cur_row < mat[i].first) {
            row_inds.push_back(i);
            ++cur_row;
        }
    }
    while (cur_row <= n) {
        row_inds.push_back(nz);
        ++cur_row;
    }
}

void print_csr(const std::vector<uint32_t> &cols, const std::vector<uint32_t> &row_inds, uint32_t n, uint32_t m) {
    uint32_t col = 0;
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < m; ++j) {
            if (row_inds[i] < row_inds[i + 1] && col < row_inds[i + 1] && j == cols[col]) {
                std::cout << 1;
                ++col;
            } else {
                std::cout << 0;
            }
        }
        std::cout << std::endl;
    }
}

void print_dense(const std::vector<std::vector<uint8_t>>& mat) {
    for (const auto& row : mat) {
        for (uint8_t x : row) {
            std::cout << int(x);
        }
        std::cout << std::endl;
    }
}

void multiply_dense(const std::vector<std::vector<uint8_t>>& a, const std::vector<std::vector<uint8_t>>& b,
                    std::vector<std::vector<uint8_t>>& c) {
    c.resize(a.size());
    for (uint32_t i = 0; i < a.size(); ++i) {
        for (uint32_t j = 0; j < b[0].size(); ++j) {
            c[i].push_back(0);
            for (uint32_t k = 0; k < b.size(); ++k) {
                c[i][j] |= a[i][k] & b[k][j];
            }
        }
    }
}

void csr_to_dense(const std::vector<uint32_t>& cols,
                  const std::vector<uint32_t>& row_inds,
                  uint32_t n, uint32_t m,
                  std::vector<std::vector<uint8_t>>& result) {
    result.resize(n);
    uint32_t col = 0;
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < m; ++j) {
            if (row_inds[i] < row_inds[i + 1] && col < row_inds[i + 1] && j == cols[col]) {
                result[i].push_back(1);
                ++col;
            } else {
                result[i].push_back(0);
            }
        }
    }
}

void multiply_csr(const std::vector<uint32_t>& cols1,
                  const std::vector<uint32_t>& row_inds1,
                  const std::vector<uint32_t>& cols2,
                  const std::vector<uint32_t>& row_inds2,
                  std::vector<uint32_t>& cols3,
                  std::vector<uint32_t>& row_inds3) {
    uint32_t n = row_inds1.size() - 1;
    row_inds3.push_back(0);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t ind1 = row_inds1[i]; ind1 < row_inds1[i + 1]; ++ind1) {
            uint32_t k = cols1[ind1];
            for (uint32_t ind2 = row_inds2[k]; ind2 < row_inds2[k + 1]; ++ind2) {
                uint32_t j = cols2[ind2];
                auto it = lower_bound(cols3.begin() + row_inds3.back(), cols3.end(), j);
                if (it != cols3.end() && *it == j) continue;
                cols3.insert(it, j);
            }
        }
        row_inds3.push_back(cols3.size());
    }
}

const uint32_t HASH_SCAL = 9973;

uint32_t hash_operation(std::vector<int32_t>& table, int32_t key, uint32_t nz) {
  uint32_t t_size = table.size();
  uint32_t hash = (key * HASH_SCAL) % t_size;
  while (true) {
    if (table[hash] == key) {
      break;
    } else if (table[hash] == -1) {
        table[hash] = key;
        nz = nz + 1;
        break;
    } else {
      hash = (hash + 1) % t_size;
    }
  }
  return nz;
}

uint32_t count_intermediate(const std::vector<uint32_t>& cols1,
                            const std::vector<uint32_t>& row_inds1,
                            const std::vector<uint32_t>& row_inds2,
                            uint32_t row) {
  uint32_t result = 0;
  for (uint32_t col_ind = row_inds1[row]; col_ind < row_inds1[row + 1]; ++col_ind) {
    uint32_t col = cols1[col_ind];
    result += (row_inds2[col + 1] - row_inds2[col]);
  }
  return result;
}

void multiply_csr_hash_table(const std::vector<uint32_t>& cols1,
                  const std::vector<uint32_t>& row_inds1,
                  const std::vector<uint32_t>& cols2,
                  const std::vector<uint32_t>& row_inds2,
                  std::vector<uint32_t>& cols3,
                  std::vector<uint32_t>& row_inds3) {
    uint32_t n = row_inds1.size() - 1;
    row_inds3.push_back(0);

    for (uint32_t i = 0; i < n; ++i) {
      std::vector<int32_t> table(count_intermediate(cols1, row_inds1, row_inds2, i), -1);
      for (uint32_t ind1 = row_inds1[i]; ind1 < row_inds1[i + 1]; ++ind1) {
        uint32_t k = cols1[ind1];
        for (uint32_t ind2 = row_inds2[k]; ind2 < row_inds2[k + 1]; ++ind2) {
          hash_operation(table, cols2[ind2], 0);
        }
      }
      table.resize(std::remove(table.begin(), table.end(), -1) - table.begin());
      std::sort(table.begin(), table.end());
      cols3.insert(cols3.end(), table.begin(), table.end());
      row_inds3.push_back(cols3.size());
    }
}

bool dense_equal(const std::vector<std::vector<uint8_t>>& a, const std::vector<std::vector<uint8_t>>& b) {
    for (uint32_t i = 0; i < a.size(); ++i) {
        for (uint32_t j = 0; j < a[i].size(); ++j) {
            if (a[i][j] != b[i][j]) return false;
        }
    }
    return true;
}

bool test_multiply(uint32_t n, uint32_t m, uint32_t k, unsigned int seed) {
    std::mt19937 rand(seed);

    std::vector<uint32_t> cols1;
    std::vector<uint32_t> row_inds1;
    generate_csr(cols1, row_inds1, n, k, rand);

    std::vector<uint32_t> cols2;
    std::vector<uint32_t> row_inds2;
    generate_csr(cols2, row_inds2, k, m, rand);

    std::vector<std::vector<uint8_t>> mat1;
    csr_to_dense(cols1, row_inds1, n, k, mat1);
    std::vector<std::vector<uint8_t>> mat2;
    csr_to_dense(cols2, row_inds2, k, m, mat2);

    std::vector<std::vector<uint8_t>> expected;
    multiply_dense(mat1, mat2, expected);

    std::vector<uint32_t> cols3;
    std::vector<uint32_t> row_inds3;
    multiply_csr_hash_table(cols1, row_inds1, cols2, row_inds2, cols3, row_inds3);

    std::vector<std::vector<uint8_t>> actual;
    csr_to_dense(cols3, row_inds3, n, m, actual);
    if (!dense_equal(expected, actual)) {
        std::cout << seed << "\n";
        print_dense(mat1);
        std::cout << "*\n";
        print_dense(mat2);
        std::cout << "=\n";
        print_dense(expected);
        std::cout << "\n";
        print_csr(cols1, row_inds1, n, k);
        std::cout << "*\n";
        print_csr(cols2, row_inds2, k, m);
        std::cout << "=\n";
        print_csr(cols3, row_inds3, n, m);
        return false;
    }
    return true;
}

int main() {
    std::random_device rd;
    for (int i = 1; i <= 1000; ++i) {
        if (!test_multiply(i / 10 + 4, i / 7 + 5, i / 5 + 6, rd())) {
            exit(1);
        }
    }
    std::cout << "OK\n";
  return 0;
}