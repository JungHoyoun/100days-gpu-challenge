#pragma once
#include <vector>
#include <string>

std::vector<float> read_floats(const std::string& path);

struct DiffStats {
    double max_abs = 0.0;
    double mean_abs = 0.0;
    int first_bad_idx = -1;
    double tol = 1e-5;
};

DiffStats compare_arrays(const std::vector<float>& out,
                         const std::vector<float>& ref,
                         double tol = 1e-5);

void print_stats(const DiffStats& s,
                 const std::vector<float>& out,
                 const std::vector<float>& ref);
