#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>

std::vector<float> read_floats(const std::string& path){
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "Failed to open " << path << "\n";
        std::exit(1);
    }
    std::vector<float> v;
    float x;
    while (fin >> x) v.push_back(x);
    return v;
}

DiffStats compare_arrays(const std::vector<float>& out,
                         const std::vector<float>& ref,
                         double tol){
    if (out.size() != ref.size()){
        std::cerr << "Size mismatch: out=" << out.size()
                  << " ref=" << ref.size() << "\n";
        std::exit(1);
    }
    DiffStats s; s.tol = tol;
    double sum = 0.0;
    for (size_t i=0;i<out.size();++i){
        double d = std::abs((double)out[i] - (double)ref[i]);
        if (d > s.max_abs) s.max_abs = d;
        sum += d;
        if (s.first_bad_idx < 0 && d > tol) s.first_bad_idx = (int)i;
    }
    s.mean_abs = sum / (double)out.size();
    return s;
}

void print_stats(const DiffStats& s,
                 const std::vector<float>& out,
                 const std::vector<float>& ref){
    std::cout.setf(std::ios::fixed);
    std::cout.precision(8);
    std::cout << "Max |diff| = " << s.max_abs
              << ", Mean |diff| = " << s.mean_abs << "\n";
    if (s.first_bad_idx >= 0){
        int i = s.first_bad_idx;
        std::cout << "First idx over tol(" << s.tol << "): " << i
                  << "  out=" << out[i]
                  << "  ref=" << ref[i] << "\n";
    } else {
        std::cout << "All within tolerance (" << s.tol << ").\n";
    }
}