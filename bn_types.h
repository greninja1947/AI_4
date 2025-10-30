#ifndef BN_TYPES_H
#define BN_TYPES_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations and type definitions shared between learn_bn.cpp and em_algorithm.cpp

struct Variable {
    std::string name;
    std::vector<std::string> states;
    std::vector<std::string> parent_names;
    std::unordered_map<std::string, int> state_to_index;
    std::vector<int> parents;
    std::vector<int> children;
    std::vector<size_t> parent_strides;
    size_t cpt_offset = 0;
    size_t q = 1;
    int r = 0;
};

struct BN {
    std::vector<Variable> vars;
    std::unordered_map<std::string, int> name_to_index;
    size_t total_cpt_entries = 0;
    size_t max_states = 0;
};

struct Data {
    std::vector<int32_t> values;
    size_t nrows = 0;
    size_t nvars = 0;
    size_t complete_rows = 0;
    size_t single_missing_rows = 0;

    const int32_t* row(size_t idx) const {
        return values.data() + idx * nvars;
    }
};

struct Buckets {
    std::vector<size_t> complete;
    std::vector<std::vector<size_t>> per_var_sorted;
};

#endif // BN_TYPES_H
