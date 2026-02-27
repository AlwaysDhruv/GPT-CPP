#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include <iostream>
#include <vector>
#include <string>

namespace Debug {

    // --- 1. FORWARD DECLARATIONS ---
    // This tells the compiler these functions exist later in the file
    template <typename T> void print_inner_shape(const std::vector<T>& vec);
    template <typename T> void print_inner_shape(const std::vector<std::vector<T>>& vec);

    // --- 2. SHAPE LOGIC ---
    template <typename T>
    void print_shape(const std::vector<T>& vec) {
        std::cout << "Shape: (" << vec.size() << ")" << std::endl;
    }

    template <typename T>
    void print_shape(const std::vector<std::vector<T>>& vec) {
        std::cout << "Shape: (" << vec.size() << ", ";
        if (!vec.empty()) {
             print_inner_shape(vec[0]); // Now the compiler knows this exists!
        } else {
            std::cout << "0)" << std::endl;
        }
    }

    // --- 3. INNER SHAPE RECURSION ---
    template <typename T>
    void print_inner_shape(const std::vector<T>& vec) {
        std::cout << vec.size() << ")" << std::endl;
    }

    template <typename T>
    void print_inner_shape(const std::vector<std::vector<T>>& vec) {
        std::cout << vec.size() << ", ";
        if (!vec.empty()) print_inner_shape(vec[0]);
        else std::cout << "0)" << std::endl;
    }

    // --- 4. DISPLAY FUNCTIONS ---
    template <typename T>
    void display(const std::vector<T>& vec, int indent = 0) {
        if (indent == 0) print_shape(vec);
        std::string padding(indent, ' ');
        std::cout << padding << "[ ";
        for (const auto& val : vec) std::cout << val << " ";
        std::cout << "]" << std::endl;
    }

    template <typename T>
    void display(const std::vector<std::vector<T>>& vec, int indent = 0) {
        if (indent == 0) print_shape(vec);
        std::string padding(indent, ' ');
        std::cout << padding << "[" << std::endl;
        for (const auto& sub_vec : vec) {
            display(sub_vec, indent + 2);
        }
        std::cout << padding << "]" << std::endl;
    }

} // namespace Debug

#endif