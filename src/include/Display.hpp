#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include <iostream>
#include <vector>
#include <string>
#include <type_traits> // Required for trait logic

namespace Debug {

    // --- 1. TYPE TRAITS (Must come first) ---
    template <typename T>
    struct is_vector : std::false_type {};
    
    template <typename T>
    struct is_vector<std::vector<T>> : std::true_type {};
    
    template <typename T>
    inline constexpr bool is_vector_v = is_vector<T>::value;

    // --- 2. SHAPE LOGIC ---

    // Base case: We reached the actual data (float/int), stop recursion.
    template <typename T>
    void get_shape_dims(const T&) {
        // No more dimensions to print
    }

    // Recursive case: We found another vector layer.
    template <typename T>
    void get_shape_dims(const std::vector<T>& vec) {
        if (vec.empty()) {
            std::cout << "0";
            return;
        }
        std::cout << vec.size();
        
        // Use if constexpr to peel through dimensions
        if constexpr (is_vector_v<T>) {
            std::cout << ", ";
            get_shape_dims(vec[0]);
        }
    }

    // Public Shape Function
    template <typename T>
    void shape(const std::vector<T>& vec) {
        std::cout << "Shape: (";
        get_shape_dims(vec);
        std::cout << ")" << std::endl;
    }

    // --- 3. DISPLAY FUNCTIONS ---

    // 1D Case
    template <typename T>
    void display(const std::vector<T>& vec, int indent = 0) {
        if (indent == 0) shape(vec);
        std::string padding(indent, ' ');
        std::cout << padding << "[ ";
        for (const auto& val : vec) std::cout << val << " ";
        std::cout << "]" << std::endl;
    }

    // N-Dimensional Case
    template <typename T>
    void display(const std::vector<std::vector<T>>& vec, int indent = 0) {
        if (indent == 0) shape(vec);
        std::string padding(indent, ' ');
        std::cout << padding << "[" << std::endl;
        for (const auto& sub_vec : vec) {
            display(sub_vec, indent + 2);
        }
        std::cout << padding << "]" << std::endl;
    }

} // namespace Debug

#endif