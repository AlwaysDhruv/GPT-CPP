#ifndef DISPLAY_HPP
#define DISPLAY_HPP

#include <iostream>
#include <vector>
#include <string>

namespace Debug {

    // Base Case: Display a 1D vector
    template <typename T>
    void display(const std::vector<T>& vec, int indent = 0) {
        std::string padding(indent, ' ');
        std::cout << padding << "[ ";
        for (const auto& val : vec) {
            std::cout << val << " ";
        }
        std::cout << "]" << std::endl;
    }

    // Recursive Case: Display N-Dimensional vectors (2D, 3D, 4D, 5D)
    template <typename T>
    void display(const std::vector<std::vector<T>>& vec, int indent = 0) {
        std::string padding(indent, ' ');
        std::cout << padding << "[" << std::endl;
        for (const auto& sub_vec : vec) {
            // Recurse to the next level down
            display(sub_vec, indent + 2);
        }
        std::cout << padding << "]" << std::endl;
    }

} // namespace Debug

#endif