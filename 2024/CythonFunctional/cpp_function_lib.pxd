# distutils: language=c++
# distutils: include_dirs= .

from libcpp.functional cimport function

cdef extern from "func.hpp":
    double add_one(double, int)
    double add_two(double a, int b)

    cdef cppclass AddAnotherFunctor:
        AddAnotherFunctor(double to_add)
        double call "operator()"(double a, int b)

    cdef cppclass FunctionKeeper:
        FunctionKeeper(function[double(double, int) noexcept] user_function)
        void set_function(function[double(double, int) noexcept] user_function)
        function[double(double, int) noexcept] get_function()
        double call_function(double a, int b) except +
