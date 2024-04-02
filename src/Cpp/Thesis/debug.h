#ifndef DEBUGGING_FUNC
#define DEBUGGING_FUNC
#ifndef NOPRINTING
#include <iostream>
// https://stackoverflow.com/questions/22964799/how-to-print-a-variable-number-of-parameters-with-a-macro
#define PRINT(...) print(__VA_ARGS__)

// base case for template recursion when one argument remains
template <typename Arg1>
void print(Arg1&& arg1)
{
    std::cout << arg1 << std::endl;
}

// recursive variadic template for multiple arguments
template <typename Arg1, typename... Args>
void print(Arg1&& arg1, Args&&... args)
{
    std::cout << arg1 << " ";
    print(args...);
}
#else
#define PRINT(...)
#endif
#endif