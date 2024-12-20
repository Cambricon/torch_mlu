#pragma once
#include <cnrt.h>
template <typename T>
void bang_sigmoid_kernel_entry(
    cnrtQueue_t queue,
    T* d_dst,
    T* d_src,
    int elem_count);
