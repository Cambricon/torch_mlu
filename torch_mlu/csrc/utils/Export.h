#ifndef TORCH_MLU_MACROS_EXPORT_H_
#define TORCH_MLU_MACROS_EXPORT_H_

#if defined(__GNUC__)
#if __GNUC__ >= 4
#define TORCH_MLU_EXPORT __attribute__((__visibility__("default")))
#define TORCH_MLU_HIDDEN __attribute__((__visibility__("hidden")))
#endif
#else // defined(__GNUC__)
#define TORCH_MLU_EXPORT
#define TORCH_MLU_HIDDEN
#endif // defined(__GNUC__)

// This one is being used by libtorch_mlu.so
#define TORCH_MLU_API TORCH_MLU_EXPORT

#endif // TORCH_MLU_MACROS_EXPORT_H_
