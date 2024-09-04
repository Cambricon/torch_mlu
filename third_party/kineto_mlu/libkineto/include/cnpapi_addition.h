#ifndef CNPAPI_ADDITION_H_
#define CNPAPI_ADDITION_H_

// TODO(fuwenguang): Delete this header when SYSTOOL-4476 done.

#include <stdint.h>

#include "cnpapi_types.h"
#include "cnpapi_activity_api.h"

#ifdef __cplusplus
extern "C" {
#endif

CNPAPI_EXPORT cnpapiResult cnpapiIsKernelTypeTCDP(CNkernel kernel, uint8_t* flag);

#ifdef __cplusplus
}
#endif

#endif  // CNPAPI_ADDITION_H_
