# - Try to find CNPAPI
#
# The following are set after configuration is done:
#  CNPAPI_FOUND
#  CNPAPI_INCLUDE_DIRS
#  CNPAPI_LIBRARIES

include(FindPackageHandleStandardArgs)

SET(CNPAPI_LIB_SEARCH_PATHS $ENV{NEUWARE_HOME}/lib64)
SET(CNPAPI_INCLUDE_SEARCH_PATHS $ENV{NEUWARE_HOME}/include)

find_library(CNPAPI_LIBRARY NAMES cnpapi
             PATHS ${CNPAPI_LIB_SEARCH_PATHS}
             NO_DEFAULT_PATH)
find_library(CNPAPI_LIBRARY NAMES cnpapi
             NO_CMAKE_FIND_ROOT_PATH)

find_path(CNPAPI_INCLUDE_DIR NAMES cnpapi.h activity_api.h callbackapi.h callbackapi_types.h cnpapi_types.h cndrv_id.h cnml_id.h cnnl_id.h cnnl_extra_id.h cnpx_id.h cnrt_id.h
          PATHS ${CNPAPI_INCLUDE_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_path(CNPAPI_INCLUDE_DIR NAMES cnpapi.h activity_api.h callbackapi.h callbackapi_types.h cnpapi_types.h cndrv_id.h cnml_id.h cnnl_id.h cnnl_extra_id.h cnpx_id.h cnrt_id.h
          NO_CMAKE_FIND_ROOT_PATH)

find_package_handle_standard_args(CNPAPI DEFAULT_MSG CNPAPI_INCLUDE_DIR CNPAPI_LIBRARY)

if(CNPAPI_FOUND)
  set(CNPAPI_INCLUDE_DIRS ${CNPAPI_INCLUDE_DIR})
  set(CNPAPI_LIBRARIES ${CNPAPI_LIBRARY})
  mark_as_advanced(CNPAPI_INCLUDE_DIR CNPAPI_LIBRARY)
endif()
