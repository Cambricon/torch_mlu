# - Try to find CNML
#
# The following are set after configuration is done:
#  CNRT_FOUND
#  CNRT_INCLUDE_DIRS
#  CNRT_LIBRARIES

include(FindPackageHandleStandardArgs)

SET(CNRT_INCLUDE_SEARCH_PATHS $ENV{NEUWARE_HOME}/include)
SET(CNRT_LIB_SEARCH_PATHS  $ENV{NEUWARE_HOME}/lib64)

find_path(CNRT_INCLUDE_DIR NAMES cnrt.h
          PATHS ${CNRT_INCLUDE_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_path(CNRT_INCLUDE_DIR NAMES cnrt.h
          NO_CMAKE_FIND_ROOT_PATH)

find_library(CNRT_LIBRARY NAMES cnrt
          PATHS ${CNRT_LIB_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_library(CNRT_LIBRARY NAMES cnrt
          NO_CMAKE_FIND_ROOT_PATH)


find_package_handle_standard_args(CNRT DEFAULT_MSG CNRT_INCLUDE_DIR CNRT_LIBRARY)

if(CNRT_FOUND)
  set(CNRT_INCLUDE_DIRS ${CNRT_INCLUDE_DIR})
  set(CNRT_LIBRARIES ${CNRT_LIBRARY})

  mark_as_advanced(CNRT_ROOT_DIR CNRT_LIBRARY_RELEASE CNRT_LIBRARY_DEBUG
    CNRT_LIBRARY CNRT_INCLUDE_DIR)
endif()
