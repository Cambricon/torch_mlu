# - Try to find CNCL
#
# The following are set after configuration is done:
#  CNCL_FOUND
#  CNCL_INCLUDE_DIRS
#  CNCL_LIBRARIES

include(FindPackageHandleStandardArgs)

SET(CNCL_LIB_SEARCH_PATHS $ENV{NEUWARE_HOME}/lib64)
SET(CNCL_INCLUDE_SEARCH_PATHS $ENV{NEUWARE_HOME}/include)

find_library(CNCL_LIBRARY NAMES cncl
             PATHS ${CNCL_LIB_SEARCH_PATHS}
             NO_DEFAULT_PATH)
find_library(CNCL_LIBRARY NAMES cncl
             NO_CMAKE_FIND_ROOT_PATH)

find_path(CNCL_INCLUDE_DIR NAMES cncl.h
          PATHS ${CNCL_INCLUDE_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_path(CNCL_INCLUDE_DIR NAMES cncl.h
          NO_CMAKE_FIND_ROOT_PATH)

find_package_handle_standard_args(CNCL DEFAULT_MSG CNCL_INCLUDE_DIR CNCL_LIBRARY)

if(CNCL_FOUND)
  set(CNCL_INCLUDE_DIRS ${CNCL_INCLUDE_DIR})
  set(CNCL_LIBRARIES ${CNCL_LIBRARY})
  mark_as_advanced(CNCL_INCLUDE_DIR CNCL_LIBRARY)
endif()
