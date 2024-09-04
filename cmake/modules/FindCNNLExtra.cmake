# - Try to find CNNLExtra
#
# The following are set after configuration is done:
#  CNNLExtra_FOUND
#  CNNLExtra_INCLUDE_DIRS
#  CNNLExtra_LIBRARIES

include(FindPackageHandleStandardArgs)

SET(CNNLExtra_INCLUDE_SEARCH_PATHS $ENV{NEUWARE_HOME}/include)
SET(CNNLExtra_LIB_SEARCH_PATHS $ENV{NEUWARE_HOME}/lib64)

find_path(CNNLExtra_INCLUDE_DIR NAMES cnnl_extra.h
          PATHS ${CNNLExtra_INCLUDE_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_path(CNNLExtra_INCLUDE_DIR NAMES cnnl_extra.h
          NO_CMAKE_FIND_ROOT_PATH)

find_library(CNNLExtra_LIBRARY NAMES cnnl_extra
          PATHS ${CNNLExtra_LIB_SEARCH_PATHS}
          NO_DEFAULT_PATH)
find_library(CNNLExtra_LIBRARY NAMES cnnl_extra
          NO_CMAKE_FIND_ROOT_PATH)

find_package_handle_standard_args(CNNLExtra DEFAULT_MSG CNNLExtra_INCLUDE_DIR CNNLExtra_LIBRARY)

if(CNNLExtra_FOUND)
  set(CNNLExtra_INCLUDE_DIRS ${CNNLExtra_INCLUDE_DIR})
  set(CNNLExtra_LIBRARIES ${CNNLExtra_LIBRARY})

  mark_as_advanced(CNNLExtra_INCLUDE_DIR CNNLExtra_LIBRARY)
endif()
