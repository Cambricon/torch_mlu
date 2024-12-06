if [ -z "$TORCH_MLU_HOME" ]; then
  PACKAGE_NAME="torch_mlu"

  PACKAGE_PATH=$(pip show "$PACKAGE_NAME" 2>/dev/null | grep -i "^Location:" | awk '{print $2}')
  export TORCH_MLU_HOME="${PACKAGE_PATH}/${PACKAGE_NAME}/"
  echo "TORCH_MLU_HOME ${TORCH_MLU_HOME}"
fi
if [ -z "$Torch_DIR" ]; then
  PACKAGE_NAME="torch"

  PACKAGE_PATH=$(pip show "$PACKAGE_NAME" 2>/dev/null | grep -i "^Location:" | awk '{print $2}')
  export Torch_DIR="${PACKAGE_PATH}/${PACKAGE_NAME}/"
fi
if ! [ -e "$TORCH_MLU_HOME" ]; then
    echo "Please set a correct TORCH_MLU_HOME to compile"
fi
if ! [ -e "$Torch_DIR" ]; then
    echo "Please set a correct Torch_DIR to compile"
fi
cmake -DCMAKE_PREFIX_PATH=${Torch_DIR} -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON ..
cmake --build . --config Release
