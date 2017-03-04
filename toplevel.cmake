set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}) # Findexternpro.cmake
set(externpro_INSTALLER_LOCATION "Installers located on hera C4ISR_Builds in externpro directory.")
set(externpro_REV 16.01.1)
set(internpro_REV 16.01.1)
find_package(externpro REQUIRED)
find_package(internpro REQUIRED)
xpEnforceOutOfSourceBuilds()
include(functions) # Shared/make/functions.cmake
find_package(CUDA QUIET)
if(CUDA_FOUND)
  # When we update to cmake 3.3, we can skip this workaround and enable
  # CUDA_USE_STATIC_CUDA_RUNTIME.
  find_library(CUDA_cudart_static_LIBRARY cudart_static 
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    )
  xpListRemoveIfExists(CUDA_LIBRARIES ${CUDA_CUDART_LIBRARY})
  xpListAppendIfDne(CUDA_LIBRARIES ${CUDA_cudart_static_LIBRARY})
endif()
sdSetOptionsProperties()
sdSetDirs() # set include and link directories
sdSetFlags() # preprocessor, compiler, linker flags
set(xpSourceDir ${CMAKE_SOURCE_DIR})
set(xpRelBranch master)
include(${externpro_DIR}/share/cmake/revision.cmake) # ${CMAKE_BINARY_DIR}/Revision.hpp
set(clas_repo ${CMAKE_BINARY_DIR}/clasrepo)
sdClassifiedRepo(isr.git 4869685a7bca ${clas_repo})
