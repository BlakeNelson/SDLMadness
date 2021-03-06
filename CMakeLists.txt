cmake_minimum_required(VERSION 3.0)
set(lib_name SDLMadness)
set(exe_name SDLMadness2018)
project(${exe_name})

# Extern Pro
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}) # Findexternpro.cmake
set(externpro_REV 16.10.3)
find_package(externpro REQUIRED)
find_package(CUDA)
xpGetExtern(externIncs externLibs PRIVATE boost)


include_directories(/usr/local/smc/lib ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Library
set(srcs
  BracketOdds.hpp
  CudaApproach.cuh
  CudaApproach.cu
  Util.hpp
  )

# Executable
set(exe_srcs
 main.cpp
 )
xpStringAppendIfDne(CUDA_NVCC_FLAGS "--generate-code arch=compute_30,code=sm_30")
xpStringAppendIfDne(CUDA_NVCC_FLAGS "--generate-code arch=compute_30,code=compute_30")
#xpStringAppendIfDne(CUDA_NVCC_FLAGS_DEBUG "-G")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CUDA_TOOLKIT_INCLUDE} ${externIncs} "/opt/externpro/externpro-16.01.1-gcc493-64-Linux/include/boost-1_57")
message(${externIncs})
source_group("Source" FILES ${srcs})

cuda_add_library(${lib_name} ${srcs})
cuda_add_executable(${exe_name} ${exe_srcs})
add_executable(MonteCarlo MonteCarlo.cpp)
target_link_libraries(${exe_name} ${CUDA_LIBRARIES} ${lib_name})
target_include_directories(${exe_name} ${externIncs})
target_link_libraries(MonteCarlo ${lib_name})
target_include_directories(MonteCarlo ${externIncs})
add_subdirectory(Test)
xpPostBuildCopy(${lib_name} ${CMAKE_CURRENT_SOURCE_DIR}/Test/fivethirtyeight_ncaa_forecasts_2018.csv .)
