cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0146 NEW)

project(custom_flash LANGUAGES CXX CUDA)

# Optional: Use conda env for Python includes
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Set CUDA arch (optional default)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Set libtorch download info
set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch\-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip")
set(LIBTORCH_ZIP "${CMAKE_SOURCE_DIR}/third-party/libtorch.zip")
set(LIBTORCH_DIR "${CMAKE_SOURCE_DIR}/third-party/libtorch")

# Download & unzip libtorch if needed
if(NOT EXISTS ${LIBTORCH_ZIP})
    file(DOWNLOAD ${LIBTORCH_URL} ${LIBTORCH_ZIP} SHOW_PROGRESS)
endif()

if(NOT EXISTS ${LIBTORCH_DIR})
    file(MAKE_DIRECTORY ${LIBTORCH_DIR})
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvf ${LIBTORCH_ZIP} --strip-components=1 -C ${LIBTORCH_DIR})
endif()

# Add torch path
list(APPEND CMAKE_PREFIX_PATH ${LIBTORCH_DIR})
find_package(Torch REQUIRED)

# Source files
file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")
file(GLOB header_files "${CMAKE_SOURCE_DIR}/src/*.cuh")

# Make shared library (Python extension module)
add_library(custom_flash SHARED ${cpu_source_files} ${gpu_source_files} ${header_files})

# Include paths
target_include_directories(custom_flash PRIVATE 
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Link Torch and Python
target_link_libraries(custom_flash PRIVATE 
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
)

# Don't prepend 'lib' prefix for Python import
set_target_properties(custom_flash PROPERTIES PREFIX "")

# Needed for .so import from Python
set_target_properties(custom_flash PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Optional: Use C++14
set(CMAKE_CXX_STANDARD 14)
