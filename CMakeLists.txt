# CMake configuration for AMD Dual-GPU OpenCL Development Environment
cmake_minimum_required(VERSION 3.20)
project(KevinFMRIAnalysis)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL - Handle Windows OpenCL installation where only DLL is available
if(WIN32)
    # Set the OpenCL variables manually since the import library might not be available
    set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(OpenCL_FOUND TRUE)
    
    # Create the OpenCL::OpenCL target manually
    add_library(OpenCL::OpenCL INTERFACE IMPORTED)
    set_target_properties(OpenCL::OpenCL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OpenCL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "OpenCL.dll"
    )
    
    # Add the System32 directory to the library search path
    link_directories("C:/Windows/System32")
    
    message(STATUS "OpenCL configured for Windows with headers in: ${OpenCL_INCLUDE_DIR}")
    message(STATUS "OpenCL library will be linked from: C:/Windows/System32")
else()
    # Use standard find_package for non-Windows systems
    find_package(OpenCL REQUIRED)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Phase 1: Basic OpenCL Examples
add_executable(opencl_device_info src/opencl_device_info.cpp)
target_link_libraries(opencl_device_info OpenCL::OpenCL)
target_include_directories(opencl_device_info PRIVATE ${OpenCL_INCLUDE_DIR})

add_executable(vector_add src/vector_add.cpp)
target_link_libraries(vector_add OpenCL::OpenCL)
target_include_directories(vector_add PRIVATE ${OpenCL_INCLUDE_DIR})

# Phase 3: Brain GPU Accelerator (Original - Single GPU)
add_executable(brain_gpu_accelerator src/brain_gpu_accelerator.cpp)
target_link_libraries(brain_gpu_accelerator OpenCL::OpenCL)
target_include_directories(brain_gpu_accelerator PRIVATE ${OpenCL_INCLUDE_DIR})

# Phase 3: Shared library for Python integration
add_library(brain_gpu_lib SHARED src/brain_gpu_accelerator.cpp)
target_link_libraries(brain_gpu_lib OpenCL::OpenCL)
target_include_directories(brain_gpu_lib PRIVATE ${OpenCL_INCLUDE_DIR})

# Phase 3+: TRUE Dual-GPU Brain Accelerator
add_executable(dual_gpu_brain_accelerator src/dual_gpu_brain_accelerator.cpp)
target_link_libraries(dual_gpu_brain_accelerator OpenCL::OpenCL)
target_include_directories(dual_gpu_brain_accelerator PRIVATE ${OpenCL_INCLUDE_DIR})

# Phase 3+: Dual-GPU shared library for Python integration
add_library(dual_gpu_brain_lib SHARED src/dual_gpu_brain_accelerator.cpp)
target_link_libraries(dual_gpu_brain_lib OpenCL::OpenCL)
target_include_directories(dual_gpu_brain_lib PRIVATE ${OpenCL_INCLUDE_DIR})

# Phase 3+: Simple Dual-GPU Demonstration
add_executable(simple_dual_gpu_demo src/simple_dual_gpu_demo.cpp)
target_link_libraries(simple_dual_gpu_demo OpenCL::OpenCL)
target_include_directories(simple_dual_gpu_demo PRIVATE ${OpenCL_INCLUDE_DIR})

# Set properties for shared libraries
set_target_properties(brain_gpu_lib PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(dual_gpu_brain_lib PROPERTIES
    PREFIX ""
    SUFFIX ".dll" 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add compiler definitions for shared libraries
target_compile_definitions(brain_gpu_lib PRIVATE BUILDING_DLL)
target_compile_definitions(dual_gpu_brain_lib PRIVATE BUILDING_DLL) 