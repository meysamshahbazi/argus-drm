cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(argus-drm VERSION 1.0 LANGUAGES C CXX)

set(CUDA_PATH /usr/local/cuda-10.2)

set(TEGRA_ARMABI aarch64-linux-gnu)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2)


find_package(CUDA REQUIRED)
find_package(CUDNN REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(TensorRT REQUIRED)
find_package(vpi 1.2 REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)

include_directories(
    /usr/include/gstreamer-1.0 
    /usr/include/glib-2.0 
    /usr/lib/aarch64-linux-gnu/glib-2.0/include 
    )

# file(GLOB CLASS_SRC 
       
#            )

link_directories(	
            ${CUDA_PATH}/lib64
            /usr/lib/${TEGRA_ARMABI}
            /usr/lib/${TEGRA_ARMABI}/tegra
            /usr/local/lib
        )

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_53,code=sm_53)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -Wunused-function -fPIC")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# TARGETS --------------------------------------------------------------------------------------------------------------

cuda_add_executable(${PROJECT_NAME} main.cpp ${CLASS_SRC} )

set(CMAKE_CXX_STANDARD 17)
add_compile_options(${CMAKE_MFC_FLAG} )
target_link_libraries(${PROJECT_NAME}  vpi	-lpthread -lv4l2 -lEGL -lGLESv2  -lnvbuf_utils 
    -lnvjpeg -lnvosd -ldrm -lcuda -lcudart ${CUDA_nppicc_LIBRARY} 
    -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 -lgio-2.0 -lgstrtsp-1.0 -lgstsdp-1.0 
    -lgstrtp-1.0 -lgstbase-1.0  -lgstnet-1.0 -lgstapp-1.0 
)

target_link_libraries(${PROJECT_NAME}  ${CUDNN_LIBRARIES}  ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ) 

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} 
    ${TensorRT_INCLUDE_DIRS}  )

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT}  -lculibos -lpthread -lrt -ldl -lm 
) 
install(TARGETS argus-drm DESTINATION bin)

target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp -fPIC ) # -Wall  -ldl

