cmake_minimum_required(VERSION 3.1)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

project(argus-drm VERSION 1.0 LANGUAGES C CXX)

set(CUDA_PATH /usr/local/cuda-10.2)

set(TEGRA_ARMABI aarch64-linux-gnu)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2)

set(CMAKE_MODULE_PATH "/usr/src/jetson_multimedia_api/argus/cmake/" "${CMAKE_MODULE_PATH}")

find_package(CUDA REQUIRED)
find_package(CUDNN REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(TensorRT REQUIRED)
find_package(vpi 1.2 REQUIRED)
# find_package(Argus REQUIRED)
find_package(X11 REQUIRED)

find_package(PkgConfig REQUIRED)

pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
pkg_check_modules(Cairo REQUIRED cairo)

# find_package(GTK3 REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)

include_directories(
    /usr/include/gstreamer-1.0 
    /usr/include/glib-2.0 
    /usr/include/gtk-3.0/
    /usr/include/cairo/
    /usr/include/pango-1.0/
    /usr/include/gtk-unix-print-2.0/
    /usr/lib/${TEGRA_ARMABI}/glib-2.0/include
    /usr/include/libdrm
    /usr/include/libpng16
    /usr/include/${TEGRA_ARMABI}
    utils/
    utils/include/
    utils/include/EGLStream
    utils/cuda
    utils/Argus/
)

file(GLOB CLASS_SRC 
    "utils/cuda/*.cu"
    "utils/*.cpp"
    "utils/classes/*.cpp"
    "utils/Argus/*.cpp"
    "argus_capture.cpp"
    "utils/gtx/*.cpp"
    "utils/gtk/*.cpp"
    "utils/Argus/nvmmapi/*.cpp"
)

link_directories(	
            ${CUDA_PATH}/lib64
            /usr/lib/${TEGRA_ARMABI}
            /usr/lib/${TEGRA_ARMABI}/tegra
            /usr/local/lib
             ${GTK3_LIBRARY_DIRS}
        )


add_definitions(
    ${GTK3_CFLAGS_OTHER}
    )

set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_62,code=sm_62 -gencode arch=compute_53,code=sm_53)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -Wunused-function -fPIC")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# TARGETS --------------------------------------------------------------------------------------------------------------

cuda_add_executable(${PROJECT_NAME} main.cpp ${CLASS_SRC} )

set(CMAKE_CXX_STANDARD 17)
add_compile_options(${CMAKE_MFC_FLAG} )
target_link_libraries(${PROJECT_NAME}  vpi	-lpthread -lv4l2 -lEGL -lGLESv2  -lnvbuf_utils 
    -lnveglstream_camconsumer -lnvargus_socketclient
    -lnvjpeg -lnvosd -ldrm -lcuda -lcudart ${CUDA_nppicc_LIBRARY} 
    -lgstreamer-1.0 -lgobject-2.0 -lglib-2.0 -lgio-2.0 -lgstrtsp-1.0 -lgstsdp-1.0 
    -lgstrtp-1.0 -lgstbase-1.0  -lgstnet-1.0 -lgstapp-1.0 
)

target_link_libraries(${PROJECT_NAME}  ${CUDNN_LIBRARIES}  ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${GTK3_LIBRARIES} ${X11_LIBRARIES}) 

target_include_directories(${PROJECT_NAME} PUBLIC ${GTK3_INCLUDE_DIRS} ${CUDNN_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} 
    ${TensorRT_INCLUDE_DIRS}  )

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT}  -lculibos -lpthread -lrt -ldl -lm ) 
install(TARGETS argus-drm DESTINATION bin)

target_compile_options(${PROJECT_NAME} PRIVATE -fopenmp -fPIC ) # -Wall  -ldl

