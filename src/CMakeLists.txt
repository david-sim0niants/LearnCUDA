set(TARGET_NAME learncuda_src)

set(SRCS
    vector_add.cu
    matmul.cu
    conv2d.cu
    conv2d.cpp
    host_texture2d.cpp
    device_texture2d.cu
)

add_library(${TARGET_NAME} INTERFACE)
target_sources(${TARGET_NAME} INTERFACE ${SRCS})
target_include_directories(${TARGET_NAME} INTERFACE ${ROOT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
set_source_files_properties(conv2d.cpp PROPERTIES COMPILE_FLAGS -fopenmp)
