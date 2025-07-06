include(NullFile)
get_null_file(cpp)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(TARGET_NAME learncuda)
add_library(${TARGET_NAME} SHARED ${NULL_FILE})
target_link_libraries(${TARGET_NAME} PUBLIC learncuda_src)

set(TARGET_NAME gaussian_blur)
add_executable(${TARGET_NAME} gaussian_blur.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE learncuda)
