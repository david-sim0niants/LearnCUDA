set(STB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)

file(DOWNLOAD https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
    ${STB_INCLUDE_DIR}/stb_image.h)

file(DOWNLOAD https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
    ${STB_INCLUDE_DIR}/stb_image_write.h)

enable_language(C)
set(CMAKE_C_STANDARD 11)
set(TARGET_NAME stb)
add_library(${TARGET_NAME} STATIC impl.c)
target_include_directories(${TARGET_NAME} PUBLIC ${STB_INCLUDE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
