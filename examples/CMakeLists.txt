add_executable(numerix_axpy axpy.cpp)

target_include_directories(numerix_axpy PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(numerix_axpy
    PRIVATE
        numerix
        numerix_warnings
)
