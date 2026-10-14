add_executable(numerix_axpy axpy.cpp)

target_link_libraries(numerix_axpy
    PRIVATE
        numerix
        numerix_options
)
