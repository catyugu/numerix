add_executable(numerix_axpy_bench axpy_bench.cpp)

target_link_libraries(numerix_axpy_bench
    PRIVATE
        numerix
        numerix_warnings
)
