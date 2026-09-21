add_executable(numerix_tests
    test_main.cpp
    test_core.cpp
    test_vector.cpp
    test_operator.cpp
)

target_include_directories(numerix_tests PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(numerix_tests
    PRIVATE
        numerix
        numerix_warnings
        GTest::gtest
)

include(GoogleTest)
gtest_discover_tests(numerix_tests)
