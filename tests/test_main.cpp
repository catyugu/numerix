#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

// 测试二进制需要 Kokkos 运行时：先剥离 gtest 参数，再交给 Kokkos 解析。
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);

    const int result = RUN_ALL_TESTS();

    Kokkos::finalize();
    return result;
}
