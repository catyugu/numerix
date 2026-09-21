#include <cstddef>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <numerix/numerix.hpp>

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        numerix::Logger logger;
        NUMERIX_LOG_INFO(logger, "numerix example: local algebra + device vector");

        // 单元局部计算：栈上定长代数，整段在编译期求值。
        constexpr auto local_operator = numerix::StaticMatrix<double, 2, 2>::Identity();
        constexpr numerix::StaticVector<double, 2> local_x {{1.0, 2.0}};
        constexpr auto local_y = local_operator.Apply(local_x);
        static_assert(local_y[1] == 2.0);

        // 全局计算：Kokkos::View 支撑的动态向量。
        constexpr std::size_t kSize = 16;
        numerix::Vector<double> x(kSize);
        Kokkos::deep_copy(x.View(), 1.5);

        numerix::Vector<double> y(kSize);
        y.CopyFrom(x);
        y.Scale(2.0);

        std::cout << "device = " << numerix::DeviceName() << std::endl;
        std::cout << "y = 2x, dot(y, y) = " << y.Dot(y) << std::endl;
        std::cout << "expected = " << 4.0 * 1.5 * 1.5 * static_cast<double>(kSize) << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
