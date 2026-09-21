#include <cstddef>
#include <iostream>

#include <Kokkos_Core.hpp>
#include <numerix/numerix.hpp>

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        numerix::Logger logger;
        NUMERIX_LOG_INFO(logger, "numerix example: static algebra + Kokkos-backed dynamic vector");

        // 单元局部计算：栈上定长代数，整段在编译期求值，也能安全进入设备 kernel。
        constexpr auto local_operator = numerix::StaticMatrix<double, 2, 2>::Identity();
        constexpr numerix::StaticVector<double, 2> local_x {{1.0, 2.0}};
        constexpr auto local_y = local_operator.Apply(local_x);
        static_assert(local_y[1] == 2.0);

        // 全局计算：Vector 拥有内存，代数运算通过 View 在显式 execution-space instance 上执行。
        constexpr std::size_t kSize = 16;
        const numerix::DefaultExecutionSpace exec {};

        numerix::Vector<double> x(kSize);
        Kokkos::deep_copy(x.View(), 1.5);

        numerix::Vector<double> y = numerix::Clone(exec, x);
        numerix::Scale(exec, 2.0, y.View());

        std::cout << "execution space = " << numerix::DefaultExecutionSpace::name() << std::endl;
        std::cout << "y = 2x, dot(y, y) = " << numerix::Dot(exec, y.View(), y.View()) << std::endl;
        std::cout << "expected = " << 4.0 * 1.5 * 1.5 * static_cast<double>(kSize) << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
