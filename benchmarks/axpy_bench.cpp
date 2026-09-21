#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <type_traits>

#include <Kokkos_Core.hpp>
#include <numerix/numerix.hpp>

namespace {

    constexpr std::size_t kSize = std::size_t(1) << 22;
    constexpr std::size_t kRepeats = 20;
    constexpr std::size_t kRounds = 5;
    constexpr double kBytesPerElement = 8.0;

    using index_type = typename numerix::Vector<double>::view_type::index_type;

    // 取多轮中的最优一轮：共享/虚拟化机器上单次测量波动可达 30%，最优值比平均值更接近真实带宽。
    template <class Fn>
    double BestSeconds(Fn&& fn)
    {
        double best = std::numeric_limits<double>::max();
        for (std::size_t round = 0; round < kRounds; ++round) {
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < kRepeats; ++i) {
                fn();
            }
            Kokkos::fence();
            const auto stop = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double>(stop - start).count());
        }
        return best;
    }

    // 每个元素搬运 3 个数组（读 x、读改写 y）。
    double BandwidthGbs(double seconds)
    {
        const double bytes = kBytesPerElement * 3.0 * static_cast<double>(kSize) * static_cast<double>(kRepeats);
        return bytes / seconds / 1e9;
    }

} // namespace

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    {
        // 执行空间实例在 initialize() 之后构造。
        const numerix::DefaultExecutionSpace exec {};

        numerix::Vector<double> x(kSize);
        numerix::Vector<double> y(kSize);
        Kokkos::deep_copy(x.View(), 1.0);
        Kokkos::deep_copy(y.View(), 2.0);

        const double numerix_seconds = BestSeconds([&] { numerix::Axpy(exec, 0.5, x.View(), y.View()); });

        const auto x_view = x.View();
        const auto y_view = y.View();
        // 对照基线：裸 Kokkos kernel，索引类型跟随 View，不做 int 窄化。
        const double kokkos_seconds = BestSeconds([&] {
            Kokkos::parallel_for(
                "raw::Axpy", Kokkos::RangePolicy<numerix::DefaultExecutionSpace, Kokkos::IndexType<index_type>>(exec, 0, y_view.extent(0)),
                KOKKOS_LAMBDA(index_type i) { y_view(i) += 0.5 * x_view(i); });
        });

        // 手写循环只在与默认内存空间同为主机内存时可用：它是零开销抽象的对照基线。
        constexpr bool kHasHostBaseline = std::is_same_v<numerix::DefaultMemorySpace, numerix::DefaultHostMemorySpace>;
        double host_seconds = 0.0;
        if constexpr (kHasHostBaseline) {
            double* const y_raw = y.Data();
            const double* const x_raw = x.Data();
            host_seconds = BestSeconds([&] {
                for (std::size_t i = 0; i < kSize; ++i) {
                    y_raw[i] += 0.5 * x_raw[i];
                }
            });
        }

        std::cout << "execution space = " << numerix::DefaultExecutionSpace::name() << ", size = " << kSize
                  << ", repeats = " << kRepeats << ", rounds = " << kRounds << std::endl;
        std::cout << "numerix axpy : " << BandwidthGbs(numerix_seconds) << " GB/s" << std::endl;
        std::cout << "kokkos  axpy : " << BandwidthGbs(kokkos_seconds) << " GB/s" << std::endl;
        if constexpr (kHasHostBaseline) {
            std::cout << "host    axpy : " << BandwidthGbs(host_seconds) << " GB/s" << std::endl;
        }
        std::cout << "checksum = " << numerix::Dot(exec, y.View(), y.View()) << std::endl;
    }
    Kokkos::finalize();
    return 0;
}
