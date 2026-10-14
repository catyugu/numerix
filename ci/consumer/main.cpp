#include <numerix/config.h>
#include <numerix/numerix.hpp>

#include <sstream>
#include <string>

int main(int argc, char** argv)
{
    Kokkos::initialize(argc, argv);
    int result = 0;
    {
        const numerix::DefaultExecutionSpace exec {};
        numerix::Vector<double> x(16);
        Kokkos::deep_copy(x.View(), 2.0);
        numerix::Scale(exec, 3.0, x.View());
        if (numerix::Dot(exec, x.View(), x.View()) != 576.0) {
            result = 1;
        }
        std::ostringstream sink;
        numerix::Logger logger(numerix::LogLevel::kInfo, sink);
        logger.Info("consumer");
        if (sink.str() != "[info] consumer\n" || numerix::config::kVersion != "0.1.0") {
            result = 1;
        }
    }
    Kokkos::finalize();
    return result;
}
