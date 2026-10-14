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
        numerix::MultiVector<double> block(2, 2);
        Kokkos::deep_copy(exec, block.View(), 2.0);
        auto copy = numerix::Clone(exec, block);
        numerix::Vector<double> dots(2);
        numerix::ColumnDots(exec, block.View(), copy.View(), dots.View());
        exec.fence();
        auto host_dots = Kokkos::create_mirror_view_and_copy(numerix::DefaultHostMemorySpace {}, dots.View());
        if (host_dots(0) != 8.0 || host_dots(1) != 8.0) {
            result = 1;
        }

        using Matrix = numerix::SparseOperator<double>::matrix_type;
        Matrix::row_map_type::non_const_type rows("rows", 3);
        Matrix::index_type columns("columns", 2);
        Matrix::values_type entries("entries", 2);
        auto host_rows = Kokkos::create_mirror_view(rows);
        auto host_columns = Kokkos::create_mirror_view(columns);
        host_rows(0) = 0;
        host_rows(1) = 1;
        host_rows(2) = 2;
        host_columns(0) = 0;
        host_columns(1) = 1;
        Kokkos::deep_copy(exec, rows, host_rows);
        Kokkos::deep_copy(exec, columns, host_columns);
        Kokkos::deep_copy(exec, entries, 3.0);
        const numerix::SparseOperator<double> op(Matrix("diagonal", 2, 2, 2, entries, rows, columns));
        op.Apply(exec, block.View(), copy.View());
        numerix::ColumnDots(exec, copy.View(), copy.View(), dots.View());
        exec.fence();
        Kokkos::deep_copy(host_dots, dots.View());
        if (host_dots(0) != 72.0 || host_dots(1) != 72.0) {
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
