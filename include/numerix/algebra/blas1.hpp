#pragma once

#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_scal.hpp>

#include <numerix/core/concepts.hpp>
#include <numerix/execution/execution_space.hpp>

namespace numerix {

    // BLAS-1 语义：numerix 固定接口与语义，kernel 交给 Kokkos Kernels，不重写成熟实现。
    // 每个操作都显式接收 execution-space instance —— 同一个内存空间配不同的 instance
    // 就是不同的 CUDA/HIP/SYCL stream/queue，这是计算/通信重叠与异步流水线的前提。
    // 不带 instance 的重载使用默认 instance，只作便利。
    //
    // 操作对象是 Kokkos::View：subview、跨步视图、MultiVector 的列都能直接参与；
    // numerix::Vector 通过 View() 交出自己拥有的 View。

    template <class Exec, class X>
        requires(ExecutionSpace<Exec> && MutableDenseVector<X, Exec>)
    void Scale(const Exec& exec, ScalarOf<X> alpha, const X& x)
    {
        KokkosBlas::scal(exec, x, alpha, x);
    }

    template <class X>
        requires MutableDenseVector<X>
    void Scale(ScalarOf<X> alpha, const X& x)
    {
        Scale(DefaultExecutionSpace {}, alpha, x);
    }

    // 输出参数同样按 Kokkos 约定用 const 引用传 View：View 是句柄，const 句柄仍然指向可写数据，
    // 可写性由 MutableDenseVector 约束保证。这样 op.Apply(exec, x, y.View()) 这类调用点不必先绑定局部变量。
    template <class Exec, class X, class Y>
        requires(ExecutionSpace<Exec> && DenseVector<X, Exec> && MutableDenseVector<Y, Exec>)
    void Axpy(const Exec& exec, ScalarOf<Y> alpha, const X& x, const Y& y)
    {
        KokkosBlas::axpy(exec, alpha, x, y);
    }

    template <class X, class Y>
        requires(DenseVector<X> && MutableDenseVector<Y>)
    void Axpy(ScalarOf<Y> alpha, const X& x, const Y& y)
    {
        Axpy(DefaultExecutionSpace {}, alpha, x, y);
    }

    // 内积按 Kokkos Kernels 的 BLAS 语义：Dot(x, y) = sum(conj(x(i)) * y(i))。
    template <class Exec, class X, class Y>
        requires(ExecutionSpace<Exec> && DenseVector<X, Exec> && DenseVector<Y, Exec> && std::same_as<ScalarOf<X>, ScalarOf<Y>>)
    auto Dot(const Exec& exec, const X& x, const Y& y)
    {
        return KokkosBlas::dot(exec, x, y);
    }

    template <class X, class Y>
        requires(DenseVector<X> && DenseVector<Y> && std::same_as<ScalarOf<X>, ScalarOf<Y>>)
    auto Dot(const X& x, const Y& y)
    {
        return Dot(DefaultExecutionSpace {}, x, y);
    }

    // 2-范数：返回实标量，复数向量也不例外。
    template <class Exec, class X>
        requires(ExecutionSpace<Exec> && DenseVector<X, Exec>)
    auto Norm2(const Exec& exec, const X& x)
    {
        return KokkosBlas::nrm2(exec, x);
    }

    template <class X>
        requires DenseVector<X>
    auto Norm2(const X& x)
    {
        return Norm2(DefaultExecutionSpace {}, x);
    }

} // namespace numerix
