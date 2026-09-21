#pragma once

#include <concepts>

#include <numerix/core/concepts.hpp>

namespace numerix {

    // 编译期算子契约：Apply(exec, x, y) 只做 y = A x，在给定的 execution-space instance
    // 上执行（因而也就指定了 stream/queue），热路径上不出现虚调用与堆分配。
    // 执行流不由 y 隐含决定：调用点必须给出 instance。
    template <class Op, class Exec, class X, class Y = X>
    concept LinearOperatorFor = ExecutionSpace<Exec> && requires(const Op& op, const Exec& exec, const X& x, Y& y) {
        { op.Apply(exec, x, y) } -> std::same_as<void>;
    };

    // 融合形式 y = alpha * A x + beta * y：省掉一次额外的内存往返，
    // KokkosSparse::spmv 原生支持这种形式。不要求所有算子都实现它：
    // 求解器可以按自己的 workspace 做 Apply + Axpby 回退（见 docs/DESIGN.md D-3）。
    template <class Op, class Exec, class X, class Y = X>
    concept ScaledLinearOperatorFor = LinearOperatorFor<Op, Exec, X, Y> && requires(const Op& op, const Exec& exec, const X& x, Y& y, ScalarOf<Y> alpha, ScalarOf<Y> beta) {
        { op.Apply(exec, alpha, x, beta, y) } -> std::same_as<void>;
    };

} // namespace numerix
