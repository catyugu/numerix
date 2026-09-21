#pragma once

#include <concepts>
#include <memory>
#include <utility>

namespace numerix {

    // 编译期算子契约：Apply 只做 y = A x，热路径上不出现虚调用与堆分配。
    template <class Op, class X, class Y = X>
    concept LinearOperator = requires(const Op& op, const X& x, Y& y) {
        { op.Apply(x, y) } -> std::same_as<void>;
    };

    // 预条件子与线性算子契约一致，单独命名以表达语义。
    template <class P, class X, class Y = X>
    concept Preconditioner = LinearOperator<P, X, Y>;

    // 类型擦除包装：只用于求解器选择、后端适配等非热路径边界（见 docs/DESIGN.md D-1）。
    template <class X, class Y = X>
    class AnyLinearOperator {
    public:
        template <LinearOperator<X, Y> Op>
        explicit AnyLinearOperator(Op op) : self_(std::make_unique<Model<Op>>(std::move(op))) { }

        void Apply(const X& x, Y& y) const { self_->Apply(x, y); }

    private:
        struct Concept {
            virtual ~Concept() = default;
            virtual void Apply(const X& x, Y& y) const = 0;
        };

        template <class Op>
        struct Model final : Concept {
            explicit Model(Op op) : op_(std::move(op)) { }

            void Apply(const X& x, Y& y) const override { op_.Apply(x, y); }

            Op op_;
        };

        std::unique_ptr<const Concept> self_;
    };

} // namespace numerix
