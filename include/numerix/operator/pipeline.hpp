#pragma once

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <numerix/operator/linear_operator.hpp>

namespace numerix {

    // 扁平流水线：N 个算子顺序作用，中间结果放在构造时一次性分配的缓冲中，热路径不再分配。
    // 相比嵌套组合 compose(f, compose(g, h))，模板实例数只随算子个数线性增长。
    template <class X, LinearOperator<X>... Ops>
    class Pipeline {
    public:
        static constexpr std::size_t kNumStages = sizeof...(Ops);
        static_assert(kNumStages > 0, "Pipeline requires at least one stage");

        // X 必须能由元素个数构造；size 同时决定所有中间缓冲的大小。
        Pipeline(std::size_t size, Ops... ops) : stages_(std::move(ops)...)
        {
            temporaries_.reserve(kNumStages - 1);
            for (std::size_t i = 0; i + 1 < kNumStages; ++i) {
                temporaries_.emplace_back(size);
            }
        }

        void Apply(const X& x, X& y) const
        {
            const X* input = &x;
            std::size_t next_stage = 0;
            auto apply_stage = [&](const auto& stage) {
                const bool is_last = (next_stage + 1 == kNumStages);
                X* output = is_last ? &y : &temporaries_[next_stage];
                stage.Apply(*input, *output);
                input = output;
                ++next_stage;
            };
            std::apply([&](const auto&... stages) { (apply_stage(stages), ...); }, stages_);
        }

    private:
        std::tuple<Ops...> stages_;
        // 中间缓冲只是暂存区，不参与逻辑状态，因此允许在 const Apply 中复用。
        mutable std::vector<X> temporaries_;
    };

    // 显式指定向量空间的组合入口：auto op = Compose<Vector<double>>(n, B, C, D);
    template <class X, LinearOperator<X>... Ops>
    Pipeline<X, Ops...> Compose(std::size_t size, Ops... ops)
    {
        return Pipeline<X, Ops...>(size, std::move(ops)...);
    }

} // namespace numerix
