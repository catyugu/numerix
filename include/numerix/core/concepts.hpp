#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

#include <numerix/core/scalar.hpp>

namespace numerix {

    // 连续一维数据的结构契约：热循环只依赖 Size/Data，不依赖具体容器类型。
    template <class V>
    concept VectorLike = requires(V& v) {
        typename std::remove_cvref_t<V>::value_type;
        { v.Size() } -> std::convertible_to<std::size_t>;
        { v.Data() } -> std::convertible_to<const typename std::remove_cvref_t<V>::value_type*>;
    };

} // namespace numerix
