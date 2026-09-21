#pragma once

#include <string_view>

namespace numerix {

    // 底层不使用异常：错误通过返回码显式传播，由调用方决定处理方式。
    enum class Status {
        kOk = 0,
        kInvalidArgument,
        kNotConverged,
        kBreakdown,
        kUnsupported,
    };

    constexpr std::string_view ToString(Status status)
    {
        switch (status) {
        case Status::kOk:
            return "ok";
        case Status::kInvalidArgument:
            return "invalid argument";
        case Status::kNotConverged:
            return "not converged";
        case Status::kBreakdown:
            return "breakdown";
        case Status::kUnsupported:
            return "unsupported";
        }
        return "unknown";
    }

    constexpr bool IsOk(Status status) { return status == Status::kOk; }

} // namespace numerix
