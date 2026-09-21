#pragma once

#include <iosfwd>
#include <string_view>

namespace numerix {

    enum class LogLevel { kDebug = 0,
        kInfo,
        kWarn,
        kError };

    constexpr std::string_view ToString(LogLevel level)
    {
        switch (level) {
        case LogLevel::kDebug:
            return "debug";
        case LogLevel::kInfo:
            return "info";
        case LogLevel::kWarn:
            return "warn";
        case LogLevel::kError:
            return "error";
        }
        return "unknown";
    }

    // 日志器是显式对象：numerix 不引入全局可变状态（见 docs/DESIGN.md D-5）。
    // 默认级别 INFO；除 DEBUG 外不应在热循环内打日志。
    class Logger {
    public:
        explicit Logger(LogLevel level = LogLevel::kInfo, std::ostream& sink = DefaultSink());

        void SetLevel(LogLevel level) { level_ = level; }
        LogLevel Level() const { return level_; }

        void Log(LogLevel level, std::string_view message);

        void Debug(std::string_view message) { Log(LogLevel::kDebug, message); }
        void Info(std::string_view message) { Log(LogLevel::kInfo, message); }
        void Warn(std::string_view message) { Log(LogLevel::kWarn, message); }
        void Error(std::string_view message) { Log(LogLevel::kError, message); }

    private:
        static std::ostream& DefaultSink();

        LogLevel level_;
        std::ostream& sink_;
    };

} // namespace numerix

// 宏封装：调用点只写 NUMERIX_LOG_INFO(logger, "msg")，无需重复拼 LogLevel。
// DEBUG 日志在 Release 构建中整体消除；未启用时仍以 sizeof（不求值）标记参数已使用，
// 避免 /WX、-Werror 下出现“未使用变量”告警。
#ifdef NUMERIX_DEBUG
#define NUMERIX_LOG_DEBUG(logger, message) (logger).Log(::numerix::LogLevel::kDebug, (message))
#else
#define NUMERIX_LOG_DEBUG(logger, message) \
    do {                                   \
        (void)sizeof(logger);              \
        (void)sizeof(message);             \
    } while (false)
#endif

#define NUMERIX_LOG_INFO(logger, message) (logger).Log(::numerix::LogLevel::kInfo, (message))
#define NUMERIX_LOG_WARN(logger, message) (logger).Log(::numerix::LogLevel::kWarn, (message))
#define NUMERIX_LOG_ERROR(logger, message) (logger).Log(::numerix::LogLevel::kError, (message))
