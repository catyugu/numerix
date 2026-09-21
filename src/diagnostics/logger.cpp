#include <numerix/diagnostics/logger.hpp>

#include <cstddef>
#include <iostream>
#include <ostream>

namespace numerix {

    std::ostream& Logger::DefaultSink() { return std::cerr; }

    Logger::Logger(LogLevel level, std::ostream& sink) : level_(level), sink_(sink) { }

    void Logger::Log(LogLevel level, std::string_view message)
    {
        if (level < level_) {
            return;
        }
        sink_ << "[" << ToString(level) << "] " << message << std::endl;
    }

} // namespace numerix
