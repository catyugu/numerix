#pragma once

#include <cassert>

// 内部实现只做最小断言（信任内部数据）；接口层应返回 Status 而不是依赖断言。
#define NUMERIX_ASSERT(condition) assert(condition)
