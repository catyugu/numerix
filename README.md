# numerix

C++20 + Kokkos-native 的通用科学计算基础库。

numerix 不提供 FEM/FVM/BEM 框架，而提供所有离散方法共享的数值执行模型：执行空间、内存空间、
稠密/定长代数、线性算子、求解器与诊断。Kokkos 是 numerix 的机器模型，numerix 只在其上提供
科学计算语义。

## 要求

- CMake >= 3.24
- 支持 C++20 的编译器（GCC / Clang / MSVC）
- 推荐生成器：Ninja

Kokkos 是唯一必需第三方依赖，默认由 CPM 拉取（版本固定在 `cmake/Dependencies.cmake`）。

## 快速开始

```bash
cmake -S . -B build -G "Ninja"
cmake --build build
ctest --test-dir build --output-on-failure
```

示例与基准：

```bash
./build/examples/numerix_axpy
./build/benchmarks/numerix_axpy_bench
```

若本机已安装 Kokkos，可改用系统包：

```bash
cmake -S . -B build -G "Ninja" -DNUMERIX_USE_SYSTEM_KOKKOS=ON
```

## 安装与使用

```bash
cmake --install build-release --prefix <prefix>
```

使用方：

```cmake
find_package(numerix REQUIRED)
target_link_libraries(my_target PRIVATE numerix::numerix)
```

注意：

- 安装导出会把 numerix 与 CPM 拉取的 Kokkos 一起安装；若 numerix 是用
  `NUMERIX_USE_SYSTEM_KOKKOS=ON` 构建的，使用方需自行提供 Kokkos。
- MSVC/clang-cl 下 ABI 与 Debug/Release 相关，使用方的构建配置必须与 numerix 安装时一致。

## 开发命令

```bash
# 格式化（仓库根目录 .clang-format 为准）
uv tool run --from clang-format clang-format -i include/numerix/*.hpp include/numerix/*/*.hpp \
    src/*/*.cpp tests/*.cpp examples/*.cpp benchmarks/*.cpp
```

## 构建选项

| 选项 | 默认 | 含义 |
| --- | --- | --- |
| `NUMERIX_BUILD_EXAMPLES` | ON | 构建示例 |
| `NUMERIX_BUILD_TESTS` | ON | 构建测试（需要 googletest，由 CPM 拉取） |
| `NUMERIX_BUILD_BENCHMARKS` | ON | 构建基准程序 |
| `NUMERIX_USE_SYSTEM_KOKKOS` | OFF | 使用已安装的 Kokkos 而非 CPM 拉取 |
| `NUMERIX_WARNINGS_AS_ERRORS` | ON | numerix 自身目标把警告视为错误 |

## 目录结构

```bash
numerix/
├── cmake/            构建配置与集中依赖声明
├── docs/             设计文档与实施计划
├── include/numerix/  公共头文件
├── src/              库实现
├── tests/            测试
├── examples/         示例
└── benchmarks/       基准
```

## 文档

- `AGENTS.md`：项目概述与宪法（编码约定）
- `docs/DESIGN.md`：设计文档（目标、模块边界、约定、决策记录）
- `docs/PLAN.md`：分阶段实施计划与验收标准

## 许可证

MIT，见 `LICENSE`。
