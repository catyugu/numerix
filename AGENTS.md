# AGENTS

## 概述

numerix 是一个 C++20 + Kokkos-native 的通用科学计算基础库。它不提供 FEM/FVM/BEM 框架，而提供所有离散方法
共享的数值执行模型：执行与内存空间、定长与动态代数、线性算子、求解器、诊断。Kokkos 是 numerix 的机器模型，
numerix 只在其上提供科学计算语义，不与 Eigen/PETSc 等特定数值计算设施竞争。

设计与计划见 `docs/`。文档与代码不一致时以 `docs/DESIGN.md` 为准，并同步修正代码。

## 宪法

### 语言与标准

- 仅 C++20：不使用 C++23、modules、`std::mdspan`、`std::expected`。
- 代码中除注释外一律英文：标识符、字符串、日志文本、断言消息不得出现中文。

### 依赖与边界

- Kokkos Core 与 Kokkos Kernels 是仅有的两个必需依赖，且版本同步；其余后端只经 `backend/` adapter 接入，
  且不得出现在核心头文件中。
- 不重新封装 Kokkos：不定义 `numerix::View`，不实现 thread pool、CUDA wrapper 或 allocator 抽象。
- 不重写成熟 kernel：BLAS/稀疏容器用 Kokkos Kernels，numerix 只固定接口与语义。
- 模块依赖单向；模块之间依赖接口与共同数据结构，不依赖彼此的具体实现。
- 组合优于继承，禁止多继承；不保留向后兼容路径。

### 性能与抽象

- 零开销：热路径中不出现虚调用、`std::function`、堆分配与运行时分派。
- 执行流显式：底层算法接口接收 execution-space instance；求解器可通过调用方显式构造的代数操作对象
  绑定该 instance，不隐式使用某个默认流。运行期多态只允许用于
  求解器选择、后端适配与用户扩展，且不得进入 kernel、装配循环与向量运算。
- 设备可移植：标量只用 `Kokkos::complex` 与浮点，进入 kernel 的函数必须 `KOKKOS_INLINE_FUNCTION`，
  定长代数与标量运算能用 `constexpr` 的一律 `constexpr`。
- 索引不做窄化：区分 `size_type` 与 local/global ordinal，RangePolicy 的索引类型跟随 View 的 `index_type`。
- 模板只用于静态多态、kernel 融合与零开销抽象；公共模板 API 必须由 concept 约束。
- 不为不存在的需求做抽象或参数化；未被使用或只用一处的抽象应删除。

### 数据契约

- `Vector` 是 owning、move-only 存储，不含数学成员函数；需要复制时显式 `Clone` / `DeepCopy`，
  需要别名或 subview 时直接用 `Kokkos::View`。
- `const` 容器只交出只读 View；只读与可写由类型区分，不靠约定。
- 现有 Vector / MultiVector、View BLAS 和 SparseOperator 具有本地语义，不隐含全局归约或 ghost 处理。
- 阶段 2 首先随 CG 实现静态绑定的代数操作接口：求解器通过它创建工作向量、执行向量更新、内积、范数
  和兼容性检查，不写死本地 View / Vector，不自行调用 MPI。分布式数学与完成契约见 DESIGN.md §5.9，
  MPI 实现留到阶段 4；现在不添加无人使用的代数对象或分布式空壳。

### 错误与日志

- 底层 API 不把异常当作正常控制流，也不使用 RTTI；契约违背用 `KOKKOS_ASSERT` 表达。
  第三方依赖抛出的异常（如 Kokkos Kernels 对非法 extent）按其自身语义处理，不为"异常纯净"与其对抗。
- 日志使用 `Logger` 对象与 `NUMERIX_LOG_*` 宏；默认级别 INFO，除 DEBUG 外不打在热循环内。

### 风格与测试

- 新功能与缺陷修复遵循 TDD：先写测试并运行，确认因缺失行为而失败；再做最小实现使测试通过；
  最后在测试保持通过的前提下重构。不得先实现后补测试。

- 命名遵循 Google Style C++；格式化以仓库根目录 `.clang-format` 为准。
- 注释只在必要时出现（隐式契约、单位、数据布局、非常见技巧），代码应通过命名与接口自明。
- 测试必须验证数学性质（线性性、对称性、正定性、内积共轭、融合算子语义、所有权与 const 语义），
  而不是只覆盖代码路径。
- 任何改动都必须在 Debug 与 Release 下零警告构建、`ctest` 全绿，并给出真实运行结果。
