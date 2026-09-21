# numerix Design Document

> C++20 + Kokkos-native 的通用科学计算基础库。numerix 不提供 FEM/FVM/BEM 框架，
> 而提供所有离散方法共享的数值执行模型。

状态：骨架已落地（v0.1），本文件是唯一的设计记录；与代码不一致时以本文件为准并同步修正代码。

## 1. 设计目标

1. 仅 C++20。不使用 `std::mdspan`、`std::expected`、modules，因为科学计算环境升级慢。
2. Kokkos-native 可移植性：Serial / OpenMP / CUDA / HIP / SYCL 由 `Kokkos::ExecutionSpace`、
   `Kokkos::MemorySpace`、`Kokkos::View` 提供，numerix 不重新设计执行抽象。
3. 零开销抽象：`y = A.Apply(x)` 的最终形态是 Kokkos kernel + SIMD/GPU kernel，
   热路径中不出现虚调用、`std::function`、堆分配与运行时分派。
4. 强编译期安全：公共模板 API 全部由 concepts 约束。

## 2. 核心哲学

依赖链只有单向的一条：

```text
Execution -> Memory -> View -> Vector -> Operator -> Solver
```

FEM/FVM/FDM/FDTD/BEM 的共同部分不是 mesh，而是数据搬运、线性代数、算子、求解器与执行模型。
因此 numerix 不出现 `Mesh -> Field -> PDE -> Solver` 这条链。

## 3. 模块与目录

```text
include/numerix/
├── core/          scalar / concepts / error / assert
├── execution/     Kokkos 执行空间别名与查询
├── memory/        Kokkos 内存空间别名与结构契约
├── algebra/       StaticVector / StaticMatrix / Vector（后续：MultiVector、稀疏算子包装）
├── operator/      LinearOperator 概念、类型擦除边界、扁平算子流水线
├── solver/        线性与非线性求解器（后续）
├── numerics/      时间积分等数值方法（后续）
├── distributed/   MPI 层（后续，可选依赖）
├── diagnostics/   Logger（后续：Timer、ConvergenceMonitor）
└── backend/       可选后端 adapter（后续）
```

`src/` 只放无法放入头文件的实现（当前仅 `diagnostics/logger.cpp`）。

## 4. 依赖策略

必需依赖：Kokkos。负责并行执行、内存层级、设备抽象与 kernel 启动。
numerix 不实现 thread pool、CUDA wrapper、OpenMP 抽象或 allocator 抽象。

可选依赖一律通过 adapter 接入（`backend/eigen`、`backend/petsc`、`backend/ginkgo`、
`backend/hdf5`、`backend/mpi`、`backend/blas`、`backend/lapack`），可选依赖不得出现在核心头文件中。

## 5. 核心约定

### 5.1 标量

`Scalar` 概念只接受浮点与 `std::complex`。`RealOfT<T>` 给出实标量类型，
`Conj` / `SquaredMagnitude` 统一实数与复数的内积语义。

### 5.2 概念优先

公共模板 API 必须有概念约束：

```cpp
template <Scalar T, MemorySpace Mem = Memory>
class Vector;
```

概念表达结构契约而非继承关系，例如 `VectorLike` 只要求 `value_type`、`Size()`、`Data()`。

### 5.3 View 不再封装

不定义 `numerix::View`。`Kokkos::View` 是底层基础，直接使用；`numerix::Vector` 内部持有
`Kokkos::View`，并提供 `View()` 以便与 Kokkos 生态互操作。构造函数不隐式拷贝。

### 5.4 算子契约

```cpp
template <class Op, class X, class Y = X>
concept LinearOperator = requires(const Op& op, const X& x, Y& y) {
    { op.Apply(x, y) } -> std::same_as<void>;
};
```

`Apply` 只做 `y = A x`（不做 `y += A x`、不返回结果），保证算子实现、组合与测试都只有一种形式。
`Preconditioner` 是 `LinearOperator` 的语义别名。

### 5.5 扁平算子流水线

算子组合使用扁平表示而非嵌套模板递归：

```cpp
const auto op = Compose<Vector<double>>(size, B, C, D);
```

`Pipeline` 在构造时一次性分配中间缓冲，`Apply` 期间不再分配；模板实例数随算子个数线性增长，
避免 `compose(f, compose(g, h))` 造成的模板与代码体积膨胀。

### 5.6 错误处理

底层不使用异常。错误通过 `Status` 返回码显式传播；断言只用于内部实现的前提，
接口层应返回 `Status` 而非依赖断言。I/O 与顶层业务才考虑异常路径。

### 5.7 日志

日志必须是显式对象 `Logger`，不引入全局可变状态；默认级别 INFO，除 DEBUG 外不打在热循环内。
调用点一律使用宏封装：

```cpp
NUMERIX_LOG_INFO(logger, "solver converged");
NUMERIX_LOG_DEBUG(logger, "iteration residual");
```

`NUMERIX_LOG_DEBUG` 在 Release 构建中整体消除（未启用时仍以不求值的 `sizeof` 标记参数已使用，
避免 `-Werror` / `/WX` 下的未使用变量告警）。

## 6. 模板规则

1. 所有公共模板 API 必须由 concept 约束，禁止 `template <class T> void Solve(T);` 这类无约束接口。
2. 模板只用于静态多态、kernel 融合与零开销抽象；不用模板构建巨型类型系统或隐式魔法。
3. 运行期多态只用于求解器选择、后端选择与用户扩展，不进入 kernel、装配循环与向量运算。
4. 组合优于继承；模块之间依赖接口与共同数据结构，不依赖彼此的具体实现。
5. 类型擦除被显式限制在边界处（`AnyLinearOperator`），并在注释中标注允许使用的场景。

## 7. 命名与风格

沿用 Google Style C++：类型/概念/函数 PascalCase，变量与文件名 snake_case，
类成员变量以 `_` 结尾，常量 `k` + PascalCase。头文件使用 `#pragma once`。
格式化由仓库根目录 `.clang-format` 决定，不做逐行人工对齐。

## 8. 测试策略

测试必须验证数学性质，而不是只覆盖代码路径：

- 线性性 `A(ax + by) = aAx + bAy`
- 对称性 `x^T A y = y^T A x`
- 正定性 `x^T A x > 0`
- Jacobian 一致性（后续）：`J v ≈ (F(x + eps v) - F(x)) / eps`
- 组合语义：`Pipeline` 的结果必须与逐级手工施加一致

数值测试容差集中管理，不用硬编码的魔数散落各处。

## 9. 明确排除

不属于 numerix：mesh、element、shape function、quadrature rule 的用法、flux、边界条件、材料、
物理、几何内核、自适应加密、可视化。这些属于 `numerix-fem`、`numerix-fvm`、`numerix-fd`、
`numerix-bem` 等上层项目。

## 10. 决策记录

- D-1 类型擦除边界：`AnyLinearOperator` 允许存在于求解器选择、后端适配与用户扩展处；
  禁止出现在 kernel 与装配循环中。
- D-2 命名规范：早期草案中的 `apply/dot/norm` 小写下划线风格改为 Google Style 的
  `Apply/Dot/Norm`，与用户既有工程习惯一致。
- D-3 定长代数命名：`StaticVector` / `StaticMatrix`（而非 `Vector<T, 3>` 特化），
  避免与动态 `Vector<T, MemorySpace>` 在调用点产生歧义。
- D-4 错误模型：不使用异常，采用 `Status` 返回码（原草案中的 `std::expected` 不可用）。
- D-5 日志：显式 `Logger` 对象 + 宏封装，不使用全局单例。
- D-6 警告传播：编译告警是 PRIVATE 需求，不通过 PUBLIC 传播给使用方；
  `numerix_options` 只承载使用方必须满足的条件。
- D-7 依赖版本：Kokkos 版本固定在 `cmake/Dependencies.cmake`，不追随 master。
- D-8 本机工具链：开发机使用 clang-cl（VS 2022 工具链）+ Ninja；GCC 环境待补验证。
- D-9 基准方法：每个变体取多轮中的最优一轮（共享/虚拟化机器上单次测量波动可达 30%），
  并且始终同时测量裸 Kokkos kernel 与手写循环，作为零开销抽象的对照基线。
- D-10 安装导出：`numerix_options` / `numerix_warnings` 与 numerix 同属一个导出集；
  Kokkos 由 CPM 拉取时随安装一并导出，使用方通过 `find_dependency(Kokkos)` 获得。

## 11. 最终架构

```text
FEM / FVM / FDM / FDTD / BEM
              |
           numerix
   Operator -> Linear Algebra -> Kokkos Runtime
              |
     CPU / CUDA / HIP / SYCL
              |
External: PETSc / Ginkgo / Eigen / HDF5 / MPI
```
