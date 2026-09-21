# numerix Design Document

> C++20 + Kokkos-native 的通用科学计算基础库。numerix 不提供 FEM/FVM/BEM 框架，
> 而提供所有离散方法共享的数值执行模型。

状态：骨架已落地（v0.1，含 Kokkos-native 纠偏）。本文件是唯一的设计记录；
与代码不一致时以本文件为准并同步修正代码。

## 1. 设计目标

1. 仅 C++20。不使用 `std::mdspan`、`std::expected`、modules，因为科学计算环境升级慢。
2. Kokkos-native 可移植性：Serial / OpenMP / CUDA / HIP / SYCL 由 `Kokkos::ExecutionSpace`、
   `Kokkos::MemorySpace`、`Kokkos::View` 提供，numerix 不重新设计执行抽象。
3. 零开销抽象：`Apply(exec, x, y)` 的最终形态是 Kokkos kernel + SIMD/GPU kernel，
   热路径中不出现虚调用、`std::function`、堆分配与运行时分派。
4. 强编译期安全：公共模板 API 全部由 concept 约束；非法空间组合在模板实例化阶段就被拒绝。
5. 同一份源码在主机与设备上语义一致：标量、定长代数与所有 kernel 内可调用的函数都必须是
   Kokkos 可移植的。

## 2. 核心哲学

依赖链只有单向的一条：

```text
Execution -> Memory -> View -> Vector -> Operator -> Solver
```

FEM/FVM/FDM/FDTD/BEM 的共同部分不是 mesh，而是数据搬运、线性代数、算子、求解器与执行模型。
因此 numerix 不出现 `Mesh -> Field -> PDE -> Solver` 这条链。

职责划分（越靠下越接近 Kokkos，越靠上越接近数学语义）：

```text
algebra / solver semantics      numerix：数学语义与接口
LinearOperator                  numerix：编译期算子契约
Kokkos Views / containers       numerix::Vector：owning 存储
Kokkos Kernels operations       BLAS / 稀疏容器：成熟 kernel
explicit ExecutionSpace instance  stream / queue
Kokkos                          machine model
CPU / CUDA / HIP / SYCL
```

## 3. 模块与目录

```text
include/numerix/
├── core/          scalar（ScalarTraits）/ concepts（Kokkos 概念再导出 + numerix 概念）
├── execution/     Kokkos 执行空间别名
├── memory/        Kokkos 内存空间别名
├── algebra/       blas1 / StaticVector / StaticMatrix / Vector（后续：MultiVector、稀疏包装）
├── operator/      LinearOperatorFor 编译期算子契约
├── solver/        线性与非线性求解器（后续）
├── numerics/      时间积分等数值方法（后续）
├── distributed/   MPI 层（后续，可选依赖）
├── diagnostics/   Logger（后续：Timer、ConvergenceMonitor）
└── backend/       可选后端 adapter（后续）
```

`src/` 只放无法放入头文件的实现（当前仅 `diagnostics/logger.cpp`）。

## 4. 依赖策略

必需依赖两个，且必须版本同步：

- **Kokkos Core**：执行空间、内存空间、`View`、并行原语。
- **Kokkos Kernels**：portable BLAS-1/2/3 与稀疏容器（`KokkosSparse::CrsMatrix` 属于它，不属于 Core）。

numerix 不实现 thread pool、CUDA wrapper、OpenMP 抽象、allocator 抽象，也不自己重写 BLAS 内核：
numerix 拥有语义与接口，成熟 kernel 交给 Kokkos Kernels。

可选依赖一律通过 adapter 接入（`backend/eigen`、`backend/petsc`、`backend/ginkgo`、`backend/hdf5`、
`backend/mpi`、`backend/lapack`），可选依赖不得出现在核心头文件中。

## 5. 核心约定

### 5.1 标量：ScalarTraits 是唯一入口

一个类型是 numerix 标量，当且仅当它有合法的 `ScalarTraits<T>` 特化：

```cpp
template <class T> struct ScalarTraits;

template <std::floating_point T>
struct ScalarTraits<Kokkos::complex<T>> {
    using value_type = Kokkos::complex<T>;
    using real_type  = T;

    KOKKOS_INLINE_FUNCTION static constexpr value_type Conj(const value_type&) noexcept;
    KOKKOS_INLINE_FUNCTION static constexpr real_type  SquaredMagnitude(const value_type&) noexcept;
};
```

- 复数标量是 `Kokkos::complex`，不是 `std::complex`：后者在设备端不可用。
- `Scalar` 概念由"存在合法 traits"定义，而不是由 `is_floating_point || is_std_complex` 定义；
  half/bfloat、AD 标量、混合精度以后加特化即可，不需要改动概念体系。
- 内核只通过 `ScalarTraits` 取实部类型与共轭/模平方语义（`RealOfT<T>`、`Conj`、`SquaredMagnitude`），
  这三个函数都是 `KOKKOS_INLINE_FUNCTION constexpr`。

### 5.2 概念优先，且以 Kokkos 概念为准

Kokkos 已经给出 `Kokkos::ExecutionSpace` / `Kokkos::MemorySpace` 两个 concept 与
`Kokkos::SpaceAccessibility`，numerix 直接再导出，不重复定义结构契约：

```cpp
using Kokkos::ExecutionSpace;
using Kokkos::MemorySpace;

template <class Exec, class Mem>
concept AccessibleFrom = ExecutionSpace<Exec> && MemorySpace<Mem> &&
    static_cast<bool>(Kokkos::SpaceAccessibility<Exec, Mem>::accessible);

template <class V, class Exec = DefaultExecutionSpace>
concept DenseVector = Kokkos::is_view_v<V> && V::rank == 1 &&
    AccessibleFrom<Exec, typename V::memory_space>;

template <class V, class Exec = DefaultExecutionSpace>
concept MutableDenseVector = DenseVector<V, Exec> && !std::is_const_v<typename V::value_type>;
```

- 契约围绕 `rank` / `value_type` / `extent` / `memory_space` / 可访问性 / 可写性建立，
  **不围绕裸指针**：设备端 `Data()` 在主机上不可解引用，subview、跨步视图与分布式向量也不该用指针表达。
- 非法的空间组合（设备内存 + 主机执行流）在模板实例化阶段被拒绝，而不是运行时出错。

### 5.3 View 不再封装，Vector 只负责所有权

不定义 `numerix::View`。`Kokkos::View` 是底层基础，直接使用。

`numerix::Vector` 是 **owning、move-only 的存储**，没有数学成员函数：

```cpp
template <Scalar T, MemorySpace Mem = DefaultMemorySpace>
class Vector {
public:
    using view_type = Kokkos::View<T*, Mem>;
    using const_view_type = typename view_type::const_type;

    explicit Vector(std::size_t size);
    explicit Vector(view_type values) noexcept;   // 接管已有 View 的所有权

    Vector(const Vector&) = delete;
    Vector(Vector&&) noexcept = default;

    std::size_t Size() const noexcept;

    T* Data() noexcept;
    const T* Data() const noexcept;

    view_type View() noexcept;
    const_view_type View() const noexcept;
};
```

理由与后果：

- Kokkos `View` 的拷贝是引用计数式句柄共享，与"数学向量"的直觉冲突；隐式共享会让 `a = b` 语义难预测。
  因此 `Vector` 删除拷贝，需要复制时显式写 `Clone(exec, x)` / `DeepCopy(exec, dst, src)`。
- 需要别名、subview 或跨步视图时直接用 `Kokkos::View`，不通过 `Vector` 表达。
- `const Vector` 只能交出 `const_view_type`，与 Kokkos 的 `View<double*>`（const 句柄、可写数据）与
  `View<const double*>`（只读数据）区分一致。

### 5.4 BLAS-1：numerix 定接口，Kokkos Kernels 出内核

```cpp
Scale(exec, alpha, x);      Axpy(exec, alpha, x, y);
Dot(exec, x, y);            Norm2(exec, x);
```

- 每个操作显式接收 execution-space instance。同一内存空间配不同 instance 就是不同的
  CUDA/HIP/SYCL stream/queue，这是计算/通信重叠、MPI halo overlap、多流与异步流水线的前提；
  执行流不能由 `y` 隐含决定。
- 不带 instance 的重载内部使用默认 instance，只作便利。
- 操作对象是 `Kokkos::View`：subview、跨步视图、MultiVector 的列都能直接参与；
  `Vector` 通过 `View()` 交出自己拥有的 View。numerix 不为容器再复制一套代数接口。
- 输出 View 按 Kokkos 约定用 const 引用传递（句柄是 const 的，数据仍可写），
  可写性由 `MutableDenseVector` 约束保证。
- 内积语义与 Kokkos Kernels 的 BLAS 一致：`Dot(x, y) = sum(conj(x(i)) * y(i))`；
  `Norm2` 对复数向量也返回实标量。

### 5.5 算子契约：编译期 + 显式执行流

```cpp
template <class Op, class Exec, class X, class Y = X>
concept LinearOperatorFor = ExecutionSpace<Exec> &&
    requires(const Op& op, const Exec& exec, const X& x, Y& y) {
        { op.Apply(exec, x, y) } -> std::same_as<void>;
    };
```

`Apply` 只做 `y = A x`（不做 `y += A x`、不返回结果），在给定的 execution-space instance 上执行。

融合形式是**可选**能力，单独用概念表达：

```cpp
template <class Op, class Exec, class X, class Y = X>
concept ScaledLinearOperatorFor = LinearOperatorFor<Op, Exec, X, Y> &&
    requires(const Op& op, const Exec& exec, const X& x, Y& y, ScalarOf<Y> alpha, ScalarOf<Y> beta) {
        { op.Apply(exec, alpha, x, beta, y) } -> std::same_as<void>;
    };
```

`y = alpha A x + beta y` 在 Krylov 迭代里很常见，KokkosSparse 的 `spmv` 原生支持这种形式。
不要求所有算子实现它；回退路径（`Apply` + `Axpby`）需要调用方提供 workspace，
因此 dispatcher 与 workspace 所有权一起留给求解器阶段（见 D-3）。

### 5.6 索引类型：区分 size / local ordinal / global ordinal

| 名字 | 语义 | 当前形态 |
| --- | --- | --- |
| `size_type` | 元素个数、extent | `std::size_t` / `View::size_type` |
| local ordinal | 单个内存空间内的下标 | `View::index_type`（Kokkos 默认 64 位） |
| global ordinal | 分布式/稀疏全局编号 | 阶段 1 稀疏层引入（KokkosSparse 用 `Ordinal`） |

- 不创造万能 `Index`。
- `KOKKOS_LAMBDA(int i)` + `static_cast<int>(size)` 不是 numerix 的写法：
  RangePolicy 的索引类型跟随 View 的 `index_type`：

  ```cpp
  Kokkos::RangePolicy<Exec, Kokkos::IndexType<typename View::index_type>>(exec, 0, view.extent(0))
  ```

### 5.7 错误处理

- numerix 底层 API 不使用异常作为正常控制流，也不使用 RTTI。
- 契约违背（内部前提）用 `KOKKOS_ASSERT` 表达，只在无法用类型系统表达时使用。
- 第三方依赖的异常按其自身语义处理：Kokkos Kernels 的 host 端 API 对非法 extent 会抛
  `runtime_exception`，numerix 不为"异常纯净"与其对抗。
- 不可恢复错误交给 Kokkos 的 abort 路径。
- 求解器状态（收敛/最大迭代/breakdown）属于 solver 层语义，届时定义 `SolveResult`；
  core 不提供跨模块的通用 `Status` 枚举（见 D-6）。

### 5.8 日志

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
3. 运行期多态只允许出现在求解器选择、后端适配与用户扩展边界，不进入 kernel、装配循环与向量运算。
   当前 numerix 没有任何类型擦除包装；引入时必须同时说明它不在热路径上。
4. 组合优于继承；模块之间依赖接口与共同数据结构，不依赖彼此的具体实现。
5. 任何进入 kernel 的 numerix 函数必须是 `KOKKOS_INLINE_FUNCTION`（能 constexpr 的一律 constexpr）。

## 7. 命名与风格

沿用 Google Style C++：类型/概念/函数 PascalCase，变量与文件名 snake_case，
类成员变量以 `_` 结尾，常量 `k` + PascalCase。头文件使用 `#pragma once`。
格式化由仓库根目录 `.clang-format` 决定，不做逐行人工对齐。

别名一律写全名，不用歧义词：`DefaultExecutionSpace` / `DefaultHostExecutionSpace` /
`DefaultMemorySpace` / `DefaultHostMemorySpace`。刻意不用 `Device`：`Kokkos::Device<Exec, Mem>`
与 `KokkosSparse::CrsMatrix` 的 `Device` 模板参数指"执行空间 + 内存空间"的组合，含义不同。

## 8. 测试策略

测试必须验证数学性质与真实契约，而不是只覆盖代码路径：

- 线性性 `A(ax + by) = aAx + bAy`
- 对称性 `x^T A y = y^T A x`
- 正定性 `x^T A x > 0`
- 复数内积只对第二个参数线性：`Dot(x, y) = conj(x) . y`，且 `Dot(x, x)` 为实数
- 融合算子语义：`Apply(exec, alpha, x, beta, y)` 与 `alpha (A x) + beta y` 一致
- 所有权与 const 语义：`Vector` 不可拷贝、移动转移所有权、`const Vector` 只给 `const` View
- 定长代数在编译期求值（实数与复数都覆盖）
- Jacobian 一致性（后续）：`J v ≈ (F(x + eps v) - F(x)) / eps`

数值测试容差集中管理，不用硬编码的魔数散落各处。

## 9. 明确排除

不属于 numerix：mesh、element、shape function、quadrature rule 的用法、flux、边界条件、材料、
物理、几何内核、自适应加密、可视化。这些属于 `numerix-fem`、`numerix-fvm`、`numerix-fd`、
`numerix-bem` 等上层项目。

## 10. 决策记录

- D-1 依赖集合：Kokkos Core + Kokkos Kernels 都是必需依赖（同 tag，版本同步）；
  BLAS-1 不再自研，numerix 只固定接口与语义。
- D-2 标量体系：`ScalarTraits<T>` 定义标量，复数一律 `Kokkos::complex`；
  `Scalar` 概念由 traits 存在性定义，为 half/bfloat/AD 留扩展点。
- D-3 算子契约：`LinearOperatorFor<Op, Exec, X, Y>` 显式携带 execution-space instance；
  融合能力用 `ScaledLinearOperatorFor` 单独表达，dispatcher 与 workspace 所有权推迟到求解器阶段
  （回退路径需要 workspace，过早引入会强迫算子在 Apply 内部持有可变状态）。
- D-4 `Vector` 是 owning、move-only 存储，无数学成员函数；复制必须显式 `Clone` / `DeepCopy`，
  别名/子视图直接用 `Kokkos::View`。`const Vector` 只交出 `const_view_type`。
- D-5 删除 `Pipeline` 与 `AnyLinearOperator`：前者隐含 `X -> X -> X`、维数不变、独占可变 workspace
  等强假设（gradient / prolongation / restriction / 混合块算子都不满足），
  后者没有真实运行期多态使用点且会把虚调用带进 Krylov 热循环。
  等出现真实组合需求再决定表达式层或 workspace planner。
- D-6 删除 core 的通用 `Status`：`NotConverged` / `Breakdown` 是 solver 语义，放在 core 会让
  core 依赖 solver，方向反了；求解器状态改在 solver 层用 `SolveResult` 表达。
- D-7 别名改名：`Device`/`Memory` → `DefaultExecutionSpace`/`DefaultMemorySpace`
  （加 host 变体），避免与 `Kokkos::Device` 及 `KokkosSparse` 的 `Device` 混淆。
- D-8 索引类型：区分 `size_type` / local ordinal / global ordinal，禁止把 extent 窄化成 `int`；
  RangePolicy 索引类型跟随 View 的 `index_type`。
- D-9 CMake 使用需求收缩：`numerix_options` 只保留 `cxx_std_20`、解析公共头文件必需的
  MSVC 开关与 `NUMERIX_DEBUG` 定义；优化级别、`-fPIC`（改为目标属性 `POSITION_INDEPENDENT_CODE`）、
  `-rdynamic`、`-fno-strict-aliasing`、覆盖率、构建类型、并行度、`compile_commands` 全部移除或
  限制在 `PROJECT_IS_TOP_LEVEL` 内。子项目默认不构建 tests/examples/benchmarks，
  且不再使用 `CMAKE_SOURCE_DIR`（子项目下那是父项目的目录）。
- D-10 警告传播：编译告警是 PRIVATE 需求，不通过 PUBLIC 传播给使用方；
  `numerix_options` 只承载使用方必须满足的条件。
- D-11 命名规范：早期草案中的 `apply/dot/norm` 小写下划线风格改为 Google Style 的
  `Apply/Dot/Norm`。
- D-12 定长代数命名：`StaticVector` / `StaticMatrix`（而非 `Vector<T, 3>` 特化），
  避免与动态 `Vector<T, MemorySpace>` 在调用点产生歧义；存储用 `Kokkos::Array`。
- D-13 日志：显式 `Logger` 对象 + 宏封装，不使用全局单例。
- D-14 依赖版本：Kokkos / Kokkos Kernels 版本固定在 `cmake/Dependencies.cmake`，不追随 master。
- D-15 本机工具链：开发机使用 clang-cl（VS 2022 工具链）+ Ninja；GCC/OpenMP 与设备后端待补验证。
- D-16 基准方法：每个变体取多轮中的最优一轮（共享/虚拟化机器上单次测量波动可达 30%），
  并且始终同时测量裸 Kokkos kernel 与手写循环，作为零开销抽象的对照基线。
- D-17 安装导出：`numerix_options` / `numerix_warnings` 与 numerix 同属一个导出集；
  依赖由 CPM 拉取时随安装一并导出，使用方通过 `find_dependency(Kokkos)` /
  `find_dependency(KokkosKernels)` 获得。

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
