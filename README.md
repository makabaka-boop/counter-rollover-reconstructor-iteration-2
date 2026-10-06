# 冷库机械累计表 · 绝对计数轨迹恢复

冷库货位的机械累计表只保留固定模数 `modulus` 内的读数。跨零回绕（甚至连续
多圈）或漏抄后，直接把两次读数相减会把**正常耗用**误记成**倒退**。本服务
根据相邻抄表位置的步长约束，恢复唯一可复算的绝对计数序列。

## 数学模型

设第 `i` 个位置的绝对计数为 `A[i]`（非负、单调不减），抄表读数为
`r[i] = A[i] mod m`，每步增量 `d[i] = A[i+1] - A[i]`。

- 已知读数：`A[i] ≡ readings[i] (mod m)`，首项 `A[0] = readings[0]`（空仓库
  起步，首项回绕次数为 0）；
- 未知读数（`null`）：余数与回绕次数都自由；
- 步长约束：`minStep[i] <= d[i] <= maxStep[i]`。

相邻已知点 a→b（读数 r_a、r_b）之间的总增量必为

```
S = m·q + (r_b - r_a),  q 为非负整数（本段累计回绕数）
```

故每个已知段对 `q` 给出一个整数区间
`[ceil((smin-(r_b-r_a))/m), floor((smax-(r_b-r_a))/m)]`。

求解两级目标：

1. **先最小化最终绝对计数** `A[-1]`（等价于最小化终点回绕次数）：对各已知点
   的回绕次数做一次 O(n) 的前向区间传播；
2. **再取完整序列字典序最小者**：后向传播得到每个已知点允许的回绕区间后，
   从左到右贪心——每一步在“仍能延伸出可行后缀”的前提下取最小增量。剩余边
   的可行总增量是连续整数区间，因此每次选择只需常数次大整数取整运算。

**实现从不枚举绝对计数值或其上界**：`modulus` 可取任意大整数，复杂度只与
读数项数 `n` 有关（O(n) 次区间运算）。

无解时返回 `INCONSISTENT` 及**最早无法延伸的位置**（最小的 `p`，使得位置
`0..p` 不存在任何合法前缀赋值）。

## API

`POST /trajectory`

```json
{
  "modulus": 10,
  "readings": [8, null, 2],
  "minStep": [0, 0],
  "maxStep": [6, 6]
}
```

约束（违反返回 **422**）：

- `modulus`：正整数；
- `readings`：2–300 项，首尾必须已知，已知值满足 `0 <= v < modulus`，其余为
  `null`；仅接受整数/`null`（拒绝浮点、字符串、布尔）；
- `minStep`、`maxStep`：长度均为 `len(readings)-1`，逐项满足
  `0 <= minStep <= maxStep <= 2*modulus`（带交接时，交接点及以后的边按
  新模数约束 `<= 2*newModulus`）；
- 未知字段一律拒绝。

成功（HTTP 200）：

```json
{
  "status": "OK",
  "absolute": [8, 8, 12],
  "increments": [0, 4],
  "cumulativeWraps": [0, 0, 1]
}
```

无解（HTTP 200）：

```json
{ "status": "INCONSISTENT", "position": 1 }
```

### 换表交接（可选 `handoff`）

两次抄表之间更换机械累计表时，新表从自己的读数起步，但货位真实累计耗用
不能归零。请求可增加一次换表交接：

```json
{
  "modulus": 10,
  "readings": [8, 2, 30],
  "minStep": [0, 0],
  "maxStep": [6, 60],
  "handoff": {
    "position": 1,
    "newModulus": 100,
    "openingReading": 0
  }
}
```

- `handoff.position`：旧表最后一次读数所在位置（该点 `readings` 必须
  已知，按**旧**模数校验）；允许取 `0`（起点就换表）或 `len(readings)-1`
  （末点才换表）；
- `handoff.newModulus`：新表模数（正整数）；
- `handoff.openingReading`：同一时点新表的开表读数，满足
  `0 <= openingReading < newModulus`；
- 交接动作本身耗用为零：没有步长跨过交接点。位置 `0..position` 按旧模数
  还原（边 `0..position-1` 属于旧表），之后按新模数（边
  `position..n-2` 由新表累计，其 `maxStep <= 2*newModulus`）；
- 交接点之后的已知读数必须满足 `0 <= v < newModulus`。

设新表自身计数为 B（从 `openingReading` 起算、回绕数从 0 计起），真实
绝对计数在交接点之后为 `A[i] = A[position] + B[i] - openingReading`。
两段拼成一条单调不减的绝对计数轨迹，仍按“先最小化最终绝对计数、再取整
条轨迹字典序最小”求解；原有 `minStep`、`maxStep` 逐段约束真实耗用。

成功（HTTP 200）：两台表分别列出回绕次数（长度均为 `len(readings)`，非
本表区间的位置为 `null`），新表清零**不**报为负耗用：

```json
{
  "status": "OK",
  "absolute": [8, 12, 42],
  "increments": [4, 30],
  "cumulativeWraps": null,
  "oldMeterWraps": [0, 1, null],
  "newMeterWraps": [null, 0, 0]
}
```

交接位置、开表读数非法（**422**）或任一段不可行（HTTP 200 的
`INCONSISTENT`，`position` 为最早无法延伸的全局位置，新表段下标已换算
回全局）时都不会产生局部成功轨迹。省略 `handoff`（或传 `null`）时请求
与响应保持原样（仍返回 `cumulativeWraps`）。

### 盘点锚点（可选 `inventoryAnchor`）

人工盘点得到某个**已有读数位置**的绝对累计数后，可将它作为硬约束传入：

```json
{
  "modulus": 10,
  "readings": [8, 2, 30],
  "minStep": [0, 0],
  "maxStep": [20, 200],
  "handoff": {
    "position": 1,
    "newModulus": 100,
    "openingReading": 0
  },
  "inventoryAnchor": {
    "position": 2,
    "absoluteCount": 142
  }
}
```

- `inventoryAnchor.position`：必须是 `readings` 中非 `null` 的已有读数位置；
- `inventoryAnchor.absoluteCount`：该位置经人工核准的非负绝对计数。

求解器不会先恢复一条轨迹再检查是否“碰巧等于”锚点，而是先把锚点值连同
各步区间一起传播。锚点位于交接之后时，旧表交接值 `A[h]` 与新表自身计数
`B[p]` 必须共同满足
`absoluteCount = A[h] + B[p] - openingReading`；算法用整圈数线性方程
联合求解，不能将两段各自独立取最优后拼接。锚点位于交接点时，按同一物理
时刻核对该绝对数。

`inventoryAnchor.absoluteCount` 与位置读数不同余、圈数方程无解，或无法满足
前后步长区间时，均属于合法请求但不可恢复：HTTP 200 返回
`INCONSISTENT` 和最早无法延伸的位置，绝不返回半条轨迹。位置越界、指向
`null`、绝对数为负或类型非法时返回 422。省略 `inventoryAnchor`（或传
`null`）时，原有请求和响应字段保持不变。

另有 `GET /health`。

## 运行

Docker Compose（Python 3.12 镜像）：

```bash
docker compose up --build
# curl -s localhost:8000/health
```

本地：

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest -q
```

## 测试

- `tests/test_solver.py`：短序列穷举增量对拍（约 18 万个确定性小实例，穷举
  所有合法增量组合求两级最优）、随机对拍、连续回绕、多个空洞、并列解取字典
  序、最早不可延伸位置、大模数（10⁶⁰）与 300 项性能；
- `tests/test_handoff.py`：换表交接的独立小规模穷举对拍（约 10 万个
  确定性小实例，穷举两侧所有合法增量组合与开表读数，覆盖两侧漏抄、多圈
  回绕、交接边界 `h=0`/`h=n-1` 与大模数）、随机对拍、新表清零不报负耗用、
  两台表分别的回绕次数与无解位置换算；
- `tests/test_anchor.py`：盘点锚点短序列穷举对拍，覆盖锚点在交接前、
  交接点、交接后、空洞、两侧多圈回绕、起点换表时由锚点反推交接累计数，
  以及非法同余/线性方程无解时的最早不可延伸位置；
- `tests/test_api.py`：422 校验矩阵、成功/无解响应形态、交接 422 矩阵与
  两表回绕响应。
