"""绝对计数轨迹恢复核心算法。

机械累计表只保留模 ``modulus`` 内的读数，因此单次或连续多次跨零回绕后，
直接用读数相减会把正常耗用（绝对计数只增不减）误判成倒退。

把第 ``i`` 个位置的绝对计数写成

    A[i] = residue[i] + wraps[i] * modulus

其中已知位置 ``residue[i]`` 为抄表读数、``wraps[i]`` 为累计回绕次数（非负
整数）；未知位置两者都自由。每一步要求

    minStep[i] <= A[i + 1] - A[i] <= maxStep[i].

记两个相邻已知点 a、b 的读数为 r_a、r_b，回绕次数为 k_{a}、k_b，则段内
总增量

    S = A_b - A_a = m * q + (r_b - r_a),   q = k_b - k_a >= 0.

优化目标（两级）：

1. 先最小化最终绝对计数 ``A[-1]``（等价于最小化最终回绕次数 k_t）；
2. 在此前提下使完整序列 ``A`` 字典序最小——等价于从左到右贪心，每一步在
   仍能延伸出可行后缀时取最小的当前值。

实现只在“回绕次数构成的整数区间”上做线性次数的区间传播与常数次大整数
运算，不枚举绝对计数值本身（绝对值可远超 modulus 的多项式倍）。

换表交接（``handoff``）
-----------------------

抄表中途可更换一次机械表：旧表在位置 ``h`` 留下最后一次读数（模数 m₀），
同一时点新表以开表读数 ``s`` 起步（模数 m₁，新表回绕数从 0 计起）。交接
动作本身耗用为零——没有步长跨过 ``h``：边 ``0..h-1`` 属于旧表段，边
``h..n-2`` 属于新表段。设新表自身计数为 B，则真实绝对计数

    A[i] = A[h] + B[i] - s,   i >= h.

两段除共享物理时点 ``A[h]`` 外无耦合，各自满足“先最小化末值、再字典序
最小”即拼成整条轨迹的两级最优；两台表的回绕次数分别列出，新表清零不产生
负耗用（真实轨迹只增不减，新表段增量仍逐边受 minStep/maxStep 约束）。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Handoff:
    """一次换表交接：旧表在 ``position`` 处读完最后一次（该点读数必须已知），
    同一时点新表以 ``opening_reading`` 开表，新表模数为 ``new_modulus``。

    交接动作本身耗用为零：没有步长跨过该点——``position-1`` 及以前的边属于
    旧表，``position`` 及以后的边属于新表。
    """

    position: int
    new_modulus: int
    opening_reading: int


@dataclass(frozen=True)
class Solved:
    """有解时的恢复结果。

    无交接时 ``cumulative_wraps`` 为单台表的逐点累计回绕数。有交接时该字段
    为 ``None``，改由 ``old_meter_wraps`` / ``new_meter_wraps`` 分别给出两台
    表的回绕数（非本表区间的位置取 ``None``）。
    """

    absolute: list[int]
    increments: list[int]
    cumulative_wraps: list[int] | None
    old_meter_wraps: list[int | None] | None = None
    new_meter_wraps: list[int | None] | None = None


@dataclass(frozen=True)
class Inconsistent:
    """无解：``position`` 是最早无法延伸的位置下标。"""

    position: int


Result = Solved | Inconsistent


def _ceil_div(a: int, b: int) -> int:
    """正数分母的向上取整除法（分子可为负）。"""
    return -((-a) // b)


def solve(
    modulus: int,
    readings: list[int | None],
    min_steps: list[int],
    max_steps: list[int],
    handoff: Handoff | None = None,
) -> Result:
    if handoff is not None:
        return _solve_with_handoff(
            modulus, readings, min_steps, max_steps, handoff
        )

    return _solve_segment(
        modulus, readings, min_steps, max_steps, readings[0]
    )


def _solve_with_handoff(
    old_modulus: int,
    readings: list[int | None],
    min_steps: list[int],
    max_steps: list[int],
    handoff: Handoff,
) -> Result:
    """两段拼接：旧表段 [0..h]（旧模数）与新表段 [h..n-1]（新模数）。

    交接位置 ``h`` 处两段共享同一个物理累计值 ``A[h]``，但新表自身的计数
    从 ``opening_reading`` 起步（新表回绕数从 0 计起）：
        A[i] = A[h] + B[i] - opening,   i >= h，
    其中 B 为新表自身的绝对计数。两段之间没有任何步长跨越（交接耗用为
    零），故两级目标可分别求解：先各自最小化末值（共同最小化 A[-1]），
    再各自字典序最小（拼成整条轨迹的字典序最小）。
    """
    n = len(readings)
    h = handoff.position
    m_new = handoff.new_modulus
    opening = handoff.opening_reading

    # 旧表段：positions 0..h，边 0..h-1。
    old_res = _solve_segment(
        old_modulus,
        readings[: h + 1],
        min_steps[:h],
        max_steps[:h],
        readings[0],
    )
    if isinstance(old_res, Inconsistent):
        return old_res
    assert old_res.old_meter_wraps is None and old_res.new_meter_wraps is None

    # 新表段：positions h..n-1，边 h..n-2。局部首值固定为开表读数（新表
    # 此刻尚未回绕）；注意交接位置的旧表读数 readings[h] 不属于新表，新表
    # 在该位置的读数是 opening，必须替换后再切片。
    new_res = _solve_segment(
        m_new,
        [opening] + readings[h + 1:],
        min_steps[h:],
        max_steps[h:],
        opening,
    )
    if isinstance(new_res, Inconsistent):
        return Inconsistent(new_res.position + h)

    join_value = old_res.absolute[-1]
    shift = join_value - opening

    absolute = old_res.absolute + [
        b + shift for b in new_res.absolute[1:]
    ]
    increments = old_res.increments + new_res.increments

    old_wraps: list[int | None] = list(old_res.cumulative_wraps) + [None] * (
        n - h - 1
    )
    new_wraps: list[int | None] = [None] * h + list(
        new_res.cumulative_wraps
    )

    return Solved(
        absolute=absolute,
        increments=increments,
        cumulative_wraps=None,
        old_meter_wraps=old_wraps,
        new_meter_wraps=new_wraps,
    )


def _solve_segment(
    m: int,
    seg_readings: list[int | None],
    seg_min: list[int],
    seg_max: list[int],
    first_value: int,
) -> Result:
    """单子表段的两级最优恢复。``seg_readings[0]`` 必须已知，且首项绝对
    计数固定为 ``first_value``（满足
    ``first_value ≡ seg_readings[0] (mod m)``）。返回的下标均为段内局部
    下标（从 0 起）。"""
    n = len(seg_readings)

    known_idx = [i for i, v in enumerate(seg_readings) if v is not None]
    t = len(known_idx)

    # 步长前缀和，段和 O(1)：smin(a,b)=prefix_min[b]-prefix_min[a]。
    prefix_min = [0] * (n + 1)
    prefix_max = [0] * (n + 1)
    for i in range(n - 1):
        prefix_min[i + 1] = prefix_min[i] + seg_min[i]
        prefix_max[i + 1] = prefix_max[i] + seg_max[i]
    prefix_min[n] = prefix_min[n - 1]
    prefix_max[n] = prefix_max[n - 1]

    # 每个已知段 j（j=1..t-1）：known_idx[j-1] -> known_idx[j]。
    seg_qlo = [0] * t      # 段回绕数 q 的下界 ceil((smin-(r_b-r_a))/m)
    seg_qhi = [0] * t      # 段回绕数 q 的上界 floor((smax-(r_b-r_a))/m)
    for j in range(1, t):
        a = known_idx[j - 1]
        b = known_idx[j]
        r_a = seg_readings[a]
        r_b = seg_readings[b]
        smin = prefix_min[b] - prefix_min[a]
        smax = prefix_max[b] - prefix_max[a]
        e = r_b - r_a
        seg_qlo[j] = _ceil_div(smin - e, m)
        seg_qhi[j] = (smax - e) // m
        # smin>=0 且 r_b-r_a <= m-1 保证 qlo>=0（A 单调不减）。
        if seg_qlo[j] > seg_qhi[j]:
            # 段本身就无法在步长范围内凑出总增量；段内未知点总能延伸，
            # 最早无法匹配的是该已知端点（前 j-1 个已知点均可达）。
            return Inconsistent(b)

    # 首项绝对计数 = 余数 + k0*m；旧表段与新开表段均有 k0=0。
    k0 = (first_value - seg_readings[0]) // m

    # ------------------------------------------------------------------
    # 1) 前向传播：F_j = [loF_j, hiF_j] 为 k_j 的可达整数区间，k_0 固定。
    # ------------------------------------------------------------------
    lo_f = [0] * t
    hi_f = [0] * t
    lo_f[0] = hi_f[0] = k0
    for j in range(1, t):
        lo_f[j] = lo_f[j - 1] + seg_qlo[j]
        hi_f[j] = hi_f[j - 1] + seg_qhi[j]
        # seg_qlo<=seg_qhi 已在段构造时检查；区间相加后只会更宽，不会变空。

    # 最小最终绝对计数 ⇔ 最小 k_{t-1} ⇔ loF_{t-1}。
    final_k = lo_f[t - 1]

    # ------------------------------------------------------------------
    # 2) 后向传播：G_j = [loG_j, hiG_j] 由 k_{t-1} = final_k 倒推；
    #    贪心时允许集合 T_j = F_j ∩ G_j（仍是整数区间）。
    # ------------------------------------------------------------------
    lo_g = [0] * t
    hi_g = [0] * t
    lo_g[t - 1] = hi_g[t - 1] = final_k
    for j in range(t - 1, 0, -1):
        lo_g[j - 1] = lo_g[j] - seg_qhi[j]
        hi_g[j - 1] = hi_g[j] - seg_qlo[j]

    # 后缀步长和：suffix_min[i] = sum(seg_min[i:n-1])，i ∈ [0, n-1]。
    suffix_min = [0] * n
    suffix_max = [0] * n
    for i in range(n - 2, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + seg_min[i]
        suffix_max[i] = suffix_max[i + 1] + seg_max[i]

    # ------------------------------------------------------------------
    # 3) 左到右贪心重建。
    #
    # 段内当前边 d_i，已选前缀增量 P（段内 d_a..d_{i-1} 之和），剩余边
    # 总增量 R ∈ [rlo, rhi]（连续整数区间）。段总量
    #     P + d + R = m*x + e,   x = k_b - k_cur（本段回绕数），
    # 且 x ∈ [T_lo - k_cur, T_hi - k_cur]。对固定 x：
    #     d ∈ [m*x - P - rhi + e, m*x - P - rlo + e] ∩ [minStep, maxStep]。
    # 取最小可行 x，再取区间内最小 d，即得字典序最优的当前 A 值。
    # ------------------------------------------------------------------
    absolute: list[int] = [first_value]
    increments: list[int] = []

    cur = first_value     # 当前绝对计数
    k_cur = k0            # 当前已知点的回绕次数
    P = 0                 # 段内已选增量之和

    for j in range(1, t):
        a = known_idx[j - 1]
        b = known_idx[j]
        e = seg_readings[b] - seg_readings[a]
        t_lo = max(lo_f[j], lo_g[j])
        t_hi = min(hi_f[j], hi_g[j])
        z_lo = t_lo - k_cur
        z_hi = t_hi - k_cur

        for i in range(a, b):
            lo_d = seg_min[i]
            hi_d = seg_max[i]
            if i == b - 1:
                rlo = rhi = 0
            else:
                rlo = suffix_min[i + 1] - suffix_min[b]
                rhi = suffix_max[i + 1] - suffix_max[b]

            # 由存在倍数 m*x 落在 [P+lo_d+rlo-e, P+hi_d+rhi-e] 求 x 范围。
            x_lo = _ceil_div(P + lo_d + rlo - e, m)
            x_hi = (P + hi_d + rhi - e) // m
            x_lo = max(x_lo, z_lo, 0)
            x_hi = min(x_hi, z_hi)

            chosen_d = None
            if x_lo <= x_hi:
                x = x_lo
                # d = m*x + e - P - R，R 取最大 rhi 时 d 最小，取最小 rlo 时最大。
                d_star = max(lo_d, m * x + e - P - rhi)
                if d_star <= min(hi_d, m * x + e - P - rlo):
                    chosen_d = d_star
                    chosen_x = x
            if chosen_d is None:
                # 理论上不可达：区间传播已证明整体可行。防御性处理。
                return Inconsistent(i + 1)

            cur += chosen_d
            increments.append(chosen_d)
            absolute.append(cur)
            P += chosen_d
            if i == b - 1:
                k_cur += chosen_x
                P = 0

    # ------------------------------------------------------------------
    # 4) 每位置累计回绕次数 wraps[i] = floor(A[i] / m)（A_i 非负）。
    # ------------------------------------------------------------------
    cumulative_wraps = [a // m for a in absolute]

    return Solved(
        absolute=absolute,
        increments=increments,
        cumulative_wraps=cumulative_wraps,
    )
