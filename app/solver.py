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

两段在无盘点锚点时除共享物理时点 ``A[h]`` 外无耦合，可分别求解；若锚点
位于交接之后，则锚点同时约束旧表交接真值与新表自身计数，必须联立求解，
不能把两段各自最优后直接拼接。

盘点锚点（``anchor``）
----------------------

请求可在某个**已有读数**的位置给定人工核准的非负绝对计数。该点成为已知
回绕次数的单点约束；锚点前后的可达区间会共同限制该点，锚点之后再最小化
终值。锚点位于换表之后时，求解的是

    K_old * m_old + w * m_new = 常数

的非负整数解，以保证旧表交接真值与新表自身计数共同落在锚点上。
"""

from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class Anchor:
    """人工盘点锚点：位置 ``position`` 的真实绝对计数为 ``absolute_count``。

    ``position`` 必须指向已有读数；该处读数是否与给定绝对计数同余由求解器
    判断，不一致属于合法请求但无可行轨迹。
    """

    position: int
    absolute_count: int


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


def _smallest_congruent_k(a: int, b: int, modulus: int, lower: int) -> int | None:
    """满足 ``a*k ≡ b (mod modulus)`` 且 ``k >= lower`` 的最小非负 k。

    无解时返回 ``None``。``modulus`` 可为 1；``a``、``lower`` 均非负。
    """
    g = gcd(a, modulus)
    if b % g != 0:
        return None
    reduced_mod = modulus // g
    if reduced_mod == 1:
        k = 0
    else:
        reduced_a = (a // g) % reduced_mod
        reduced_b = (b // g) % reduced_mod
        k = (reduced_b * pow(reduced_a, -1, reduced_mod)) % reduced_mod
    if k < lower:
        k += _ceil_div(lower - k, reduced_mod) * reduced_mod
    return k


def solve(
    modulus: int,
    readings: list[int | None],
    min_steps: list[int],
    max_steps: list[int],
    handoff: Handoff | None = None,
    anchor: Anchor | None = None,
) -> Result:
    if handoff is not None:
        return _solve_with_handoff(
            modulus, readings, min_steps, max_steps, handoff, anchor
        )

    return _solve_segment(
        modulus,
        readings,
        min_steps,
        max_steps,
        readings[0],
        anchor,
    )


def _segment_q_intervals(
    m: int,
    known_idx: list[int],
    readings: list[int | None],
    prefix_min: list[int],
    prefix_max: list[int],
) -> tuple[list[int], list[int], int | None]:
    """计算相邻已知点之间的回绕增量区间，返回 ``(lo, hi, first_bad)``。"""
    t = len(known_idx)
    q_lo = [0] * t
    q_hi = [0] * t
    for j in range(1, t):
        a = known_idx[j - 1]
        b = known_idx[j]
        e = readings[b] - readings[a]
        smin = prefix_min[b] - prefix_min[a]
        smax = prefix_max[b] - prefix_max[a]
        q_lo[j] = _ceil_div(smin - e, m)
        q_hi[j] = (smax - e) // m
        if q_lo[j] > q_hi[j]:
            return q_lo, q_hi, b
    return q_lo, q_hi, None


def _solve_with_handoff(
    old_modulus: int,
    readings: list[int | None],
    min_steps: list[int],
    max_steps: list[int],
    handoff: Handoff,
    anchor: Anchor | None = None,
) -> Result:
    """求解含一次换表的真实轨迹，必要时用盘点锚点联立两段。"""
    n = len(readings)
    if anchor is not None:
        if not (0 <= anchor.position < n) or readings[anchor.position] is None:
            position = anchor.position
            return Inconsistent(position if 0 <= position < n else n)

    h = handoff.position
    m_new = handoff.new_modulus
    opening = handoff.opening_reading

    # 锚点在交接之前或恰在交接点：它固定旧表轨迹（含交接真值）。锚点后
    # 的旧表后缀仍取最小交接真值，从而新表段也能继续得到最小全局终值。
    if anchor is not None and anchor.position <= h:
        old_anchor = anchor
    else:
        old_anchor = None

    old_res = _solve_segment(
        old_modulus,
        readings[: h + 1],
        min_steps[:h],
        max_steps[:h],
        readings[0],
        old_anchor,
    )
    if isinstance(old_res, Inconsistent):
        return old_res
    assert old_res.old_meter_wraps is None and old_res.new_meter_wraps is None

    # 无锚点或锚点在交接点时，旧表段求解已经固定同一物理时刻；新表从
    # opening 起步，无需跨段拼接后再二次检查锚点。
    if anchor is None or anchor.position <= h:
        new_res = _solve_segment(
            m_new,
            [opening] + readings[h + 1:],
            min_steps[h:],
            max_steps[h:],
            opening,
            None,
        )
        if isinstance(new_res, Inconsistent):
            return Inconsistent(new_res.position + h)
        return _join_handoff(
            n,
            h,
            old_modulus,
            m_new,
            opening,
            old_res,
            new_res,
        )

    p = anchor.position

    # 锚点位于交接之后。旧表交接真值 X 和新表自身计数 B_p 共同决定锚点：
    #   X = r_h + K*m_old
    #   B_p = r_p + w*m_new
    #   C = X + B_p - opening
    # 因而 K*m_old + w*m_new = C + opening - r_h - r_p。
    if p > h:
        old_n = h + 1
        old_known = [i for i, v in enumerate(readings[: h + 1]) if v is not None]
        old_pref_min = [0] * (old_n + 1)
        old_pref_max = [0] * (old_n + 1)
        for i in range(h):
            old_pref_min[i + 1] = old_pref_min[i] + min_steps[i]
            old_pref_max[i + 1] = old_pref_max[i] + max_steps[i]
        old_pref_min[old_n] = old_pref_min[old_n - 1]
        old_pref_max[old_n] = old_pref_max[old_n - 1]
        old_qlo, old_qhi, bad = _segment_q_intervals(
            old_modulus, old_known, readings, old_pref_min, old_pref_max
        )
        if bad is not None:
            return Inconsistent(bad)

        new_readings = [opening] + readings[h + 1:]
        local_p = p - h
        new_n = len(new_readings)
        new_pref_min = [0] * (new_n + 1)
        new_pref_max = [0] * (new_n + 1)
        for i in range(new_n - 1):
            old_i = h + i
            new_pref_min[i + 1] = new_pref_min[i] + min_steps[old_i]
            new_pref_max[i + 1] = new_pref_max[i] + max_steps[old_i]
        new_pref_min[new_n] = new_pref_min[new_n - 1]
        new_pref_max[new_n] = new_pref_max[new_n - 1]
        new_known = [i for i, v in enumerate(new_readings) if v is not None]
        new_qlo, new_qhi, bad = _segment_q_intervals(
            m_new, new_known, new_readings, new_pref_min, new_pref_max
        )
        # 锚点之前的新表已知段若已不可能，应先报告该已知点，而不是锚点。
        if bad is not None and bad <= local_p:
            return Inconsistent(bad + h)

        old_lo = sum(old_qlo)
        old_hi = sum(old_qhi)
        # 找到 local_p 所属的已知点编号，并求到该点的 w 可行区间。
        anchor_j = next(j for j, idx in enumerate(new_known) if idx == local_p)
        w_lo = sum(new_qlo[1 : anchor_j + 1])
        w_hi = sum(new_qhi[1 : anchor_j + 1])

        r_h = readings[h]
        r_p = readings[p]
        target = anchor.absolute_count + opening - r_h - r_p
        if target < 0 or w_lo < 0 or w_lo > w_hi:
            return Inconsistent(p)

        # w 最大时允许的 K 最小；取满足等式和区间的最小 K。旧段在更小交接
        # 真值下的字典序轨迹不会更靠后，故最小 K 对应整条轨迹字典序最小。
        k_min = max(0, old_lo, _ceil_div(target - w_hi * m_new, old_modulus))
        k_max = min(old_hi, (target - w_lo * m_new) // old_modulus)
        if k_min > k_max:
            return Inconsistent(p)

        k = _smallest_congruent_k(old_modulus, target, m_new, k_min)
        if k is None or k > k_max:
            return Inconsistent(p)
        w = (target - k * old_modulus) // m_new
        if not (w_lo <= w <= w_hi):
            return Inconsistent(p)

        join_value = r_h + k * old_modulus
        b_anchor = r_p + w * m_new
        old_res = _solve_segment(
            old_modulus,
            readings[: h + 1],
            min_steps[:h],
            max_steps[:h],
            readings[0],
            Anchor(h, join_value),
        )
        if isinstance(old_res, Inconsistent):
            return old_res
        new_res = _solve_segment(
            m_new,
            new_readings,
            min_steps[h:],
            max_steps[h:],
            opening,
            Anchor(local_p, b_anchor),
        )
        if isinstance(new_res, Inconsistent):
            return Inconsistent(new_res.position + h)
        return _join_handoff(
            n,
            h,
            old_modulus,
            m_new,
            opening,
            old_res,
            new_res,
        )

    raise AssertionError("unhandled anchor/handoff position")


def _join_handoff(
    n: int,
    h: int,
    old_modulus: int,
    new_modulus: int,
    opening: int,
    old_res: Solved,
    new_res: Solved,
) -> Solved:
    """把已分别满足共同交接真值的两段子轨迹拼成全局响应。"""
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
    anchor: Anchor | None = None,
) -> Result:
    """单子表段的两级最优恢复。

    ``seg_readings[0]`` 必须已知，且首项绝对计数固定为 ``first_value``。
    若给定段内锚点，则该点绝对计数同时固定；锚点前在满足该终点的约束下取
    字典序最小，锚点后继续执行“终值最小、序列字典序最小”。返回的下标均为
    段内局部下标（从 0 起）。
    """
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

    seg_qlo, seg_qhi, first_bad = _segment_q_intervals(
        m, known_idx, seg_readings, prefix_min, prefix_max
    )

    if anchor is not None:
        p = anchor.position
        if not (0 <= p < n) or seg_readings[p] is None:
            return Inconsistent(p if 0 <= p < n else n)
        if p not in known_idx:
            return Inconsistent(p)

        # 与锚点无关的更早结构性无解必须先报；锚点后的结构性无解留给后缀。
        if first_bad is not None and first_bad <= p:
            return Inconsistent(first_bad)
        if anchor.absolute_count < 0:
            return Inconsistent(p)
        pin_residue = seg_readings[p]
        pin_k, remainder = divmod(
            anchor.absolute_count - pin_residue, m
        )
        if remainder != 0 or pin_k < 0:
            return Inconsistent(p)

        # 内部锚点把序列拆成两个更简单的问题：前缀固定终点，后缀从锚点
        # 重新按原两级目标求解。二者拼接即得到锚点约束下的全局最优。
        if p < n - 1:
            prefix_res = _solve_segment(
                m,
                seg_readings[: p + 1],
                seg_min[:p],
                seg_max[:p],
                first_value,
                Anchor(p, anchor.absolute_count),
            )
            if isinstance(prefix_res, Inconsistent):
                return prefix_res
            suffix_res = _solve_segment(
                m,
                seg_readings[p:],
                seg_min[p:],
                seg_max[p:],
                anchor.absolute_count,
                None,
            )
            if isinstance(suffix_res, Inconsistent):
                return Inconsistent(suffix_res.position + p)
            return Solved(
                absolute=prefix_res.absolute + suffix_res.absolute[1:],
                increments=prefix_res.increments + suffix_res.increments,
                cumulative_wraps=prefix_res.cumulative_wraps
                + suffix_res.cumulative_wraps[1:],
            )

        target_k = pin_k
    else:
        if first_bad is not None:
            return Inconsistent(first_bad)
        target_k = None

    # 首项绝对计数 = 余数 + k0*m；旧表段与新开表段均有 k0=0。
    k0 = (first_value - seg_readings[0]) // m

    # ------------------------------------------------------------------
    # 1) 前向区间传播：F_j 为从固定起点可达的 k_j 整数区间。
    # ------------------------------------------------------------------
    lo_f = [0] * t
    hi_f = [0] * t
    lo_f[0] = hi_f[0] = k0
    for j in range(1, t):
        lo_f[j] = lo_f[j - 1] + seg_qlo[j]
        hi_f[j] = hi_f[j - 1] + seg_qhi[j]

    if target_k is None:
        # 最小最终绝对计数 ⇔ 最小 k_{t-1} ⇔ loF_{t-1}。
        final_k = lo_f[t - 1]
    else:
        final_k = target_k
        if not (lo_f[t - 1] <= final_k <= hi_f[t - 1]):
            return Inconsistent(known_idx[t - 1])

    # ------------------------------------------------------------------
    # 2) 后向传播：G_j 由固定的最终回绕次数倒推；贪心允许集合 T_j=F_j∩G_j。
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
