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

盘点锚点（``anchor``）
----------------------

锚点指定一个已有读数位置 p 和该位置经人工核准的非负绝对计数 C。若锚点在
旧表段，它直接固定旧表绝对计数；若在新表段，则由

    C = A[h] + B[p] - s

联合决定交接值与新表自身计数，不能把两段分别取无约束最优后再拼接。
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
class Anchor:
    """盘点锚点：``position`` 处已有读数，且其人工核准绝对计数为
    ``absolute_count``。"""

    position: int
    absolute_count: int


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
    anchor: Anchor | None = None,
) -> Result:
    if handoff is not None:
        return _solve_with_handoff(
            modulus, readings, min_steps, max_steps, handoff, anchor
        )
    if anchor is not None:
        return _solve_plain_with_anchor(
            modulus, readings, min_steps, max_steps, anchor
        )

    return _solve_segment(
        modulus, readings, min_steps, max_steps, readings[0]
    )


def _solve_plain_with_anchor(
    m: int,
    readings: list[int | None],
    min_steps: list[int],
    max_steps: list[int],
    anchor: Anchor,
) -> Result:
    """无交接但有锚点：在锚点处拆成前后两段。前段末端被锚点固定，后段从
    固定的锚点绝对值继续按原两级目标求解。"""
    p = anchor.position
    value = anchor.absolute_count

    first_value = value if p == 0 else readings[0]
    prefix = _solve_segment(
        m,
        readings[: p + 1],
        min_steps[:p],
        max_steps[:p],
        first_value,
        fixed_final_value=value,
    )
    if isinstance(prefix, Inconsistent):
        return prefix

    suffix = _solve_segment(
        m,
        readings[p:],
        min_steps[p:],
        max_steps[p:],
        value,
    )
    if isinstance(suffix, Inconsistent):
        return Inconsistent(p + suffix.position)

    absolute, increments = _join_segments(prefix, suffix)
    return Solved(
        absolute=absolute,
        increments=increments,
        cumulative_wraps=[a // m for a in absolute],
    )


def _solve_with_handoff(
    old_modulus: int,
    readings: list[int | None],
    min_steps: list[int],
    max_steps: list[int],
    handoff: Handoff,
    anchor: Anchor | None = None,
) -> Result:
    """旧表段 [0..h] 与新表段 [h..n-1] 在交接点共享物理累计值。"""
    n = len(readings)
    h = handoff.position
    m_new = handoff.new_modulus
    opening = handoff.opening_reading

    if anchor is None:
        old_res = _solve_segment(
            old_modulus,
            readings[: h + 1],
            min_steps[:h],
            max_steps[:h],
            readings[0],
        )
        if isinstance(old_res, Inconsistent):
            return old_res
        return _finish_handoff(
            readings,
            min_steps,
            max_steps,
            handoff,
            old_res,
        )

    p = anchor.position
    value = anchor.absolute_count

    if p <= h:
        # 锚点位于交接前或交接点：它固定旧表段中的一个值。若在交接点，这也
        # 是同一物理时刻的交接值；若在交接前，旧表后缀仍取最小交接值。
        old_first = value if p == 0 else readings[0]
        if p == h:
            old_res = _solve_segment(
                old_modulus,
                readings[: h + 1],
                min_steps[:h],
                max_steps[:h],
                old_first,
                fixed_final_value=value,
            )
            if isinstance(old_res, Inconsistent):
                return old_res
        else:
            prefix = _solve_segment(
                old_modulus,
                readings[: p + 1],
                min_steps[:p],
                max_steps[:p],
                old_first,
                fixed_final_value=value,
            )
            if isinstance(prefix, Inconsistent):
                return prefix
            suffix = _solve_segment(
                old_modulus,
                readings[p : h + 1],
                min_steps[p:h],
                max_steps[p:h],
                value,
            )
            if isinstance(suffix, Inconsistent):
                return Inconsistent(p + suffix.position)
            old_absolute, old_increments = _join_segments(prefix, suffix)
            old_res = Solved(
                absolute=old_absolute,
                increments=old_increments,
                cumulative_wraps=[a // old_modulus for a in old_absolute],
            )

        return _finish_handoff(
            readings,
            min_steps,
            max_steps,
            handoff,
            old_res,
        )

    # 锚点位于交接之后。设 r_h/r_p 为两侧读数，w/k 分别为交接值、新表锚点
    # 自身计数中的整圈数：
    #   A[h] = r_h + m0*w
    #   B[p] = r_p + m1*k
    #   C = A[h] + B[p] - opening
    # 因此 m0*w + m1*k = C-r_h-r_p+opening。
    r_h = readings[h]
    r_p = readings[p]

    old_bounds = _forward_bounds(
        old_modulus,
        readings[: h + 1],
        min_steps[:h],
        max_steps[:h],
        readings[0],
    )
    if isinstance(old_bounds, Inconsistent):
        return old_bounds
    _, old_lo, old_hi = old_bounds
    if h == 0:
        # 起点即换表时，旧表在交接前没有任何边；锚点可能反推出交接时旧表
        # 已有的真实累计数 r_h + m0*w（w 可取任意非负整数）。
        old_hi[-1] = None

    new_prefix_readings = [opening] + readings[h + 1 : p + 1]
    new_bounds = _forward_bounds(
        m_new,
        new_prefix_readings,
        min_steps[h:p],
        max_steps[h:p],
        opening,
    )
    if isinstance(new_bounds, Inconsistent):
        return Inconsistent(h + new_bounds.position)
    _, new_lo, new_hi = new_bounds

    pair = _smallest_nonnegative_linear_pair(
        old_modulus,
        m_new,
        value - r_h - r_p + opening,
        old_lo[-1],
        old_hi[-1],
        new_lo[-1],
        new_hi[-1],
    )
    if pair is None:
        return Inconsistent(p)
    w, k = pair

    join_value = r_h + old_modulus * w
    anchor_self_value = r_p + m_new * k

    if h == 0:
        # 起点即换表时没有旧表边，交接值由锚点联合反解得到，不能再把它
        # 固定成 readings[0] 对应的零圈读数。
        old_res = Solved(
            absolute=[join_value],
            increments=[],
            cumulative_wraps=[join_value // old_modulus],
        )
    else:
        old_res = _solve_segment(
            old_modulus,
            readings[: h + 1],
            min_steps[:h],
            max_steps[:h],
            readings[0],
            fixed_final_value=join_value,
        )
        if isinstance(old_res, Inconsistent):
            return old_res

    new_prefix_res = _solve_segment(
        m_new,
        new_prefix_readings,
        min_steps[h:p],
        max_steps[h:p],
        opening,
        fixed_final_value=anchor_self_value,
    )
    if isinstance(new_prefix_res, Inconsistent):
        return Inconsistent(h + new_prefix_res.position)

    new_suffix_res = _solve_segment(
        m_new,
        readings[p:],
        min_steps[p:],
        max_steps[p:],
        anchor_self_value,
    )
    if isinstance(new_suffix_res, Inconsistent):
        return Inconsistent(p + new_suffix_res.position)

    new_self_absolute, new_self_increments = _join_segments(
        new_prefix_res, new_suffix_res
    )
    shift = join_value - opening
    absolute = old_res.absolute + [
        b + shift for b in new_self_absolute[1:]
    ]
    increments = old_res.increments + new_self_increments

    old_wraps: list[int | None] = [
        a // old_modulus for a in old_res.absolute
    ] + [None] * (n - h - 1)
    new_wraps: list[int | None] = [None] * h + [
        b // m_new for b in new_self_absolute
    ]

    return Solved(
        absolute=absolute,
        increments=increments,
        cumulative_wraps=None,
        old_meter_wraps=old_wraps,
        new_meter_wraps=new_wraps,
    )


def _finish_handoff(
    readings: list[int | None],
    min_steps: list[int],
    max_steps: list[int],
    handoff: Handoff,
    old_res: Solved,
) -> Result:
    """给定已求好的旧表段，求无锚点约束的新表段并拼接。"""
    n = len(readings)
    h = handoff.position
    m_new = handoff.new_modulus
    opening = handoff.opening_reading

    new_res = _solve_segment(
        m_new,
        [opening] + readings[h + 1:],
        min_steps[h:],
        max_steps[h:],
        opening,
    )
    if isinstance(new_res, Inconsistent):
        return Inconsistent(h + new_res.position)

    join_value = old_res.absolute[-1]
    shift = join_value - opening
    absolute = old_res.absolute + [
        b + shift for b in new_res.absolute[1:]
    ]
    increments = old_res.increments + new_res.increments

    old_wraps: list[int | None] = list(old_res.cumulative_wraps) + [
        None
    ] * (n - h - 1)
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


def _join_segments(left: Solved, right: Solved) -> tuple[list[int], list[int]]:
    """拼接首尾值相同的两个相邻段结果。"""
    return (
        left.absolute + right.absolute[1:],
        left.increments + right.increments,
    )


def _smallest_nonnegative_linear_pair(
    a: int,
    b: int,
    c: int,
    w_lo: int,
    w_hi: int | None,
    k_lo: int,
    k_hi: int,
) -> tuple[int, int] | None:
    """求使最终绝对计数最小的 (w,k)；并列时取最小 w（即最小交接值）。"""
    g, x0, y0 = _extended_gcd(a, b)
    if c % g != 0:
        return None

    scale = c // g
    w0 = x0 * scale
    k0 = y0 * scale
    w_step = b // g
    k_step = a // g

    # w = w0 + w_step*t;  k = k0 - k_step*t。
    t_lo = max(
        _ceil_div(w_lo - w0, w_step),
        _ceil_div(k0 - k_hi, k_step),
    )
    upper_from_w = (
        (w_hi - w0) // w_step
        if w_hi is not None
        else None
    )
    t_hi = (k0 - k_lo) // k_step
    if upper_from_w is not None:
        t_hi = min(t_hi, upper_from_w)
    if t_lo > t_hi:
        return None

    w = w0 + w_step * t_lo
    k = k0 - k_step * t_lo
    return w, k


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """返回 (g, x, y)，其中 ``a*x + b*y = g = gcd(a,b)``。"""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_r, old_s, old_t


def _forward_bounds(
    m: int,
    seg_readings: list[int | None],
    seg_min: list[int],
    seg_max: list[int],
    first_value: int,
) -> tuple[list[int], list[int], list[int]] | Inconsistent:
    """前向传播所有已知点的可达回绕次数区间。"""
    n = len(seg_readings)
    first_r = seg_readings[0]
    if first_value < 0 or first_value % m != first_r:
        return Inconsistent(0)

    known_idx = [i for i, v in enumerate(seg_readings) if v is not None]
    t = len(known_idx)

    prefix_min = [0] * (n + 1)
    prefix_max = [0] * (n + 1)
    for i in range(n - 1):
        prefix_min[i + 1] = prefix_min[i] + seg_min[i]
        prefix_max[i + 1] = prefix_max[i] + seg_max[i]
    prefix_min[n] = prefix_min[n - 1]
    prefix_max[n] = prefix_max[n - 1]

    seg_qlo = [0] * t
    seg_qhi = [0] * t
    for j in range(1, t):
        a = known_idx[j - 1]
        b = known_idx[j]
        r_a = seg_readings[a]
        r_b = seg_readings[b]
        smin = prefix_min[b] - prefix_min[a]
        smax = prefix_max[b] - prefix_max[a]
        e = r_b - r_a
        q_lo = _ceil_div(smin - e, m)
        q_hi = (smax - e) // m
        if q_lo > q_hi:
            return Inconsistent(b)
        seg_qlo[j] = q_lo
        seg_qhi[j] = q_hi

    k0 = (first_value - first_r) // m
    lo_f = [0] * t
    hi_f = [0] * t
    lo_f[0] = hi_f[0] = k0
    for j in range(1, t):
        lo_f[j] = lo_f[j - 1] + seg_qlo[j]
        hi_f[j] = hi_f[j - 1] + seg_qhi[j]

    return known_idx, lo_f, hi_f


def _solve_segment(
    m: int,
    seg_readings: list[int | None],
    seg_min: list[int],
    seg_max: list[int],
    first_value: int,
    fixed_final_value: int | None = None,
) -> Result:
    """单子表段的两级最优恢复。``seg_readings[0]`` 必须已知，且首项绝对
    计数固定为 ``first_value``（满足同余）。``fixed_final_value`` 可把最后
    一个已知点固定为盘点锚点值。返回下标均为段内局部下标。"""
    n = len(seg_readings)
    bounds = _forward_bounds(
        m, seg_readings, seg_min, seg_max, first_value
    )
    if isinstance(bounds, Inconsistent):
        return bounds
    known_idx, lo_f, hi_f = bounds
    t = len(known_idx)

    final_index = known_idx[t - 1]
    final_residue = seg_readings[final_index]
    if fixed_final_value is None:
        final_k = lo_f[t - 1]
    else:
        if (
            fixed_final_value < 0
            or fixed_final_value % m != final_residue
            or not (lo_f[t - 1] <= fixed_final_value // m <= hi_f[t - 1])
        ):
            return Inconsistent(final_index)
        final_k = fixed_final_value // m

    lo_g = [0] * t
    hi_g = [0] * t
    lo_g[t - 1] = hi_g[t - 1] = final_k
    for j in range(t - 1, 0, -1):
        q_lo = lo_f[j] - lo_f[j - 1]
        q_hi = hi_f[j] - hi_f[j - 1]
        lo_g[j - 1] = lo_g[j] - q_hi
        hi_g[j - 1] = hi_g[j] - q_lo

    suffix_min = [0] * n
    suffix_max = [0] * n
    for i in range(n - 2, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + seg_min[i]
        suffix_max[i] = suffix_max[i + 1] + seg_max[i]

    absolute: list[int] = [first_value]
    increments: list[int] = []
    cur = first_value
    k_cur = (first_value - seg_readings[0]) // m
    P = 0

    for j in range(1, t):
        a = known_idx[j - 1]
        b = known_idx[j]
        e = seg_readings[b] - seg_readings[a]
        t_lo = max(lo_f[j], lo_g[j])
        t_hi = min(hi_f[j], hi_g[j])
        z_lo = t_lo - k_cur
        z_hi = t_hi - k_cur
        q_lo = lo_f[j] - lo_f[j - 1]
        q_hi = hi_f[j] - hi_f[j - 1]

        for i in range(a, b):
            lo_d = seg_min[i]
            hi_d = seg_max[i]
            if i == b - 1:
                rlo = rhi = 0
            else:
                rlo = suffix_min[i + 1] - suffix_min[b]
                rhi = suffix_max[i + 1] - suffix_max[b]

            x_lo = _ceil_div(P + lo_d + rlo - e, m)
            x_hi = (P + hi_d + rhi - e) // m
            x_lo = max(x_lo, z_lo, q_lo if i == b - 1 else 0)
            x_hi = min(x_hi, z_hi, q_hi if i == b - 1 else z_hi)

            chosen_d = None
            if x_lo <= x_hi:
                x = x_lo
                d_star = max(lo_d, m * x + e - P - rhi)
                if d_star <= min(hi_d, m * x + e - P - rlo):
                    chosen_d = d_star
                    chosen_x = x
            if chosen_d is None:
                return Inconsistent(i + 1)

            cur += chosen_d
            increments.append(chosen_d)
            absolute.append(cur)
            P += chosen_d
            if i == b - 1:
                k_cur += chosen_x
                P = 0

    cumulative_wraps = [a // m for a in absolute]
    return Solved(
        absolute=absolute,
        increments=increments,
        cumulative_wraps=cumulative_wraps,
    )
