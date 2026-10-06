"""盘点锚点的穷举、边界与交接联立测试。"""

import itertools
import random

from app.solver import Anchor, Handoff, Inconsistent, Solved, solve

from .brute import (
    brute_solve,
    brute_solve_handoff,
    earliest_blocked,
    earliest_blocked_handoff,
)


def _anchor_values(m, readings, min_steps, max_steps, position):
    """构造小规模确定性锚点候选：覆盖可达圈数、相邻圈数和异余数。"""
    reading = readings[position]
    total_hi = sum(max_steps)
    max_wrap = total_hi // m + 2
    values = {0, reading, reading + m, total_hi + m + 1}
    for wrap in range(max_wrap + 1):
        values.add(reading + wrap * m)
        if wrap:
            values.add(reading + (wrap - 1) * m + 1)
            values.add(reading + wrap * m - 1)
    for residue in range(min(m, 4)):
        if residue != reading:
            values.add(residue)
    return sorted(v for v in values if v >= 0)


def _check_plain_anchor(m, readings, min_steps, max_steps, position, count):
    anchor = Anchor(position, count)
    res = solve(m, list(readings), list(min_steps), list(max_steps), None, anchor)
    ref = brute_solve(
        m, readings, min_steps, max_steps, (position, count)
    )
    if ref[0] == "inconsistent":
        assert isinstance(res, Inconsistent), (
            m, readings, min_steps, max_steps, position, count, res
        )
        assert res.position == ref[1]
        assert res.position == earliest_blocked(
            m, readings, min_steps, max_steps, (position, count)
        )
        return

    assert isinstance(res, Solved)
    assert res.absolute == ref[1]
    assert res.absolute[position] == count
    assert len(res.absolute) == len(readings)
    assert len(res.increments) == len(readings) - 1
    A = [readings[0]]
    if position == 0:
        A = [count]
    for i, d in enumerate(res.increments):
        A.append(A[-1] + d)
    assert A == res.absolute
    for i, d in enumerate(res.increments):
        assert min_steps[i] <= d <= max_steps[i]
    for i, v in enumerate(readings):
        if v is not None:
            assert res.absolute[i] % m == v


def test_anchor_exhaustive_small():
    count = 0
    for m in range(1, 4):
        step_modes = [
            (lo, hi)
            for lo in range(0, 3)
            for hi in range(lo, min(2 * m, lo + 2) + 1)
        ]
        for n in range(2, 6):
            inner_choices = list(range(m)) + [None]
            all_step_combos = list(itertools.product(step_modes, repeat=n - 1))
            rng_s = random.Random(hash((m, n, "anchor")) & 0xFFFFFFFF)
            step_combos = (
                all_step_combos
                if len(all_step_combos) <= 10
                else rng_s.sample(all_step_combos, 10)
            )
            for inner in itertools.product(inner_choices, repeat=n - 2):
                for r0 in range(m):
                    for rn in range(m):
                        readings = [r0] + list(inner) + [rn]
                        known_positions = [
                            i for i, v in enumerate(readings) if v is not None
                        ]
                        for combo in step_combos:
                            min_steps = [c[0] for c in combo]
                            max_steps = [c[1] for c in combo]
                            for position in known_positions:
                                for value in _anchor_values(
                                    m,
                                    readings,
                                    min_steps,
                                    max_steps,
                                    position,
                                ):
                                    _check_plain_anchor(
                                        m,
                                        readings,
                                        min_steps,
                                        max_steps,
                                        position,
                                        value,
                                    )
                                    count += 1
    assert count > 15_000


def test_anchor_random_medium():
    rng = random.Random(20261006)
    for _ in range(500):
        m = rng.randrange(1, 8)
        n = rng.randrange(2, 9)
        readings = [None] * n
        readings[0] = rng.randrange(m)
        readings[-1] = rng.randrange(m)
        for i in range(1, n - 1):
            readings[i] = None if rng.random() < 0.45 else rng.randrange(m)
        min_steps, max_steps = [], []
        for _ in range(n - 1):
            lo = rng.randrange(0, 2 * m + 1)
            hi = lo + rng.randrange(0, 2 * m - lo + 1)
            min_steps.append(lo)
            max_steps.append(hi)
        position = rng.choice(
            [i for i, v in enumerate(readings) if v is not None]
        )
        if rng.random() < 0.8:
            count = readings[position] + m * rng.randrange(
                0, sum(max_steps) // m + 3
            )
        else:
            count = rng.randrange(0, sum(max_steps) + 2 * m + 1)
        _check_plain_anchor(
            m, readings, min_steps, max_steps, position, count
        )


def test_anchor_forces_higher_wrap_and_lex_order():
    # 无锚点时末值最小为 12；锚定 22 迫使多走一圈。前缀仍取字典序最小。
    res = solve(
        10,
        [8, None, 2],
        [0, 0],
        [20, 20],
        None,
        Anchor(2, 22),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [8, 8, 22]
    assert res.increments == [0, 14]
    assert res.cumulative_wraps == [0, 0, 2]


def test_anchor_residue_mismatch_inconsistent_at_anchor():
    res = solve(10, [8, None, 2], [0, 0], [20, 20], None, Anchor(2, 23))
    assert isinstance(res, Inconsistent)
    assert res.position == 2


def test_anchor_at_start_conflicts_with_empty_warehouse_start():
    res = solve(10, [8, 2], [0], [20], None, Anchor(0, 18))
    assert isinstance(res, Inconsistent)
    assert res.position == 0


def test_anchor_final_min_after_anchor():
    # 锚点固定 A[1]=18；之后 8 -> 2 的最小可行末值是 22，而不是 32。
    res = solve(
        10,
        [8, 8, None, 2],
        [10, 0, 0],
        [10, 20, 20],
        None,
        Anchor(1, 18),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [8, 18, 18, 22]


def test_anchor_holes_multi_wrap():
    m = 5
    res = solve(
        m,
        [2, None, None, 3],
        [4, 4, 4],
        [10, 10, 10],
        None,
        Anchor(3, 23),
    )
    assert isinstance(res, Solved)
    assert res.absolute[3] == 23
    assert res.increments == [4, 7, 10]


# ---------------------------------------------------------------------------
# 锚点与换表交接：交接前、交接点、交接后都必须和真实轨迹耦合。
# ---------------------------------------------------------------------------


def _check_handoff_anchor(
    m_old, readings, min_steps, max_steps, h, m_new, opening, position, count
):
    res = solve(
        m_old,
        list(readings),
        list(min_steps),
        list(max_steps),
        Handoff(h, m_new, opening),
        Anchor(position, count),
    )
    ref = brute_solve_handoff(
        m_old,
        readings,
        min_steps,
        max_steps,
        h,
        m_new,
        opening,
        (position, count),
    )
    if ref[0] == "inconsistent":
        assert isinstance(res, Inconsistent), (
            m_old, m_new, h, readings, min_steps, max_steps,
            opening, position, count, res,
        )
        assert res.position == ref[1]
        assert res.position == earliest_blocked_handoff(
            m_old,
            readings,
            min_steps,
            max_steps,
            h,
            m_new,
            opening,
            (position, count),
        )
        return

    assert isinstance(res, Solved)
    _, ref_abs, ref_old_wraps, ref_new_wraps = ref
    assert res.absolute == ref_abs
    assert res.absolute[position] == count
    assert res.old_meter_wraps == ref_old_wraps
    assert res.new_meter_wraps == ref_new_wraps


def _handoff_anchor_candidates(
    m_old, m_new, readings, min_steps, max_steps, h, opening, position
):
    total_hi = sum(max_steps)
    values = {0, total_hi + m_old + m_new + 1}
    for x in range(total_hi + m_old + m_new + 2):
        if position <= h:
            if x % m_old == readings[position]:
                values.add(x)
        else:
            # C = X + B_p - opening；枚举旧交接真值和新表自身计数的小组合。
            x_old = readings[h] + m_old * (x % (m_old + 1))
            b = readings[position] + m_new * ((x // (m_old + 1)) % (m_new + 1))
            values.add(x_old + b - opening)
    # 额外异余数候选，确保盘点值不是仅由表面读数决定。
    for residue in range(min(6, m_old + m_new)):
        values.add(residue)
    return sorted(v for v in values if 0 <= v <= total_hi + m_old + m_new + 1)


def test_handoff_anchor_exhaustive_small():
    count = 0
    for m_old in range(1, 3):
        for m_new in range(1, 3):
            n = 4
            for h in range(n):
                side_m = [
                    m_old if i <= h else m_new for i in range(n)
                ]
                pools = []
                for i in range(n):
                    if i in (0, h, n - 1):
                        pools.append(list(range(side_m[i])))
                    else:
                        pools.append(list(range(side_m[i])) + [None])
                edge_modes = [
                    [
                        (lo, hi)
                        for lo in range(0, 3)
                        for hi in range(lo, min(2 * side_m[i], lo + 2) + 1)
                    ]
                    for i in range(n - 1)
                ]
                all_step_combos = list(itertools.product(*edge_modes))
                rng_s = random.Random(
                    hash((m_old, m_new, n, h, "anchor")) & 0xFFFFFFFF
                )
                step_combos = (
                    all_step_combos
                    if len(all_step_combos) <= 40
                    else rng_s.sample(all_step_combos, 40)
                )
                for reading_combo in itertools.product(*pools):
                    readings = list(reading_combo)
                    for opening in range(m_new):
                        for combo in step_combos:
                            min_steps = [c[0] for c in combo]
                            max_steps = [c[1] for c in combo]
                            for position in [
                                i for i, v in enumerate(readings)
                                if v is not None
                            ]:
                                for value in _handoff_anchor_candidates(
                                    m_old,
                                    m_new,
                                    readings,
                                    min_steps,
                                    max_steps,
                                    h,
                                    opening,
                                    position,
                                ):
                                    _check_handoff_anchor(
                                        m_old,
                                        readings,
                                        min_steps,
                                        max_steps,
                                        h,
                                        m_new,
                                        opening,
                                        position,
                                        value,
                                    )
                                    count += 1
    assert count > 10_000


def test_handoff_anchor_random_medium():
    rng = random.Random(20261007)
    for _ in range(500):
        m_old = rng.randrange(1, 7)
        m_new = rng.randrange(1, 7)
        n = rng.randrange(2, 9)
        h = rng.randrange(n)
        opening = rng.randrange(m_new)
        readings = [None] * n
        for i in range(n):
            side_m = m_old if i <= h else m_new
            if i in (0, h, n - 1) or rng.random() < 0.6:
                readings[i] = rng.randrange(side_m)
        min_steps, max_steps = [], []
        for i in range(n - 1):
            side_m = m_old if i < h else m_new
            lo = rng.randrange(0, 2 * side_m + 1)
            hi = lo + rng.randrange(0, 2 * side_m - lo + 1)
            min_steps.append(lo)
            max_steps.append(hi)
        position = rng.choice(
            [i for i, v in enumerate(readings) if v is not None]
        )
        count = rng.randrange(0, sum(max_steps) + m_old + m_new + 2)
        _check_handoff_anchor(
            m_old,
            readings,
            min_steps,
            max_steps,
            h,
            m_new,
            opening,
            position,
            count,
        )


def test_anchor_before_handoff_affects_join_value():
    # 锚定旧表 A[0]=8（默认值）不改变基础实例；改为可达更高圈数会联动新表。
    res = solve(
        10,
        [8, 8, 30],
        [10, 0],
        [10, 60],
        Handoff(1, 100, 0),
        Anchor(0, 8),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [8, 18, 48]
    assert res.old_meter_wraps == [0, 1, None]


def test_anchor_at_handoff_checks_same_physical_moment():
    good = solve(
        10,
        [8, 2, 30],
        [0, 0],
        [6, 60],
        Handoff(1, 100, 0),
        Anchor(1, 12),
    )
    assert isinstance(good, Solved)
    assert good.absolute == [8, 12, 42]

    bad = solve(
        10,
        [8, 2, 30],
        [0, 0],
        [6, 60],
        Handoff(1, 100, 0),
        Anchor(1, 22),
    )
    assert isinstance(bad, Inconsistent)
    assert bad.position == 1


def test_anchor_after_handoff_couples_old_and_new_meters():
    # 交接真值可取 12 或 22...；锚点 C=52 可用 X=22、B=30 或 X=12、B=40。
    # 旧表优先字典序/最小交接值，选择 X=12，新表 B=40（读数 40，回绕 0）。
    res = solve(
        10,
        [8, 2, 40],
        [0, 0],
        [20, 100],
        Handoff(1, 100, 0),
        Anchor(2, 52),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [8, 12, 52]
    assert res.old_meter_wraps == [0, 1, None]
    assert res.new_meter_wraps == [None, 0, 0]

    forced = solve(
        10,
        [8, 2, 40],
        [0, 0],
        [20, 100],
        Handoff(1, 100, 0),
        Anchor(2, 62),
    )
    assert isinstance(forced, Solved)
    # 62 = 22 + 40；旧表交接真值必须抬高到 22，不能把两段独立最优拼起来。
    assert forced.absolute == [8, 22, 62]
    assert forced.old_meter_wraps == [0, 2, None]
    assert forced.new_meter_wraps == [None, 0, 0]


def test_anchor_after_handoff_with_new_meter_wraps():
    res = solve(
        10,
        [8, 2, 5],
        [0, 0],
        [20, 100],
        Handoff(1, 10, 0),
        Anchor(2, 27),
    )
    assert isinstance(res, Solved)
    # X=12，B=15（新表读数 5、回绕 1）⇒ C=27。
    assert res.absolute == [8, 12, 27]
    assert res.new_meter_wraps == [None, 0, 1]


def test_anchor_after_handoff_large_coprime_moduli():
    # 大整数且模数互素时，用线性丢番图方程选择共同决定锚点的旧交接真值
    # 与新表自身计数，不枚举任何绝对计数值。
    m_old = 10**30 + 1
    m_new = 10**30 + 3
    c = m_old * 7 + m_new * 11
    res = solve(
        m_old,
        [0, 0, 0],
        [0, 0],
        [8 * m_old, 12 * m_new],
        Handoff(1, m_new, 0),
        Anchor(2, c),
    )
    assert isinstance(res, Solved)
    assert res.absolute[1] == 7 * m_old
    assert res.absolute[2] == c
    assert res.old_meter_wraps == [0, 7, None]
    assert res.new_meter_wraps == [None, 0, 11]
