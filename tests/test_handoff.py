"""换表交接（handoff）的 solver 对拍与边界测试。

穷举参考实现见 tests/brute.py 的 brute_solve_handoff：直接 DFS 枚举所有
合法增量组合，在“先最小化最终绝对计数、再取整条轨迹字典序最小”的两级
目标下求标准答案，并给出两台表各自的回绕次数与最早不可延伸位置。
"""

import itertools
import random

from app.solver import Handoff, Inconsistent, Solved, solve

from .brute import brute_solve_handoff, earliest_blocked_handoff


def check_handoff(modulus, readings, min_steps, max_steps, position,
                  new_modulus, opening):
    """跑高效实现并完整校验返回结构（含两台表各自的回绕次数）。"""
    res = solve(
        modulus, list(readings), list(min_steps), list(max_steps),
        Handoff(position, new_modulus, opening),
    )
    n = len(readings)
    h = position
    if isinstance(res, Inconsistent):
        # 与穷举的最早不可延伸位置一致。
        assert res.position == earliest_blocked_handoff(
            modulus, readings, min_steps, max_steps, h, new_modulus, opening
        )
        return res
    assert isinstance(res, Solved)
    # 换表响应不再给单表 cumulative_wraps。
    assert res.cumulative_wraps is None
    assert res.old_meter_wraps is not None and res.new_meter_wraps is not None
    assert len(res.absolute) == n
    assert len(res.increments) == n - 1
    assert len(res.old_meter_wraps) == n
    assert len(res.new_meter_wraps) == n
    # 重建一致性。
    A = [readings[0]]
    for d in res.increments:
        A.append(A[-1] + d)
    assert A == res.absolute
    # 步长区间逐段约束真实耗用（交接点两侧都检查）。
    for i, d in enumerate(res.increments):
        assert min_steps[i] <= d <= max_steps[i]
    # 交接动作本身耗用为零：没有额外增量，轨迹在交接点连续。
    assert all(a >= 0 for a in A)
    assert all(A[i] <= A[i + 1] for i in range(n - 1))
    # 已知点按所在侧的模数匹配读数。
    for i, v in enumerate(readings):
        if v is None:
            continue
        if i <= h:
            assert A[i] % modulus == v
        else:
            assert (A[i] - A[h] + opening) % new_modulus == v
    # 两台表分别的回绕次数；非本表区间为 None。
    for i in range(n):
        if i <= h:
            assert res.old_meter_wraps[i] == A[i] // modulus
        else:
            assert res.old_meter_wraps[i] is None
        if i < h:
            assert res.new_meter_wraps[i] is None
        else:
            assert res.new_meter_wraps[i] == (
                (A[i] - A[h] + opening) // new_modulus
            )
    # 新表在交接点恰好开表：回绕数从 0 计起。
    assert res.new_meter_wraps[h] == opening // new_modulus == 0
    return res


def assert_matches_brute(modulus, readings, min_steps, max_steps, position,
                         new_modulus, opening):
    res = check_handoff(
        modulus, readings, min_steps, max_steps, position,
        new_modulus, opening,
    )
    ref = brute_solve_handoff(
        modulus, readings, min_steps, max_steps, position,
        new_modulus, opening,
    )
    if ref[0] == "inconsistent":
        assert isinstance(res, Inconsistent)
        assert res.position == ref[1]
    else:
        assert isinstance(res, Solved)
        _, abs_ref, old_wraps_ref, new_wraps_ref = ref
        assert res.absolute == abs_ref, (
            f"m={modulus}->{new_modulus} h={position} open={opening} "
            f"readings={readings} steps={list(zip(min_steps, max_steps))}"
        )
        assert res.old_meter_wraps == old_wraps_ref
        assert res.new_meter_wraps == new_wraps_ref


# ---------------------------------------------------------------------------
# 穷举对拍：枚举所有（旧模数, 新模数, n, 交接位置, 读数, 开表读数, 步长）
# 的小实例，覆盖两侧漏抄、多圈回绕与交接边界（h=0 与 h=n-1）。
# ---------------------------------------------------------------------------

def _step_modes(m):
    return [
        (lo, hi)
        for lo in range(0, 3)
        for hi in range(lo, min(2 * m, lo + 2) + 1)
    ]


def test_handoff_exhaustive_small():
    count = 0
    for m_old in range(1, 4):
        for m_new in range(1, 4):
            for n in range(2, 5):
                for h in range(n):
                    # 逐位置的模数与读数取值池。
                    side_m = [
                        m_old if i <= h else m_new for i in range(n)
                    ]
                    pools = []
                    for i in range(n):
                        if i in (0, h, n - 1):
                            pools.append(list(range(side_m[i])))
                        else:
                            pools.append(list(range(side_m[i])) + [None])
                    # 步长模式逐边生成（边上界取决于所属侧的模数）。
                    edge_modes = [
                        _step_modes(m_old if i < h else m_new)
                        for i in range(n - 1)
                    ]
                    all_combos = list(itertools.product(*edge_modes))
                    if len(all_combos) > 24:
                        rng_s = random.Random(
                            hash((m_old, m_new, n, h)) & 0xFFFFFFFF
                        )
                        combos = rng_s.sample(all_combos, 24)
                    else:
                        combos = all_combos
                    for reading_combo in itertools.product(*pools):
                        readings = list(reading_combo)
                        for opening in range(m_new):
                            for combo in combos:
                                min_steps = [c[0] for c in combo]
                                max_steps = [c[1] for c in combo]
                                assert_matches_brute(
                                    m_old, readings, min_steps, max_steps,
                                    h, m_new, opening,
                                )
                                count += 1
    assert count > 60_000


# ---------------------------------------------------------------------------
# 随机对拍（稍大范围）。
# ---------------------------------------------------------------------------

def test_handoff_random_medium():
    rng = random.Random(20260927)
    for _ in range(400):
        m_old = rng.randrange(1, 7)
        m_new = rng.randrange(1, 7)
        n = rng.randrange(2, 8)
        h = rng.randrange(n)
        opening = rng.randrange(m_new)
        readings = [None] * n
        for i in range(n):
            side_m = m_old if i <= h else m_new
            if i in (0, h, n - 1) or rng.random() < 0.5:
                readings[i] = rng.randrange(side_m)
        min_steps, max_steps = [], []
        for i in range(n - 1):
            side_m = m_old if i < h else m_new
            lo = rng.randrange(0, min(4, 2 * side_m + 1))
            hi = lo + rng.randrange(0, min(4, 2 * side_m - lo + 1))
            min_steps.append(lo)
            max_steps.append(hi)
        assert_matches_brute(
            m_old, readings, min_steps, max_steps, h, m_new, opening
        )


# ---------------------------------------------------------------------------
# 手工场景。
# ---------------------------------------------------------------------------

def test_handoff_basic_join():
    # 旧表 m=10：8 -> 2（回绕一次，A=12）；新表 m=100 开表 0，读到 30。
    res = solve(10, [8, 2, 30], [0, 0], [6, 60], Handoff(1, 100, 0))
    assert isinstance(res, Solved)
    assert res.absolute == [8, 12, 42]
    assert res.increments == [4, 30]
    assert res.cumulative_wraps is None
    assert res.old_meter_wraps == [0, 1, None]
    assert res.new_meter_wraps == [None, 0, 0]


def test_handoff_zero_reset_is_not_negative_consumption():
    # 旧表高位 9 换新表开表 0：真实耗用不降，新表清零不产生负耗用。
    res = solve(10, [8, 9, 0], [1, 0], [1, 0], Handoff(1, 100, 0))
    assert isinstance(res, Solved)
    assert res.absolute == [8, 9, 9]
    assert res.increments == [1, 0]
    assert all(d >= 0 for d in res.increments)
    assert res.new_meter_wraps == [None, 0, 0]


def test_handoff_position_zero():
    # 交接在首位：旧表只有最后一次读数，其余全按新表。
    res = solve(10, [3, 1], [0], [20], Handoff(0, 10, 3))
    assert isinstance(res, Solved)
    assert res.absolute == [3, 11]
    assert res.old_meter_wraps == [0, None]
    assert res.new_meter_wraps == [0, 1]


def test_handoff_position_last():
    # 交接在末位：新表只有开表读数，其余全按旧表。
    res = solve(10, [8, 2], [0], [6], Handoff(1, 100, 0))
    assert isinstance(res, Solved)
    assert res.absolute == [8, 12]
    assert res.old_meter_wraps == [0, 1]
    assert res.new_meter_wraps == [None, 0]


def test_handoff_old_side_multi_wrap():
    # 旧表侧连续多圈：m=5，2 -> 3，三步各 [4,10]，总量须为 1+5q ∈ [12,30]
    # 最小 q=3 ⇒ 总量 16，字典序最小 d=[4,4,8]（交接点回绕 3 次）；
    # 交接后新表 m=7 开表 1 再走到 4（d=3）。
    res = solve(5, [2, None, None, 3, 4], [4, 4, 4, 0], [10, 10, 10, 6],
                Handoff(3, 7, 1))
    assert isinstance(res, Solved)
    assert res.absolute == [2, 6, 10, 18, 21]
    assert res.old_meter_wraps == [0, 1, 2, 3, None]
    # 新表段：B = [1, 4]，A[4] = 18 + (4 - 1) = 21。
    assert res.new_meter_wraps == [None, None, None, 0, 0]


def test_handoff_new_side_multi_wrap():
    # 新表侧多圈：m=10 开表 8，单步 [19,20] 到读数 7：B=27，回绕 2 次。
    res = solve(10, [0, 8, 7], [8, 19], [8, 20], Handoff(1, 10, 8))
    assert isinstance(res, Solved)
    assert res.absolute == [0, 8, 27]
    assert res.new_meter_wraps == [None, 0, 2]


def test_handoff_leaks_on_both_sides():
    # 两侧都有漏抄：旧表 [8, ?, 2]，新表开表 3 后 [?, 5]。
    res = solve(10, [8, None, 2, None, 5], [0, 0, 0, 0], [6, 6, 14, 14],
                Handoff(2, 7, 3))
    assert isinstance(res, Solved)
    assert res.absolute == [8, 8, 12, 12, 14]
    assert res.increments == [0, 4, 0, 2]
    assert res.old_meter_wraps == [0, 0, 1, None, None]
    assert res.new_meter_wraps == [None, None, 0, 0, 0]


def test_handoff_lexicographic_tie():
    # 新表段内部并列解取字典序最小：开表 0，末读 0，两步各 [3,6]，
    # 总量须为 10：d1=4, d2=6（d1=3 时 d2=7 越界）。
    res = solve(10, [0, 0, None, 0], [0, 3, 3], [0, 6, 6],
                Handoff(1, 10, 0))
    assert isinstance(res, Solved)
    assert res.increments == [0, 4, 6]
    assert res.absolute == [0, 0, 4, 10]
    assert res.new_meter_wraps == [None, 0, 0, 1]


def test_handoff_inconsistent_old_side():
    # 旧表段自身不可行：8 -> 9 但步长为 0。
    res = solve(10, [8, 9, 0], [0, 0], [0, 0], Handoff(1, 10, 0))
    assert isinstance(res, Inconsistent)
    assert res.position == 1


def test_handoff_inconsistent_new_side_position_mapped():
    # 新表段不可行：开表 0 到读数 99（m=100）只允许走 5。
    res = solve(10, [8, 2, 99], [0, 0], [6, 5], Handoff(1, 100, 0))
    assert isinstance(res, Inconsistent)
    assert res.position == 2


def test_handoff_inconsistent_at_handoff_reading():
    # 交接点的旧表末读与前面矛盾：最早不可延伸位置即交接点。
    res = solve(10, [1, None, 9, 0], [0, 0, 0], [3, 3, 3],
                Handoff(2, 10, 0))
    assert isinstance(res, Inconsistent)
    assert res.position == 2


def test_handoff_new_modulus_one():
    # 新表模数为 1：读数恒为 0，新表自身计数 B 全部以整数累计；
    # 字典序最小取 d=0（不增加即不回绕）。
    res = solve(10, [4, 4, 0], [0, 0], [0, 5], Handoff(1, 1, 0))
    assert isinstance(res, Solved)
    assert res.absolute == [4, 4, 4]
    assert res.new_meter_wraps == [None, 0, 0]


def test_handoff_huge_new_modulus():
    m_new = 10**60
    # 旧表 8 -> 2（A=12）；新表开表 m_new-1，读数 2，单步 [0, 2*m_new]。
    res = solve(10, [8, 2, 2], [0, 0], [6, 2 * m_new],
                Handoff(1, m_new, m_new - 1))
    assert isinstance(res, Solved)
    # 新表自身 B：m_new-1 -> m_new+2（d=3，回绕 1 次）。
    assert res.absolute == [8, 12, 12 + 3]
    assert res.new_meter_wraps == [None, 0, 1]
    assert res.old_meter_wraps == [0, 1, None]


def test_handoff_huge_modulus_many_positions():
    n = 300
    m_old = 10**30
    m_new = 10**40
    rng = random.Random(11)
    h = 150
    readings = [None] * n
    readings[0] = 0
    readings[h] = 5
    readings[-1] = 0
    for i in range(1, n - 1):
        if i != h and rng.random() < 0.3:
            readings[i] = rng.randrange(m_old if i <= h else m_new)
    min_steps = [0] * (n - 1)
    max_steps = [m_old if i < h else m_new for i in range(n - 1)]
    res = solve(m_old, readings, min_steps, max_steps,
                Handoff(h, m_new, 7))
    assert isinstance(res, Solved)
    check_handoff(m_old, readings, min_steps, max_steps, h, m_new, 7)


def test_no_handoff_response_unchanged():
    # 省略交接时结果结构与单表完全一致。
    res = solve(10, [8, None, 2], [0, 0], [6, 6])
    assert isinstance(res, Solved)
    assert res.absolute == [8, 8, 12]
    assert res.cumulative_wraps == [0, 0, 1]
    assert res.old_meter_wraps is None
    assert res.new_meter_wraps is None
