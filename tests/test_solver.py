"""solver 对拍与边界测试。"""

import itertools
import random

import pytest

from app.solver import Inconsistent, Solved, solve

from .brute import brute_solve, earliest_blocked


def check_result(modulus, readings, min_steps, max_steps):
    """跑高效实现并完整校验返回结构。"""
    res = solve(modulus, list(readings), list(min_steps), list(max_steps))
    n = len(readings)
    if isinstance(res, Inconsistent):
        # 与穷举的最早不可延伸位置一致。
        assert res.position == earliest_blocked(
            modulus, readings, min_steps, max_steps
        )
        return res
    assert isinstance(res, Solved)
    assert len(res.absolute) == n
    assert len(res.increments) == n - 1
    assert len(res.cumulative_wraps) == n
    # 重建一致性。
    A = [readings[0]]
    for d in res.increments:
        A.append(A[-1] + d)
    assert A == res.absolute
    # 步长区间。
    for i, d in enumerate(res.increments):
        assert min_steps[i] <= d <= max_steps[i]
    # 已知点取模等于原读数。
    for i, v in enumerate(readings):
        if v is not None:
            assert res.absolute[i] % modulus == v
    # 非负、单调不减。
    assert all(a >= 0 for a in res.absolute)
    assert all(res.absolute[i] <= res.absolute[i + 1] for i in range(n - 1))
    # 累计回绕 = floor(A / m)。
    assert res.cumulative_wraps == [a // modulus for a in res.absolute]
    return res


# ---------------------------------------------------------------------------
# 穷举对拍：枚举所有（m, n, readings, 步长区间）的小实例。
# ---------------------------------------------------------------------------

def test_exhaustive_small():
    rng = random.Random(20260923)
    count = 0
    # 步长区间取值池（lo<=hi<=2m 的小范围，含不可行构造）。
    for m in range(1, 4):
        step_modes = [
            (lo, hi)
            for lo in range(0, 3)
            for hi in range(lo, min(2 * m, lo + 2) + 1)
        ]
        for n in range(2, 6):
            # 读数取值：0..m-1 或 None；首尾固定已知，逐一枚举。
            inner_choices = list(range(m)) + [None]
            all_combos = list(itertools.product(step_modes, repeat=n - 1))
            if len(all_combos) <= 400:
                sampled = None
            else:
                # 抽样与读数无关：同一组步长模式对每种读数都跑一遍，
                # 保证读数维度全覆盖，步长维度随机覆盖。
                rng_s = random.Random(hash((m, n)) & 0xFFFFFFFF)
                sampled = rng_s.sample(all_combos, 200)
            for inner in itertools.product(inner_choices, repeat=n - 2):
                for r0 in range(m):
                    for rn in range(m):
                        readings = [r0] + list(inner) + [rn]
                        combos = all_combos if sampled is None else sampled
                        for combo in combos:
                            min_steps = [c[0] for c in combo]
                            max_steps = [c[1] for c in combo]
                            res = check_result(m, readings, min_steps, max_steps)
                            ref = brute_solve(m, readings, min_steps, max_steps)
                            if ref[0] == "inconsistent":
                                assert isinstance(res, Inconsistent)
                                assert res.position == ref[1]
                            else:
                                assert isinstance(res, Solved)
                                assert res.absolute == ref[1], (
                                    f"m={m} readings={readings} "
                                    f"steps={list(zip(min_steps, max_steps))}"
                                )
                            count += 1
    assert count > 150_000


# ---------------------------------------------------------------------------
# 随机对拍（稍大范围）。
# ---------------------------------------------------------------------------

def test_random_medium():
    rng = random.Random(42)
    for _ in range(300):
        m = rng.randrange(1, 7)
        n = rng.randrange(2, 8)
        readings = [None] * n
        readings[0] = rng.randrange(m)
        readings[-1] = rng.randrange(m)
        for i in range(1, n - 1):
            readings[i] = None if rng.random() < 0.5 else rng.randrange(m)
        min_steps, max_steps = [], []
        for _ in range(n - 1):
            lo = rng.randrange(0, min(4, 2 * m + 1))
            hi = lo + rng.randrange(0, min(4, 2 * m - lo + 1))
            min_steps.append(lo)
            max_steps.append(hi)
        check_result(m, readings, min_steps, max_steps)
        ref = brute_solve(m, readings, min_steps, max_steps)
        res = solve(m, readings, min_steps, max_steps)
        if ref[0] == "inconsistent":
            assert isinstance(res, Inconsistent)
            assert res.position == ref[1]
        else:
            assert isinstance(res, Solved)
            assert res.absolute == ref[1]


# ---------------------------------------------------------------------------
# 手工场景：连续回绕、多个空洞、并列解取字典序。
# ---------------------------------------------------------------------------

def test_continuous_wraparound():
    # m=5，三步各 [4,10]，2 -> 3：最小末尾 18（回绕 3 次），
    # 字典序最小序列 2,6,10,18。
    res = solve(5, [2, None, None, 3], [4, 4, 4], [10, 10, 10])
    assert isinstance(res, Solved)
    assert res.absolute == [2, 6, 10, 18]
    assert res.increments == [4, 4, 8]
    assert res.cumulative_wraps == [0, 1, 2, 3]


def test_multiple_holes_choose_lex_min():
    # m=10，[8, ?, ?, 2]，步长各 [0,6]。
    # 段总量 = 4+10q；三步范围 [0,18] ⇒ q=0(4) 或 q=1(14)。
    # 最小末尾 q=0 ⇒ 总量 4；字典序最小：0,0,4。
    res = solve(10, [8, None, None, 2], [0, 0, 0], [6, 6, 6])
    assert isinstance(res, Solved)
    assert res.absolute == [8, 8, 8, 12]
    assert res.increments == [0, 0, 4]
    assert res.cumulative_wraps == [0, 0, 0, 1]


def test_tie_lexicographic():
    # 总量须为 10 时两个端点之间有多个空洞：验证逐步取最小。
    # m=10，[0,?,?,0]，步长各 [0,6]：q=0(总量0) 或 q=1(总量10)。
    # 最小末尾选 q=0 ⇒ 全 0；这里构造必须 q=1 的场景：
    # 首读 0、末读 0，步长各 [3,6]，总量范围 [6,12]，q=1 唯一。
    # 总量 10，字典序最小逐步贪心：d0=3（剩余 [3,7] 可凑 7），
    # d1=3（剩余 [3,6] 可凑 4），d2=4。
    res = solve(10, [0, None, None, 0], [3, 3, 3], [6, 6, 6])
    assert isinstance(res, Solved)
    assert res.increments == [3, 3, 4]
    assert res.absolute == [0, 3, 6, 10]
    assert res.cumulative_wraps == [0, 0, 0, 1]


def test_known_interior_points():
    # 两个已知段，段间传播取交。
    res = solve(10, [0, 4, None, 8], [0, 0, 0], [20, 20, 20])
    assert isinstance(res, Solved)
    assert res.absolute[1] == 4
    assert res.absolute[-1] == 8
    assert res.absolute[3] % 10 == 8


def test_inconsistent_earliest_position():
    # [1, ?, 9]，单步 [0,3]：总量范围 [0,6]，需求 8+10q ⇒ q=0 时 8>6，无解；
    # 位置 1 是未知点总能延伸（d∈[0,3]），位置 2 已知点匹配不上。
    res = solve(10, [1, None, 9], [0, 0], [3, 3])
    assert isinstance(res, Inconsistent)
    assert res.position == 2


def test_two_point_inconsistent():
    res = solve(10, [1, 9], [0], [3])
    assert isinstance(res, Inconsistent)
    assert res.position == 1


def test_modulus_one():
    res = solve(1, [0, None, 0], [0, 0], [2, 2])
    assert isinstance(res, Solved)
    assert res.absolute == [0, 0, 0]
    assert res.cumulative_wraps == [0, 0, 0]


def test_max_step_two_modulus_double_wrap():
    # 单步即可回绕两次：m=10，5 -> 3，步长 [0,20]。
    # S = -2 + 10q ∈ [0,20]：q=1(8) 最小。
    res = solve(10, [5, 3], [0], [20])
    assert isinstance(res, Solved)
    assert res.absolute == [5, 13]
    assert res.increments == [8]
    assert res.cumulative_wraps == [0, 1]


def test_zero_steps_force_equal():
    res = solve(10, [7, 7], [0], [0])
    assert isinstance(res, Solved)
    assert res.absolute == [7, 7]
    res_bad = solve(10, [7, 8], [0], [0])
    assert isinstance(res_bad, Inconsistent)
    assert res_bad.position == 1


# ---------------------------------------------------------------------------
# 大模数/大整数：不枚举绝对值上界，只做区间算术。
# ---------------------------------------------------------------------------

def test_huge_modulus_no_enumeration():
    m = 10**60
    res = solve(m, [m - 1, 2], [0], [2 * m])
    assert isinstance(res, Solved)
    # S = 3 + m*q，最小 q=0（A 不减）⇒ d=3，不回绕。
    assert res.increments == [3]
    assert res.absolute == [m - 1, m + 2]
    assert res.cumulative_wraps == [0, 1]


def test_huge_modulus_many_wraps():
    m = 10**40
    # 各步 [2m-3, 2m+?]——为满足 maxStep<=2m，取 [2m-3, 2m]。
    # 总量范围 [6m-9, 6m]，末读 5：总量 = 5 + mq。
    # q=6 -> 6m+5 超上界；q=5 -> 5m+5 < 6m-9（m 很大）⇒ 无解，INCONSISTENT。
    bad = solve(
        m,
        [0, None, None, 5],
        [2 * m - 3, 2 * m - 3, 2 * m - 3],
        [2 * m, 2 * m, 2 * m],
    )
    assert isinstance(bad, Inconsistent)
    assert bad.position == 3

    # 可行构造：各步 [2m-6, 2m]，总量范围 [6m-18, 6m]，仍差 5……
    # 改末读为 m-9：总量 = mq - 9。q=6 -> 6m-9 ∈ [6m-18, 6m] 可行。
    ok = solve(
        m,
        [0, None, None, m - 9],
        [2 * m - 6, 2 * m - 6, 2 * m - 6],
        [2 * m, 2 * m, 2 * m],
    )
    assert isinstance(ok, Solved)
    assert ok.absolute[-1] == 6 * m - 9
    assert ok.cumulative_wraps[-1] == 5  # floor((6m-9)/m) = 5
    assert sum(ok.increments) == 6 * m - 9
    # 字典序最小：前两步取最小 2m-6，剩余 = (6m-9)-2(2m-6) = 2m+3 > 2m 不可行，
    # 故第一步 2m-6 后后两步需和 4m-3；第二步最小 (4m-3)-2m = 2m-3。
    assert ok.increments == [2 * m - 6, 2 * m - 3, 2 * m]


def test_three_hundred_positions_performance():
    n = 300
    m = 10**30
    rng = random.Random(7)
    readings = [None] * n
    readings[0] = 0
    readings[-1] = 0
    for i in range(1, n - 1):
        if rng.random() < 0.3:
            readings[i] = rng.randrange(m)
    min_steps = [m] * (n - 1)
    max_steps = [m + 2] * (n - 1)
    res = solve(m, readings, min_steps, max_steps)
    # 只要求快速给出一个合法解或 INCONSISTENT。
    if isinstance(res, Solved):
        check_result(m, readings, min_steps, max_steps)
    else:
        assert isinstance(res, Inconsistent)
        assert 0 <= res.position < n
