"""盘点锚点的 solver 对拍与边界测试。"""

import itertools
import random

from app.solver import Anchor, Handoff, Inconsistent, Solved, solve

from .brute import (
    brute_solve,
    brute_solve_handoff,
    earliest_blocked,
    earliest_blocked_handoff,
)


def _step_modes(m):
    return [
        (lo, hi)
        for lo in range(0, 3)
        for hi in range(lo, min(2 * m, lo + 2) + 1)
    ]


def _validate_plain(m, readings, min_steps, max_steps, anchor, res):
    assert isinstance(res, Solved)
    p, count = anchor.position, anchor.absolute_count
    A = res.absolute
    assert len(A) == len(readings)
    assert A[p] == count
    assert A[0] == (count if p == 0 else readings[0])
    assert all(a >= 0 for a in A)
    for i, d in enumerate(res.increments):
        assert min_steps[i] <= d <= max_steps[i]
        assert A[i + 1] - A[i] == d
    for i, v in enumerate(readings):
        if v is not None:
            assert A[i] % m == v
    assert res.cumulative_wraps == [a // m for a in A]


def _validate_handoff(
    m_old, readings, min_steps, max_steps, h, m_new, opening, anchor, res
):
    assert isinstance(res, Solved)
    p, count = anchor.position, anchor.absolute_count
    A = res.absolute
    assert len(A) == len(readings)
    assert A[p] == count
    assert all(a >= 0 for a in A)
    for i, d in enumerate(res.increments):
        assert min_steps[i] <= d <= max_steps[i]
        assert A[i + 1] - A[i] == d
    for i, v in enumerate(readings):
        if v is None:
            continue
        if i <= h:
            assert A[i] % m_old == v
        else:
            assert (A[i] - A[h] + opening) % m_new == v
    for i in range(len(readings)):
        assert res.old_meter_wraps[i] == (
            A[i] // m_old if i <= h else None
        )
        assert res.new_meter_wraps[i] == (
            None if i < h else (A[i] - A[h] + opening) // m_new
        )


def test_plain_anchor_exhaustive_small():
    rng = random.Random(20261006)
    checked = 0
    for m in range(1, 4):
        modes = _step_modes(m)
        for n in range(2, 6):
            combos = list(itertools.product(modes, repeat=n - 1))
            if len(combos) > 24:
                combos = rng.sample(combos, 24)
            pools = [list(range(m)) + [None]] * n
            for values in itertools.product(*pools):
                readings = list(values)
                if readings[0] is None or readings[-1] is None:
                    continue
                for combo in combos:
                    min_steps = [x[0] for x in combo]
                    max_steps = [x[1] for x in combo]
                    upper = readings[0] + sum(max_steps)
                    for p, v in enumerate(readings):
                        if v is None:
                            continue
                        # 同余的不同圈数及若干非同余非法值都覆盖。
                        candidates = set(range(max(0, v - m), upper + 1, m))
                        candidates.update([v + 1, v + m + 1])
                        if p == 0:
                            candidates.add(v + 2 * m)
                        for count in sorted(candidates):
                            anchor = Anchor(p, count)
                            res = solve(
                                m, readings, min_steps, max_steps,
                                anchor=anchor,
                            )
                            ref = brute_solve(
                                m, readings, min_steps, max_steps,
                                (p, count),
                            )
                            if ref[0] == "inconsistent":
                                assert isinstance(res, Inconsistent)
                                assert res.position == earliest_blocked(
                                    m, readings, min_steps, max_steps,
                                    (p, count),
                                )
                            else:
                                assert res.absolute == ref[1]
                                _validate_plain(
                                    m, readings, min_steps, max_steps,
                                    anchor, res,
                                )
                            checked += 1
    assert checked > 80_000


def test_handoff_anchor_exhaustive_small():
    rng = random.Random(20261007)
    checked = 0
    for m_old in range(1, 4):
        for m_new in range(1, 4):
            for n in range(2, 5):
                for h in range(n):
                    side_m = [m_old if i <= h else m_new for i in range(n)]
                    pools = []
                    for i in range(n):
                        choices = list(range(side_m[i]))
                        if i not in (0, h, n - 1):
                            choices.append(None)
                        pools.append(choices)
                    edge_modes = [
                        _step_modes(m_old if i < h else m_new)
                        for i in range(n - 1)
                    ]
                    combos = list(itertools.product(*edge_modes))
                    if len(combos) > 16:
                        combos = rng.sample(combos, 16)
                    for values in itertools.product(*pools):
                        readings = list(values)
                        for opening in range(m_new):
                            for combo in combos:
                                min_steps = [x[0] for x in combo]
                                max_steps = [x[1] for x in combo]
                                upper = readings[0] + sum(max_steps)
                                for p in range(n):
                                    if readings[p] is None:
                                        continue
                                    # 对所有小的非负绝对计数对拍，覆盖：
                                    # 交接前/交接点/交接后、合法多圈与非法同余。
                                    for count in range(upper + 1):
                                        anchor = Anchor(p, count)
                                        handoff = Handoff(
                                            h, m_new, opening
                                        )
                                        res = solve(
                                            m_old, readings, min_steps,
                                            max_steps, handoff, anchor,
                                        )
                                        ref = brute_solve_handoff(
                                            m_old, readings, min_steps,
                                            max_steps, h, m_new, opening,
                                            (p, count),
                                        )
                                        if ref[0] == "inconsistent":
                                            assert isinstance(res, Inconsistent)
                                            assert res.position == earliest_blocked_handoff(
                                                m_old, readings, min_steps,
                                                max_steps, h, m_new, opening,
                                                (p, count),
                                            )
                                        else:
                                            assert res.absolute == ref[1]
                                            _validate_handoff(
                                                m_old, readings, min_steps,
                                                max_steps, h, m_new, opening,
                                                anchor, res,
                                            )
                                        checked += 1
    assert checked > 100_000


def test_plain_anchor_raises_final_wrap():
    # m=10：8 -> 2，无锚点时 A[-1]=12；锚点要求交接值 22（再绕一圈）。
    res = solve(
        10,
        [8, None, 2],
        [0, 0],
        [20, 20],
        anchor=Anchor(2, 22),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [8, 8, 22]
    assert res.increments == [0, 14]
    assert res.cumulative_wraps == [0, 0, 2]


def test_plain_anchor_in_hole_sequence_lex_min():
    # 锚点固定中间值，后续仍按最小末值和字典序恢复，不能先无约束恢复。
    res = solve(
        10,
        [0, 5, None, 0],
        [0, 0, 0],
        [10, 10, 10],
        anchor=Anchor(1, 5),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [0, 5, 5, 10]
    assert res.increments == [5, 0, 5]


def test_plain_anchor_residue_mismatch_inconsistent():
    # 位置 1 的读数为 2，锚点绝对计数 13 余数为 3：格式合法但轨迹不可满足。
    res = solve(10, [0, 2, 0], [0, 0], [20, 20], anchor=Anchor(1, 13))
    assert res == Inconsistent(1)


def test_handoff_anchor_at_join_checks_same_physical_time():
    # 交接点真实值固定为 22；旧表读数仍为 2，必须由旧表已绕两圈表达。
    res = solve(
        10,
        [8, 2, 30],
        [0, 0],
        [20, 60],
        Handoff(1, 100, 0),
        Anchor(1, 22),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [8, 22, 52]
    assert res.old_meter_wraps == [0, 2, None]
    assert res.new_meter_wraps == [None, 0, 0]


def test_handoff_anchor_after_join_is_jointly_solved():
    # 旧表读数 2、新表开表 0、新读数 30。锚点 C=142 要求旧表交接值与新表
    # 自身计数联合满足，而不是两段分别取 12 与 30 后拼接成 42。
    res = solve(
        10,
        [8, 2, 30],
        [0, 0],
        [20, 200],
        Handoff(1, 100, 0),
        Anchor(2, 142),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [8, 12, 142]
    assert res.increments == [4, 130]
    assert res.old_meter_wraps == [0, 1, None]
    assert res.new_meter_wraps == [None, 0, 1]


def test_handoff_anchor_after_join_with_open_offset():
    # A[h]=2，新表开表 6；锚点 A[p]=2 + (B[p]-6)=4 ⇒ B[p]=8。
    res = solve(
        10,
        [2, 2, 8],
        [0, 0],
        [6, 20],
        Handoff(1, 100, 6),
        Anchor(2, 4),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [2, 2, 4]
    assert res.new_meter_wraps == [None, 0, 0]


def test_handoff_anchor_at_start_can_recover_old_cumulative():
    # 起点即换表，锚点在其后反推出同一物理时刻旧表已有真实累计数 2。
    res = solve(
        2,
        [0, 0],
        [0],
        [1],
        Handoff(0, 1, 0),
        Anchor(1, 2),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [2, 2]
    assert res.old_meter_wraps == [1, None]
    assert res.new_meter_wraps == [0, 0]


def test_handoff_anchor_after_join_multi_wrap_both_meters():
    # m_old=3、m_new=5，交接读数 1、开表 2、锚点读数 0。
    # 3*w + 5*k = 20-1-0+2=21；最小 w 解为 w=2,k=3。
    res = solve(
        3,
        [0, 1, 0],
        [0, 0],
        [20, 30],
        Handoff(1, 5, 2),
        Anchor(2, 20),
    )
    assert isinstance(res, Solved)
    assert res.absolute == [0, 7, 20]
    assert res.old_meter_wraps == [0, 2, None]
    assert res.new_meter_wraps == [None, 0, 3]


def test_handoff_anchor_after_join_inconsistent_reports_anchor():
    # 3*w + 100*k = 41-2-30=9，无非负整数解；不能返回任何半条轨迹。
    res = solve(
        10,
        [8, 2, 30],
        [0, 0],
        [20, 200],
        Handoff(1, 100, 0),
        Anchor(2, 41),
    )
    assert res == Inconsistent(2)


def test_anchor_does_not_change_unanchored_requests():
    res = solve(10, [8, None, 2], [0, 0], [6, 6])
    assert isinstance(res, Solved)
    assert res.absolute == [8, 8, 12]
    assert res.old_meter_wraps is None
    assert res.new_meter_wraps is None
