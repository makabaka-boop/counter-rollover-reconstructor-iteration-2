"""短序列穷举对拍的参考实现。

在步长取值范围很小的实例上直接 DFS 枚举所有增量组合，构造两级目标
（先最小化最终绝对计数，再最小化完整序列字典序）下的标准答案，
并计算“最早无法延伸的位置”，供与高效实现逐一比对。
"""

from __future__ import annotations


def brute_solve(modulus, readings, min_steps, max_steps, anchor=None):
    """返回 ("ok", absolute) 或 ("inconsistent", position)。

    anchor 可选为 (position, absolute_count)。锚点在首位时会固定首项，
    否则首项仍按 readings[0] 起算。
    """
    n = len(readings)
    m = modulus
    best = None  # (final_value, tuple(A))

    def dfs(i, A):
        nonlocal best
        v = readings[i]
        if anchor is not None and i == anchor[0] and A[-1] != anchor[1]:
            return
        if v is not None and A[-1] % m != v:
            return
        if i == n - 1:
            key = (A[-1], tuple(A))
            if best is None or key < best:
                best = key
            return
        for d in range(min_steps[i], max_steps[i] + 1):
            A.append(A[-1] + d)
            dfs(i + 1, A)
            A.pop()

    start = anchor[1] if anchor is not None and anchor[0] == 0 else readings[0]
    dfs(0, [start])
    if best is None:
        return ("inconsistent", earliest_blocked(modulus, readings, min_steps, max_steps))
    return ("ok", list(best[1]))


def prefix_feasible(modulus, readings, min_steps, max_steps, p, anchor=None):
    """位置 0..p 的前缀是否存在合法赋值（p 处若已知须匹配读数）。"""
    n = len(readings)
    m = modulus

    def dfs(i, A):
        v = readings[i]
        if anchor is not None and i == anchor[0] and A[-1] != anchor[1]:
            return False
        if v is not None and A[-1] % m != v:
            return False
        if i == p:
            return True
        for d in range(min_steps[i], max_steps[i] + 1):
            A.append(A[-1] + d)
            if dfs(i + 1, A):
                A.pop()
                return True
            A.pop()
        return False

    start = anchor[1] if anchor is not None and anchor[0] == 0 else readings[0]
    return dfs(0, [start])


def earliest_blocked(modulus, readings, min_steps, max_steps, anchor=None):
    n = len(readings)
    for p in range(0, n):
        if not prefix_feasible(
            modulus, readings, min_steps, max_steps, p, anchor
        ):
            return p
    return None


# ---------------------------------------------------------------------------
# 换表交接的穷举参考：旧表段 [0..h] 用旧模数，新表段 [h..n-1] 用新模数，
# 新表自身计数 B[i] = A[i] - A[h] + opening（交接动作耗用为零，无跨段边）。
# ---------------------------------------------------------------------------

def _handoff_residue_ok(old_modulus, new_modulus, opening, position, readings, A, i):
    """位置 i 的已知读数是否与绝对计数 A 匹配（按所在侧取模）。"""
    v = readings[i]
    if v is None:
        return True
    if i <= position:
        return A[i] % old_modulus == v
    return (A[i] - A[position] + opening) % new_modulus == v


def brute_solve_handoff(old_modulus, readings, min_steps, max_steps,
                        position, new_modulus, opening, anchor=None):
    """返回 ("ok", absolute, old_wraps, new_wraps) 或 ("inconsistent", position)。

    anchor 可选为 (position, absolute_count)。锚点在首位时直接固定首项；
    当交接也在首位时，锚点可反推出交接时旧表已有累计数，因此需枚举可能
    的交接值。
    """
    n = len(readings)
    assert readings[position] is not None  # 交接点必须是旧表最后一次读数
    best = None  # (final_value, tuple(A))

    def dfs(i, A):
        nonlocal best
        if anchor is not None and i == anchor[0] and A[-1] != anchor[1]:
            return
        if not _handoff_residue_ok(
            old_modulus, new_modulus, opening, position, readings, A, i
        ):
            return
        if i == n - 1:
            key = (A[-1], tuple(A))
            if best is None or key < best:
                best = key
            return
        for d in range(min_steps[i], max_steps[i] + 1):
            A.append(A[-1] + d)
            dfs(i + 1, A)
            A.pop()

    if position == 0 and anchor is not None:
        starts = range(readings[0], readings[0] + sum(max_steps) + 1, old_modulus)
    elif anchor is not None and anchor[0] == 0:
        starts = [anchor[1]]
    else:
        starts = [readings[0]]
    for start in starts:
        dfs(0, [start])
    if best is None:
        return (
            "inconsistent",
            earliest_blocked_handoff(
                old_modulus, readings, min_steps, max_steps,
                position, new_modulus, opening,
            ),
        )
    A = list(best[1])
    old_wraps = [A[i] // old_modulus if i <= position else None for i in range(n)]
    new_wraps = [
        None if i < position else (A[i] - A[position] + opening) // new_modulus
        for i in range(n)
    ]
    return ("ok", A, old_wraps, new_wraps)


def prefix_feasible_handoff(old_modulus, readings, min_steps, max_steps,
                            position, new_modulus, opening, p, anchor=None):
    """位置 0..p 的前缀是否存在合法赋值（按两侧各自的模数校验读数）。"""

    def dfs(i, A):
        if anchor is not None and i == anchor[0] and A[-1] != anchor[1]:
            return False
        if not _handoff_residue_ok(
            old_modulus, new_modulus, opening, position, readings, A, i
        ):
            return False
        if i == p:
            return True
        for d in range(min_steps[i], max_steps[i] + 1):
            A.append(A[-1] + d)
            if dfs(i + 1, A):
                A.pop()
                return True
            A.pop()
        return False

    if position == 0 and anchor is not None:
        for start in range(
            readings[0], readings[0] + sum(max_steps) + 1, old_modulus
        ):
            if dfs(0, [start]):
                return True
        return False
    start = anchor[1] if anchor is not None and anchor[0] == 0 else readings[0]
    return dfs(0, [start])


def earliest_blocked_handoff(old_modulus, readings, min_steps, max_steps,
                             position, new_modulus, opening, anchor=None):
    n = len(readings)
    for p in range(0, n):
        if not prefix_feasible_handoff(
            old_modulus, readings, min_steps, max_steps,
            position, new_modulus, opening, p, anchor,
        ):
            return p
    return None
