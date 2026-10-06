"""短序列穷举对拍的参考实现。

在步长取值范围很小的实例上直接 DFS 枚举所有增量组合，构造两级目标
（先最小化最终绝对计数，再最小化完整序列字典序）下的标准答案，
并计算“最早无法延伸的位置”，供与高效实现逐一比对。
"""

from __future__ import annotations


def brute_solve(modulus, readings, min_steps, max_steps):
    """返回 ("ok", absolute) 或 ("inconsistent", position)。"""
    n = len(readings)
    m = modulus
    best = None  # (final_value, tuple(A))

    def dfs(i, A):
        nonlocal best
        v = readings[i]
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

    dfs(0, [readings[0]])
    if best is None:
        return ("inconsistent", earliest_blocked(modulus, readings, min_steps, max_steps))
    return ("ok", list(best[1]))


def prefix_feasible(modulus, readings, min_steps, max_steps, p):
    """位置 0..p 的前缀是否存在合法赋值（p 处若已知须匹配读数）。"""
    n = len(readings)
    m = modulus

    def dfs(i, A):
        v = readings[i]
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

    return dfs(0, [readings[0]])


def earliest_blocked(modulus, readings, min_steps, max_steps):
    n = len(readings)
    for p in range(1, n):
        if not prefix_feasible(modulus, readings, min_steps, max_steps, p):
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
                        position, new_modulus, opening):
    """返回 ("ok", absolute, old_wraps, new_wraps) 或 ("inconsistent", position)。"""
    n = len(readings)
    assert readings[position] is not None  # 交接点必须是旧表最后一次读数
    best = None  # (final_value, tuple(A))

    def dfs(i, A):
        nonlocal best
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

    dfs(0, [readings[0]])
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
                            position, new_modulus, opening, p):
    """位置 0..p 的前缀是否存在合法赋值（按两侧各自的模数校验读数）。"""

    def dfs(i, A):
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

    return dfs(0, [readings[0]])


def earliest_blocked_handoff(old_modulus, readings, min_steps, max_steps,
                             position, new_modulus, opening):
    n = len(readings)
    for p in range(1, n):
        if not prefix_feasible_handoff(
            old_modulus, readings, min_steps, max_steps,
            position, new_modulus, opening, p,
        ):
            return p
    return None
