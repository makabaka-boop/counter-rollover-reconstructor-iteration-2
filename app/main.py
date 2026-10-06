"""FastAPI 纯后端 JSON API。"""

from fastapi import FastAPI

from .models import TrajectoryRequest
from .solver import Anchor, Handoff, Inconsistent, Solved, solve

app = FastAPI(
    title="冷库机械累计表轨迹恢复",
    description=(
        "恢复跨零回绕的机械累计表绝对计数序列：先最小化最终绝对计数，"
        "再取完整序列字典序最小者；无解返回最早无法延伸的位置。"
        "可选一次换表交接（handoff）与盘点锚点（inventoryAnchor）。"
    ),
    version="1.2.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/trajectory")
def trajectory(req: TrajectoryRequest) -> dict:
    handoff = None
    if req.handoff is not None:
        handoff = Handoff(
            position=req.handoff.position,
            new_modulus=req.handoff.newModulus,
            opening_reading=req.handoff.openingReading,
        )
    anchor = None
    if req.inventoryAnchor is not None:
        anchor = Anchor(
            position=req.inventoryAnchor.position,
            absolute_count=req.inventoryAnchor.absoluteCount,
        )
    result = solve(
        req.modulus,
        req.readings,
        req.minStep,
        req.maxStep,
        handoff,
        anchor,
    )
    if isinstance(result, Inconsistent):
        return {"status": "INCONSISTENT", "position": result.position}
    assert isinstance(result, Solved)
    body: dict = {
        "status": "OK",
        "absolute": result.absolute,
        "increments": result.increments,
    }
    if result.old_meter_wraps is not None:
        assert result.new_meter_wraps is not None
        # 换表请求：两台表分别列回绕次数，未归属位置为 null；
        # 不再给单一的 cumulativeWraps（新表清零不是负耗用，无法共用）。
        body["cumulativeWraps"] = None
        body["oldMeterWraps"] = result.old_meter_wraps
        body["newMeterWraps"] = result.new_meter_wraps
    else:
        body["cumulativeWraps"] = result.cumulative_wraps
    return body
