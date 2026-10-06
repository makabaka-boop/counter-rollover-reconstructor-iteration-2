"""请求/响应 Pydantic 模型与跨字段校验。"""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

# 严格整数：拒绝 1.0、"10"、true 等隐式转换。
StrictInt = Annotated[int, Field(strict=True)]


class HandoffSpec(BaseModel):
    """一次换表交接：旧表在 ``position`` 处留下最后一次读数，同一时点新表
    以 ``openingReading`` 开表（新表模数 ``newModulus``）。交接动作本身耗用
    为零，该位置之前按旧模数还原，之后的步长从新表开表读数起算。"""

    model_config = {"extra": "forbid"}

    position: StrictInt = Field(ge=0)
    newModulus: StrictInt = Field(ge=1)
    openingReading: StrictInt = Field(ge=0)


class TrajectoryRequest(BaseModel):
    model_config = {"extra": "forbid"}

    modulus: StrictInt = Field(ge=1)
    readings: list[StrictInt | None] = Field(min_length=2, max_length=300)
    minStep: list[StrictInt]
    maxStep: list[StrictInt]
    handoff: HandoffSpec | None = None

    @model_validator(mode="after")
    def _check_constraints(self) -> "TrajectoryRequest":
        n = len(self.readings)
        if len(self.minStep) != n - 1 or len(self.maxStep) != n - 1:
            raise ValueError(
                "minStep and maxStep must each have exactly len(readings) - 1 items"
            )
        # 首尾均已知；其余允许 null。
        if self.readings[0] is None or self.readings[-1] is None:
            raise ValueError("first and last readings are required")

        # 逐位置模数：无交接时全段为旧模数；有交接时交接点之前（含交接点
        # 的最后一次旧表读数）按旧模数，之后按新模数。边 i 连接 i -> i+1，
        # 交接点及以后的步长由新表累计，故边 i 属于旧表当且仅当 i < h。
        ho = self.handoff
        if ho is None:
            moduli = [self.modulus] * n
            edge_moduli = [self.modulus] * (n - 1)
        else:
            if ho.position > n - 1:
                raise ValueError(
                    "handoff.position must be within the readings range"
                )
            if self.readings[ho.position] is None:
                raise ValueError(
                    "handoff.position must hold the old meter's last reading "
                    "(a known value)"
                )
            if ho.openingReading >= ho.newModulus:
                raise ValueError(
                    "handoff.openingReading must satisfy "
                    "0 <= openingReading < newModulus"
                )
            moduli = [self.modulus] * (ho.position + 1) + [ho.newModulus] * (
                n - ho.position - 1
            )
            edge_moduli = [self.modulus] * ho.position + [ho.newModulus] * (
                n - 1 - ho.position
            )

        for v, m in zip(self.readings, moduli):
            if v is not None and not (0 <= v < m):
                raise ValueError(
                    "known readings must satisfy 0 <= v < modulus "
                    "(newModulus after the handoff position)"
                )
        for lo, hi, m in zip(self.minStep, self.maxStep, edge_moduli):
            # 步长上限按登记该步的表的模数约束。
            if not (0 <= lo <= hi <= 2 * m):
                raise ValueError(
                    "each pair must satisfy 0 <= minStep <= maxStep "
                    "<= 2 * (segment modulus)"
                )
        return self


class TrajectoryResponse(BaseModel):
    status: str
    absolute: list[int]
    increments: list[int]
    cumulativeWraps: list[int] | None = None
    oldMeterWraps: list[int | None] | None = None
    newMeterWraps: list[int | None] | None = None
