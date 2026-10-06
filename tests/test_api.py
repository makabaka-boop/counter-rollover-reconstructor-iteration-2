"""FastAPI 端点测试：422 校验、成功响应、INCONSISTENT 形态。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def post(payload):
    return client.post("/trajectory", json=payload)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ok_response_shape():
    r = post(
        {
            "modulus": 10,
            "readings": [8, None, 2],
            "minStep": [0, 0],
            "maxStep": [6, 6],
        }
    )
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "OK",
        "absolute": [8, 8, 12],
        "increments": [0, 4],
        "cumulativeWraps": [0, 0, 1],
    }


def test_inconsistent_response():
    r = post(
        {
            "modulus": 10,
            "readings": [1, 9],
            "minStep": [0],
            "maxStep": [3],
        }
    )
    assert r.status_code == 200
    assert r.json() == {"status": "INCONSISTENT", "position": 1}


# ---------------------------------------------------------------------------
# 422：未知字段、数组错长、越界、类型错误、首尾缺失等。
# ---------------------------------------------------------------------------

BASE = {
    "modulus": 10,
    "readings": [1, None, 9],
    "minStep": [0, 0],
    "maxStep": [9, 9],
}


def _expect_422(payload):
    r = post(payload)
    assert r.status_code == 422, r.text


def test_unknown_top_level_field():
    bad = dict(BASE, extra=1)
    _expect_422(bad)


def test_missing_field():
    bad = {k: v for k, v in BASE.items() if k != "modulus"}
    _expect_422(bad)


def test_wrong_array_length_minstep():
    bad = dict(BASE, minStep=[0])
    _expect_422(bad)


def test_wrong_array_length_maxstep():
    bad = dict(BASE, maxStep=[9, 9, 9])
    _expect_422(bad)


def test_readings_too_short():
    bad = dict(BASE, readings=[5], minStep=[], maxStep=[])
    _expect_422(bad)


def test_readings_too_long():
    bad = dict(
        BASE,
        readings=[0] * 301,
        minStep=[0] * 300,
        maxStep=[0] * 300,
    )
    _expect_422(bad)


def test_first_reading_null():
    bad = dict(BASE, readings=[None, 5, 9])
    _expect_422(bad)


def test_last_reading_null():
    bad = dict(BASE, readings=[1, 5, None])
    _expect_422(bad)


def test_reading_out_of_range_high():
    bad = dict(BASE, readings=[1, 10, 9])
    _expect_422(bad)


def test_reading_negative():
    bad = dict(BASE, readings=[-1, None, 9])
    _expect_422(bad)


def test_modulus_zero():
    bad = dict(BASE, modulus=0)
    _expect_422(bad)


def test_step_order_violation():
    bad = dict(BASE, minStep=[5, 0], maxStep=[4, 9])
    _expect_422(bad)


def test_step_exceeds_two_modulus():
    bad = dict(BASE, minStep=[0, 0], maxStep=[21, 9])
    _expect_422(bad)


def test_negative_step():
    bad = dict(BASE, minStep=[-1, 0], maxStep=[9, 9])
    _expect_422(bad)


def test_float_rejected():
    bad = dict(BASE, modulus=10.5)
    _expect_422(bad)


def test_float_integer_like_rejected():
    bad = dict(BASE, minStep=[0.0, 0])
    _expect_422(bad)


def test_string_rejected():
    bad = dict(BASE, modulus="10")
    _expect_422(bad)


def test_bool_rejected():
    bad = dict(BASE, readings=[True, None, 9])
    _expect_422(bad)


def test_null_in_steps_rejected():
    bad = dict(BASE, minStep=[None, 0])
    _expect_422(bad)


def test_nested_unknown_field_inside_readings():
    # readings 元素只能是整数或 null。
    bad = dict(BASE, readings=[1, {"x": 1}, 9])
    _expect_422(bad)


# ---------------------------------------------------------------------------
# 换表交接：成功 / 无解 / 422 校验矩阵。
# ---------------------------------------------------------------------------

HO = {
    "modulus": 10,
    "readings": [8, 2, 30],
    "minStep": [0, 0],
    "maxStep": [6, 60],
    "handoff": {"position": 1, "newModulus": 100, "openingReading": 0},
}


def test_handoff_ok_response_shape():
    r = post(HO)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "status": "OK",
        "absolute": [8, 12, 42],
        "increments": [4, 30],
        "cumulativeWraps": None,
        "oldMeterWraps": [0, 1, None],
        "newMeterWraps": [None, 0, 0],
    }


def test_handoff_position_zero():
    payload = dict(
        HO,
        readings=[3, 1],
        minStep=[0],
        maxStep=[20],
        handoff={"position": 0, "newModulus": 10, "openingReading": 3},
    )
    r = post(payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["absolute"] == [3, 11]
    assert body["oldMeterWraps"] == [0, None]
    assert body["newMeterWraps"] == [0, 1]
    assert body["cumulativeWraps"] is None


def test_handoff_position_last():
    payload = dict(
        HO,
        readings=[8, 2],
        minStep=[0],
        maxStep=[6],
        handoff={"position": 1, "newModulus": 100, "openingReading": 0},
    )
    r = post(payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["absolute"] == [8, 12]
    assert body["oldMeterWraps"] == [0, 1]
    assert body["newMeterWraps"] == [None, 0]


def test_handoff_inconsistent_position():
    # 新表段不可行：开表 0 -> 99 但只允许走 5。
    payload = dict(HO, maxStep=[6, 5])
    r = post(payload)
    assert r.status_code == 200
    assert r.json() == {"status": "INCONSISTENT", "position": 2}


def test_handoff_null_falls_back_to_plain_request():
    # 省略交接的请求与响应保持原样（readings 全部按旧模数）。
    payload = {
        "modulus": 10,
        "readings": [8, None, 2],
        "minStep": [0, 0],
        "maxStep": [6, 6],
        "handoff": None,
    }
    r = post(payload)
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "OK",
        "absolute": [8, 8, 12],
        "increments": [0, 4],
        "cumulativeWraps": [0, 0, 1],
    }
    assert "oldMeterWraps" not in body
    assert "newMeterWraps" not in body


def _expect_422_ho(**changes):
    payload = dict(HO)
    handoff_changes = changes.pop("handoff", {})
    payload.update(changes)
    if isinstance(handoff_changes, dict):
        payload["handoff"] = dict(HO["handoff"], **handoff_changes)
    else:
        payload["handoff"] = handoff_changes
    _expect_422(payload)


def test_handoff_unknown_field():
    _expect_422_ho(handoff={"extra": 1})


def test_handoff_missing_field():
    bad = dict(HO, handoff={"position": 1, "openingReading": 0})
    _expect_422(bad)


def test_handoff_position_negative():
    _expect_422_ho(handoff={"position": -1})


def test_handoff_position_equals_length():
    _expect_422_ho(handoff={"position": 3})


def test_handoff_position_beyond_length():
    _expect_422_ho(handoff={"position": 99})


def test_handoff_new_modulus_zero():
    _expect_422_ho(handoff={"newModulus": 0})


def test_handoff_opening_negative():
    _expect_422_ho(handoff={"openingReading": -1})


def test_handoff_opening_equals_new_modulus():
    _expect_422_ho(handoff={"newModulus": 100, "openingReading": 100})


def test_handoff_position_reading_null():
    # 交接点必须持有旧表最后一次读数（已知值）。
    _expect_422_ho(readings=[8, None, 30])


def test_handoff_reading_after_position_exceeds_new_modulus():
    # 30 对旧模数 10 合法，但在交接之后必须 < 新模数 20。
    _expect_422_ho(handoff={"newModulus": 20})


def test_handoff_reading_before_position_exceeds_old_modulus():
    # 交接点读数 12 >= 旧模数 10。
    _expect_422_ho(readings=[8, 12, 30], handoff={"newModulus": 100})


def test_handoff_step_after_position_exceeds_two_new_modulus():
    # 交接后的边按新模数约束：maxStep=50 > 2*20（读数 15 仍合法）。
    _expect_422_ho(
        readings=[8, 2, 15],
        maxStep=[6, 50],
        handoff={"newModulus": 20},
    )


def test_handoff_step_after_position_allowed_with_two_new_modulus():
    # 对照：maxStep=50 == 2*25 合法（且该步足够新表从 0 走到 15）。
    payload = dict(
        HO,
        readings=[8, 2, 15],
        maxStep=[6, 50],
        handoff={"position": 1, "newModulus": 25, "openingReading": 0},
    )
    r = post(payload)
    assert r.status_code == 200, r.text
    assert r.json()["absolute"] == [8, 12, 27]


def test_handoff_wrong_type():
    _expect_422_ho(handoff="now")


def test_handoff_float_position():
    _expect_422_ho(handoff={"position": 1.0})


def test_handoff_float_opening():
    _expect_422_ho(handoff={"openingReading": 0.0})


def test_handoff_bool_new_modulus():
    _expect_422_ho(handoff={"newModulus": True})


# ---------------------------------------------------------------------------
# 盘点锚点：成功 / 无解 / 422 校验。
# ---------------------------------------------------------------------------

ANCHOR_BASE = {
    "modulus": 10,
    "readings": [8, 2, 30],
    "minStep": [0, 0],
    "maxStep": [20, 200],
    "handoff": {"position": 1, "newModulus": 100, "openingReading": 0},
    "inventoryAnchor": {"position": 2, "absoluteCount": 142},
}


def test_anchor_after_handoff_ok_response_shape():
    r = post(ANCHOR_BASE)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "status": "OK",
        "absolute": [8, 12, 142],
        "increments": [4, 130],
        "cumulativeWraps": None,
        "oldMeterWraps": [0, 1, None],
        "newMeterWraps": [None, 0, 1],
    }


def test_anchor_at_join_ok_response_shape():
    payload = dict(
        ANCHOR_BASE,
        maxStep=[20, 60],
        inventoryAnchor={"position": 1, "absoluteCount": 22},
    )
    r = post(payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["absolute"] == [8, 22, 52]
    assert body["oldMeterWraps"] == [0, 2, None]
    assert body["newMeterWraps"] == [None, 0, 0]


def test_anchor_residue_mismatch_returns_inconsistent_not_422():
    payload = dict(
        ANCHOR_BASE,
        inventoryAnchor={"position": 2, "absoluteCount": 41},
    )
    r = post(payload)
    assert r.status_code == 200
    assert r.json() == {"status": "INCONSISTENT", "position": 2}


def test_anchor_null_keeps_plain_response():
    payload = {
        "modulus": 10,
        "readings": [8, None, 2],
        "minStep": [0, 0],
        "maxStep": [6, 6],
        "inventoryAnchor": None,
    }
    r = post(payload)
    assert r.status_code == 200
    body = r.json()
    assert body == {
        "status": "OK",
        "absolute": [8, 8, 12],
        "increments": [0, 4],
        "cumulativeWraps": [0, 0, 1],
    }
    assert "oldMeterWraps" not in body
    assert "newMeterWraps" not in body


def test_short_anchor_field_name_is_rejected():
    payload = dict(ANCHOR_BASE)
    payload["anchor"] = payload.pop("inventoryAnchor")
    _expect_422(payload)


def _expect_422_anchor(**changes):
    payload = dict(ANCHOR_BASE)
    anchor_changes = changes.pop("inventoryAnchor", {})
    payload.update(changes)
    if isinstance(anchor_changes, dict):
        payload["inventoryAnchor"] = dict(
            ANCHOR_BASE["inventoryAnchor"], **anchor_changes
        )
    else:
        payload["inventoryAnchor"] = anchor_changes
    _expect_422(payload)


def test_anchor_unknown_field():
    _expect_422_anchor(inventoryAnchor={"extra": 1})


def test_anchor_missing_field():
    bad = dict(ANCHOR_BASE, inventoryAnchor={"position": 2})
    _expect_422(bad)


def test_anchor_position_negative():
    _expect_422_anchor(inventoryAnchor={"position": -1})


def test_anchor_position_beyond_length():
    _expect_422_anchor(inventoryAnchor={"position": 99})


def test_anchor_position_null_reading():
    _expect_422_anchor(readings=[8, None, 30])


def test_anchor_negative_count():
    _expect_422_anchor(inventoryAnchor={"absoluteCount": -1})


def test_anchor_float_count():
    _expect_422_anchor(inventoryAnchor={"absoluteCount": 1.0})


def test_anchor_bool_position():
    _expect_422_anchor(inventoryAnchor={"position": True})


def test_anchor_string_count():
    _expect_422_anchor(inventoryAnchor={"absoluteCount": "142"})


def test_anchor_wrong_type():
    _expect_422_anchor(inventoryAnchor=142)
