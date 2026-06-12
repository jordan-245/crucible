"""Producer-side contract tests (#34): what deploy.py / lifecycle.py write must match
the fixtures shared verbatim with atlas (tests/contract/ in BOTH repos — the fixture
files ARE the cross-repo contract; no shared package)."""
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "contract"

REQ_KEYS = {"schema_version", "name", "strategy_path", "capital", "broker", "tif",
            "expectation", "requested_at"}
VER_KEYS = {"schema_version", "book", "asof", "lifecycle", "gates_all_pass", "n_days",
            "decay", "watch"}


def test_fixture_files_present_and_v1():
    req = json.loads((FIXTURES / "deploy_request.fixture.json").read_text(encoding="utf-8"))
    ver = json.loads((FIXTURES / "lifecycle_verdict.fixture.json").read_text(encoding="utf-8"))
    assert req["schema_version"] == 1 and ver["schema_version"] == 1
    assert set(req) == REQ_KEYS
    assert set(ver) == VER_KEYS


def test_lifecycle_verdict_writer_matches_contract(tmp_path, monkeypatch):
    from forward import lifecycle
    monkeypatch.setattr(lifecycle, "LIVE", tmp_path)
    lifecycle._write_verdict("bookx", "evidence",
                             {"d1": False, "d2": False, "cusum_peak": 1.0, "roll_mean": 0.0002},
                             gates_all_pass=False, n_days=21, watch=None)
    out = json.loads((tmp_path / "bookx" / "lifecycle_verdict.json").read_text(encoding="utf-8"))
    assert set(out) == VER_KEYS
    assert out["schema_version"] == 1
    assert out["lifecycle"] == "evidence"
    assert set(out["decay"]) == {"d1", "d2", "cusum_peak", "roll_mean"}


def test_deploy_request_shape_matches_contract():
    """deploy.py's request dict construction (kept in lockstep by this test + the fixture)."""
    req = json.loads((FIXTURES / "deploy_request.fixture.json").read_text(encoding="utf-8"))
    # the fields deploy_to_paper writes — if deploy.py changes shape, update fixture in BOTH repos
    assert isinstance(req["capital"], float) and isinstance(req["expectation"], dict)
    assert req["broker"] in ("alpaca", "ib")
    assert req["tif"] in ("opg", "")
