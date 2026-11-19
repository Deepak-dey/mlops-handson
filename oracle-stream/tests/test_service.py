"""Tests for service.py: features, batch scoring and the stream loop, all with fakes (no Kafka, no MLflow)."""
import json
import pytest

from conftest import FakeMessage, fake_kafka, ride_message

# ---------- start-up ----------

def test_refuses_to_start_without_run_id(load_service):
    with pytest.raises((SystemExit, RuntimeError)):
        load_service(run_id = None)

# ---------- features and prediction ----------
@pytest.mark.parametrize(
    "pu, do, expected", [
    (43, 151, "43_151"),
    (43.0, 151.0, "43_151"),        
    ("43", "151", "43_151"),
]
)
def test_prepare_features_builds_pu_do_from_ints(load_service, pu,do,expected):
    svc = load_service()
    assert svc.prepare_features({"PULocationID": pu, "DOLocationID": do, "trip_distance": 1.2}) == {"PU_DO": expected, "trip_distance": 1.2} == {"PU_DO": expected, "trip_distance": 1.2}


def test_prepare_features_needs_every_feild(load_service):
    svc = load_service()
    with pytest.raises(KeyError):
        svc.prepare_features({"PULocationID": 1, "DOLocationID": 2})
        
def test_prepare_features_returns_one_float_per_row(load_service):
    svc = load_service()
    out = svc.predict([{"PU_DO": "43_151", "trip_distance": 1.2}, {"PU_DO": "1_2", "trip_distance": 5.0}])
    assert len(out) == 2 and all(type(x) is float for x in out)

# ---------- score_batch ----------

def test_score_batch_produces_one_message_per_good_ride(load_service,producer,events):
    svc = load_service(run_id="run-xyz")
    scored,skipped = svc.score_batch([ride_message(1), ride_message(2)], producer)
    assert (scored,skipped) == (2,0)
    produced = [e for e in events if e[0] == "produce"]
    assert [e[1] for e in produced] == ["predictions", "predictions"]
    assert [e[2] for e in produced] == ["r1", "r2"]                  # keyed by ride_id
    assert [e[3]["ride_id"] for e in produced] == ["r1", "r2"]       # ride_id must reach the output message
    assert {e[3]["model_version"] for e in produced} == {"run-xyz"}
    assert all(isinstance(e[3]["duration"], float) for e in produced)
    
