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
    
def test_score_batch_skips_malformed_messages(load_service, producer, events):
    svc = load_service()
    msgs = [ride_message(1), FakeMessage(b"not json"), FakeMessage(json.dumps({"ride_id":"x"}).encode()),
            ride_message(2,trip_distance = "far")]
    assert svc.score_batch(msgs,producer) == (1,3)
    assert [e[2] for e in events if e[0] == "produce"] == ["r1"]
    

def test_score_batch_matches_the_model(load_service,producer,events,tiny_model):
    svc = load_service()
    svc.score_batch(
        [ride_message(1)], producer
    )
    expected = float(tiny_model.predict([{"PU_DO": "43_151", "trip_distance": 1.2}])[0])
    assert events[0][3]["duration"] == pytest.approx(expected)

def test_score_batch_raises_on_a_kafka_error(load_service,producer,events):
    svc = load_service()
    with pytest.raises(RuntimeError, match="boom"):
        svc.score_batch([ride_message(1), FakeMessage(error="boom")],producer)
    assert events == []

# ---------- stream_loop ----------    

def run_loop(load_service, events, batches, **kw):
    box = {}
    kafka = fake_kafka(events, batches, on_empty=lambda: box['svc'].stop_event.set(),**kw)
    box["svc"] = load_service(stream = "1", kafka = kafka)
    box["svc"].stream_loop()
    return box["svc"], kafka

def test_loop_flushes_then_commits_once_per_batch(load_service, events):
    run_loop(load_service, events, [[ride_message(1), ride_message(2)], [ride_message(3)]])
    kinds = [e[0] for e in events if e[0] in ("produce", "flush", "commit")]
    assert kinds == ["produce", "produce", "flush", "commit", "produce", "flush", "commit"]

def test_loop_counts_each_ride_once(load_service,events):
    svc,_ = run_loop(load_service, events, [[ride_message(i) for i in range(5)],[ride_message(9)]])
    assert svc.stats["streamed"] == 6 and svc.stats["error"] is None
    
def test_loop_commits_manually_and_reads_from_the_start(load_service, events):
    _,kafka = run_loop(load_service,events,[])
    assert kafka.state["conf"]["enable.auto.commit"] is False
    assert kafka.state["conf"]["auto.offset.reset"] == "earliest"
    assert ("subscribe", ["rides"]) in events
    
def test_loop_closes_the_consumer_on_a_clean_stop(load_service, events):
    _,kafka = run_loop(load_service,events,[[ride_message(1)]])
    assert kafka.state["closed"] is True

def test_a_crash_is_recorded_and_the_consumer_still_closes(load_service, events):
    svc,kafka = run_loop(load_service,events,[], boom = ConnectionError("broker gone"))
    assert svc.stats["error"] == "broker gone"
    assert kafka.state["closed"] is True

def test_nothing_is_committed_when_a_batch_fails(load_service, events):
    svc,_ = run_loop(load_service,events,[[ride_message(1),FakeMessage(error= "boom")]])
    assert not any(e[0] in ("flush","commit") for e in events)
    assert svc.stats["streamed"] == 0