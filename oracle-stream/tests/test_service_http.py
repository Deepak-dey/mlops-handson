"""HTTP tests for service.py. Skipped unless fastapi and httpx are installed (pip install -r requirements-dev.txt)."""

import pytest
from conftest import fake_kafka, ride_message

fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="pip install fastapi httpx")
pytest.importorskip("httpx")

RIDE = {"PULocationID": 43, "DOLocationID": 151, "trip_distance": 1.2}

@pytest.fixture
def client(load_service):
    svc = load_service(run_id ="run-http")
    with fastapi_testclient.TestClient(svc.app) as c:
        yield c

def test_predict_endpoint(client,tiny_model):
    r = client.post("/predict",json = RIDE)
    assert r.status_code == 200
    assert r.json()["duration"] == pytest.approx(float(tiny_model.predict([{"PU_DO": "43_151", "trip_distance": 1.2}])[0]))
    
def test_predict_accpets_float_zone_ids(client):
    r = client.post("/predict", json={**RIDE, "PULocationID": 43.0, "DOLocationID": 151.0})
    assert r.status_code == 200 and r.json() == client.post("/predict",json = RIDE).json()

@pytest.mark.parametrize("payload", [
    {}, {"PULocationID": 43, "DOLocationID": 151}, {**RIDE, "trip_distance": -1},
    {**RIDE, "trip_distance": "far"}, {**RIDE, "PULocationID": "downtown"},
])
def test_predict_reject_bad_input(client,payload):
    assert client.post("/predict", json = payload).status_code == 422

def test_health_without_stream(client):
    assert client.get("/health").json() == {"model_version": "run-http", "streaming": False,
                                            "streamed": 0, "error": None}

def test_the_stream_starts_with_the_app_and_stops_with_it(load_service, events):
    box = {}
    kafka = fake_kafka(events,[[ride_message(1)]])
    svc = load_service(stream = "1", kafka = kafka)
    box["svc"] = svc
    with fastapi_testclient.TestClient(svc.app) as c:
        assert c.get("/health").json()["streaming"] is True
    assert svc.stop_event.is_set()
    assert kafka.state["closed"] is True
    