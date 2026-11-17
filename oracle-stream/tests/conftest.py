"""Fixtures that load service.py without MLflow, Kafka or a network.

service.py loads the model when it is imported, so each test imports a fresh copy with a fake `mlflow`
(returning a tiny model) and, when the stream runs, a fake `confluent_kafka`.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session")
def tiny_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.pipeline import make_pipeline
    
    X = [{"PU_DO": "43_151", "trip_distance": 1.2}, {"PU_DO": "43_151", "trip_distance": 1.0},
         {"PU_DO": "1_2", "trip_distance": 5.0}, {"PU_DO": "10_20", "trip_distance": 9.5}]
    y = [9.0, 8.0, 20.0, 35.0]
    
    return make_pipeline(DictVectorizer(),RandomForestRegressor(n_estimators=5,random_state=0)).fit(X,y)


class FakeMessage:
    def __init__(self, value = None, error = None):
        self._value = value
        self._error = error
    
    def value(self):
        return self._value
    
    def error(self):
        return self._error

def ride_message(i = 1, **overrides):
    body = {"ride_id":f"r{i}","PULocationID": 43, "DOLocationID": 151, "trip_distance": 1.2}
    body.update(overrides)
    return FakeMessage(json.dumps(body).encode())

class FakeProducer:
    def __init__(self,events):
        self.events = events
    
    def producer(self, topic, key = None, value = None,**_):
        self.events.append(("producer",topic, key,json.loads(value)))
        
    def flush(self, timeout = None):
        self.events.append(("flush",))


@pytest.fixture
def events():
    return []


@pytest.fixture
def producer (events):
    return FakeProducer(events)


@pytest.fixture
def load_service(monkeypatch, tiny_model):
    """load_service(env=..., kafka=...) -> a freshly imported service module."""
    def _load(run_id = "run-1", stream = "0", kafka = None, extra_env = None):
        monkeypatch.delenv("RUN_ID", raising = False)
        if run_id is not None:
            monkeypatch.setenv("RUN_ID",run_id)
        monkeypatch.setenv("STREAM", stream)
        for k,v in (extra_env or {}).items():
            monkeypatch.setenv(k,v)
        
        ml = types.ModuleType("mlflow")
        ml.set_tracking_uri = lambda uri:None
        sk = types.ModuleType("mlflow.sklearn")
        sk.load_model = lambda uri:tiny_model
        ml.sklearn = sk
        monkeypatch.setitem(sys.modules, "mlflow", ml)
        monkeypatch.setitem(sys.modules, "mlflow.sklearn", sk)
        if kafka is not None:
            monkeypatch.setitem(sys.modules, "confluent_kafka", kafka)
        
        spec = importlib.util.spec_from_file_location("service_under_test",ROOT/"Service.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return _load


def fake_kafka(events,batches, on_empty = None, boom = None):
    """A fake confluent_kafka module. `batches` are lists of messages handed out one per consume() call."""
    queue = list(batches)
    state = {"closed":False}
    
    
    class Consumer:
        def __init__(self, conf):
            self.conf = conf
            state["conf"] = conf
        
        def subscribe(self, topics):
            events.append(("subscribe",topics))
        
        def consume(self,num_messages, timeout):
            if boom:
                raise boom
            if queue:
                return queue.pop(0)
            if on_empty:
                on_empty()
            return []
        
        def commit(self,asynchronous=True):
            events.append(("commit",asynchronous))
        
        def close(self):
            state["close"] = True
            events.append(("close",))
        
    class Producer:
        def __init__(self,conf):
            self._p = FakeProducer(events)
        
        def produce(self,*a, **kw):
            self._p.produce(*a, **kw)
        
        def flush(self, timeout = None):
            self._p.flush(timeout)
    module = types.ModuleType("confluent_kafka")
    module.Consumer, module.Producer = Consumer, Producer
    module.state = state
    return module
            
                
    