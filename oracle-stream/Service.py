
"""FastAPI service: your /predict endpoint plus the Kafka streaming scorer, in one process.
 
    POST /predict   one ride in, one duration out (what your Flask app did)
    GET  /health    model version and how many rides the stream has scored
    background      reads the `rides` topic, writes the `predictions` topic
 
Run:
    export RUN_ID=<mlflow run id>
    export MLFLOW_TRACKING_URI=sqlite:////absolute/path/to/mlflow.db   # or http://127.0.0.1:5000
    uvicorn service:app --host 0.0.0.0 --port 9696
Set STREAM=0 to run only the HTTP API (no Kafka needed).
"""


import json 
import logging
import os
import threading
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
import mlflow.sklearn
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

log = logging.getLogger("duration-service")

RUN_ID = os.getenv("RUN_ID")
ARTIFACT = os.getenv("MODEL_ARTIFACT", "model")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
IN_TOPIC = os.getenv("IN_TOPIC", "rides")
OUT_TOPIC = os.getenv("OUT_TOPIC", "predictions")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
STREAM = os.getenv("STREAM", "1") != "0"

if not RUN_ID:
    raise SystemExit("set RUN_ID to the MLflow run that holds the model")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/{ARTIFACT}")


stats = {"streamed": 0, "error": None}

stop_event = threading.Event()

class Ride(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float = Field(ge=0)
    

def prepare_features(ride):
    # Must match training. Ints keep 43.0 from becoming "43.0_151.0".
    return {"PU_DO": "%s_%s" % (int(ride["PULocationID"]), int(ride["DOLocationID"])),
            "trip_distance": float(ride["trip_distance"])}

def predict(features_list):
    return [float(p) for p in model.predict(features_list)]

def score_batch(msgs, producer):
    """Score one batch of Kafka messages. Returns (scored, skipped). Does not flush or commit."""
    rides, feats, skipped = [], [], 0
    for m in msgs:
        if m.error():
            raise RuntimeError(f"Kafka error: {m.error()}")
        try:
            ride = json.loads(m.value())
            feats.append(prepare_features(ride))
            rides.append(ride)
        except (ValueError, KeyError, TypeError) as exc:
            skipped += 1
            log.warning("skipping malformed message: %s", exc)
    if rides:
        for ride, pred in zip(rides, predict(feats)):
            producer.produce(OUT_TOPIC, key=str(ride.get("ride_id", "")),
                             value=json.dumps({"ride_id": ride.get("ride_id"),
                                               "duration": pred,
                                               "model_version": RUN_ID}))
    return len(rides), skipped

def stream_loop():
    from confluent_kafka import Consumer, Producer

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "duration-scorer",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    producer = Producer({"bootstrap.servers": BOOTSTRAP})
    consumer.subscribe([IN_TOPIC])
    log.info("streaming: consuming %s, producing %s", IN_TOPIC, OUT_TOPIC)

    try:
        while not stop_event.is_set():
            msgs = consumer.consume(num_messages=BATCH_SIZE, timeout=1.0)
            if not msgs:
                continue

            scored, _ = score_batch(msgs, producer)
            producer.flush(30)
            consumer.commit(asynchronous=False)
            stats["streamed"] += scored
    except Exception as exc:
        log.exception("stream loop stopped")
        stats["error"] = str(exc)
    finally:
        consumer.close()

@asynccontextmanager
async def lifespan(_app):
    thread = None
    if STREAM:
        thread = threading.Thread(target=stream_loop,daemon=True)
        thread.start()
    yield
    stop_event.set()
    if thread:
        thread.join(timeout=10)

app = FastAPI(title="duration-prediction", lifespan=lifespan)

@app.post('/predict')
def predict_endpoint(ride:Ride):
    pred = predict([prepare_features(ride.model_dump())])[0]
    return {'duration':pred}


@app.get('/health')
def health():
    return {"model_version": RUN_ID,
            "streaming": STREAM and stats["error"] is None,
            "streamed": stats["streamed"],
            "error": stats["error"]}
