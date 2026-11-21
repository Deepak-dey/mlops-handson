FROM python:3.12-slim

# The model must be loaded with the same scikit-learn and MLflow versions that trained and logged it.
# Check them in your training environment:  pip show scikit-learn mlflow
ARG SKLEARN_VERSION
ARG MLFLOW_VERSION
RUN test -n "$SKLEARN_VERSION" && test -n "$MLFLOW_VERSION" || (echo "ERROR: set SKLEARN_VERSION and MLFLOW_VERSION in .env" && exit 1)

WORKDIR /app

RUN pip install --no-cache-dir confluent-kafka fastapi "uvicorn[standard]" \
    "scikit-learn==${SKLEARN_VERSION}" "mlflow==${MLFLOW_VERSION}"

COPY service.py ./

# the compose file sets the entrypoint (uvicorn service:app)
ENTRYPOINT ["python"]
