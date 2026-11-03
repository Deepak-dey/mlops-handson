import os
import pickle
import mlflow
from flask import Flask, request, jsonify

RUN_ID = os.getenv("RUN_ID")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))

logged_model = f"runs:/{RUN_ID}/model"

model = mlflow.sklearn.load_model(logged_model)


# with open("lin_reg.bin", "rb") as f_in:
#     (dv,model) = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s'%(ride['PULocationID'],ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict([features])
    return float(preds[0])


app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    
    features = prepare_features(ride)
    pred = predict(features)
    
    result = {
        'duration': pred
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
