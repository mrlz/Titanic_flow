import pandas as pd
import xgboost as xgb
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently import ColumnMapping
import pickle
from data_preprocessing import preprocess_samples

model_path = './models/'

model = xgb.XGBClassifier()

model.load_model(model_path + 'xgboost_titanic.json')


with open(model_path + 'scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open(model_path + 'median.pkl', 'rb') as file:
    median = pickle.load(file)

data_path = './dataset/'

train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test_augmented.csv')

X_train, y_train, _, _ = preprocess_samples(train, has_y=True, sample_scaler=scaler, sample_median=median)
X_test, y_test, _, _ = preprocess_samples(test, has_y=True, sample_scaler=scaler, sample_median=median)

X_train['prediction'] = model.predict(X_train.values)
X_test['prediction'] = model.predict(X_test.values)

target = 'Result'
prediction = 'Prediction'

data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataQualityPreset(),
    TargetDriftPreset()
])

data_drift_report.run(reference_data=X_train, current_data=X_test)

data_drift_report.save_html("test_drift.html")