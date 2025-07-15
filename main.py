import logging
import mlflow
import mlflow.sklearn
import os 
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import set_config
import xgboost as xgb
from sklearn.metrics import accuracy_score, matthews_corrcoef, make_scorer, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pickle
from data_preprocessing import prepare_data, features_and_labels, fill_missing_age, compute_text_features, join_and_scale, preprocess_samples

def train_and_log():
    with mlflow.start_run(log_system_metrics=True) as run:
        mlflow.set_tag('Dev', 'mrlz')

        # Data import
        data_path = './dataset/'
        model_path = './models/'

        os.makedirs(data_path, exist_ok=True)

        train_data = data_path + 'train.csv'
        test_data = data_path + 'test_augmented.csv'


        training = pd.read_csv(train_data)
        test = pd.read_csv(test_data)
        logging.info("Data correctly loaded.")
        ###########################################


        ###################### Data split
        X, y = features_and_labels(training)


        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=23)

        for train_index, val_index in sss.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        logging.info("Data correctly split.")
        
        sample_scaler = None
        sample_median = None

        X_train, y_dummy, sample_scaler, sample_median = preprocess_samples(X_train, False, sample_scaler, sample_median)
        X_val, y_dummy, sample_scaler, sample_median = preprocess_samples(X_val, False, sample_scaler, sample_median)


        logging.info("Data correctly cleaned.")


        ################### Define classifier
        xgb_clf = xgb.XGBClassifier()
        logging.info("Classifier correctly defined.")
        ################### Grid search hyperparameters

        #Already explored all of these, so we can skip ahead. 
        #But do explore other configurations, if you so wish.
        # param_grid = {
        #     'objective': ['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic'],
        #     'max_depth': [3,5,7],
        #     'learning_rate':[0.1, 0.2, 0.3, 0.4],
        #     'subsample': [0.6, 1],
        #     'gamma' : [0, 1],
        #     'lambda' : [1, 10]
        # }

        param_grid = {'gamma': [0], 'lambda': [10], 'learning_rate': [0.1], 'max_depth': [5], 'objective': ['reg:squaredlogerror'], 'subsample': [0.6]}

        mcc_scorer = make_scorer(matthews_corrcoef)
        grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, scoring=mcc_scorer, n_jobs=-1)
        grid_search.fit(X_train.values,y_train.values)
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)

        logging.info("Grid search correctly executed.")
        ################## Fit best classifier

        final_clf = xgb.XGBClassifier(**grid_search.best_params_)

        final_scaler = None
        final_median = None

        X, y_dummy, final_scaler, final_median = preprocess_samples(X, False, final_scaler, final_median)
        X_test, y_test, final_scaler, final_median = preprocess_samples(test, True, final_scaler, final_median) 
        final_clf.fit(X, y.values) 


        logging.info("Final classifier correctly trained.")


        model_name = "Titanic classifier" 

        model_params = grid_search.best_params_
        mlflow.log_params(model_params)
        
        ################# Predict over sets
        train_predictions  = final_clf.predict(X.values)
        test_predictions = final_clf.predict(X_test.values)

        train_acc = accuracy_score(y.values, train_predictions)
        test_acc = accuracy_score(y_test.values, test_predictions)
        logging.info("Predictions correctly executed.")

        train_mcc = matthews_corrcoef(y.values, train_predictions)
        test_mcc = matthews_corrcoef(y_test.values, test_predictions)

        mlflow.log_metric("Best classifier train accuracy", train_acc)
        print(f"Train accuracy {train_acc}")
        mlflow.log_metric("Best classifier test accuracy", test_acc)
        print(f"Test accuracy {test_acc}")
        mlflow.log_metric("Best mcc in grid search", grid_search.best_score_)
        print(f"Best mcc in grid search {grid_search.best_score_}")

        print(f"Best classifier train MCC {train_mcc}")
        mlflow.log_metric("Best classifier train MCC", train_mcc)
        print(f"Best classifier test MCC {test_mcc}")
        mlflow.log_metric("Best classifier test MCC", test_mcc)


        conf_matrix = confusion_matrix(y_test.values ,test_predictions)
        print(conf_matrix)

        class_names = ["Dies", "Survives"]
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='horizontal')
        plt.title("Confusion Matrix - Titanic")
        plt.savefig("Conf_matrix.png")
        mlflow.log_artifact("Conf_matrix.png")
        mlflow.log_dict(np.array(conf_matrix).tolist(), "Confusion_matrix dictionary")

        ################ Save best model
        os.makedirs(model_path, exist_ok=True)
        final_clf.save_model(model_path + 'xgboost_titanic.json')

        with open(model_path + 'median.pkl', 'wb') as file:
            pickle.dump(final_median, file)
        
        with open(model_path + 'scaler.pkl', 'wb') as file:
            pickle.dump(final_scaler, file)

        mlflow.log_artifacts(model_path)
        logging.info("Model parameters correctly saved.")

        #############################

        mlflow.set_tag('preprocessing', 'OneHotEncoder, StandardScaler, Age fill in, Embarked fill in.')
        mlflow.set_tag('feature_construction', 'Sentence Transformer Embeddings.')

        logging.info("MLflow tracking completed successfully")

        
if __name__ == "__main__":
    logging.basicConfig(filename='example_log.log',level=logging.DEBUG,format='%(asctime)s:%(levelname)s:%(message)s')
    set_config(transform_output="pandas")
    mlflow.xgboost.autolog()
    mlflow_logger = logging.getLogger("mlflow")
    mlflow_logger.setLevel(logging.INFO)
    train_and_log()
