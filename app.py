from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import xgboost as xgb
from contextlib import asynccontextmanager
import pickle
from data_preprocessing import preprocess_samples, init_text_model
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Literal
import logging


class InputData(BaseModel):
    PassengerId: int | None = 1
    Pclass: int = Field(gt=0, lt=4)
    Name: str | None = ''
    Sex: Literal['male', 'female']
    Age: float | None = None
    SibSp: int | None = 0
    Parch: int | None = 0
    Ticket: str | None = ''
    Fare: float | None = 0
    Cabin: str | None = ''
    Embarked: Literal['S', 'C', 'Q'] | None = 'S'




@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all necessary APP resources upon start, such as
    the model to compute the embeddings, and the scaler
    and median.
    """
    model_path = './models/'
    app.state.loaded_model = xgb.XGBClassifier()
    app.state.loaded_model.load_model('./models/xgboost_titanic.json')
    
    init_text_model()

    logging.basicConfig(filename='example_log.log',level=logging.DEBUG,format='%(asctime)s:%(levelname)s:%(message)s')
    app.state.logger = logging.getLogger('uvicorn.error')

    with open(model_path + 'scaler.pkl', 'rb') as file:
        app.state.scaler = pickle.load(file)
    
    with open(model_path + 'median.pkl', 'rb') as file:
        app.state.median = pickle.load(file)

    yield
    del app.state.logger
    del app.state.loaded_model
    del app.state.scaler
    del app.state.median


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root():
    return {"health_check": "OK", "model_version": 1}

@app.post("/predict", response_model = list[int])
async def predict(input_data: list[InputData]):
        """
        Takes a list of passengers as input. 

        Passengers are defined by the features:
        PassengerId: An int. Optional.
        Pclass: An int. 0, 1 or 2.
        Name: A string. Can be empty. Optional.
        Sex: A string. Must be male or female.
        Age: A float. Optional.
        SibSp: An int. Optional.
        Parch: An int. Optional.
        Ticket: A string. Can be empty. Optional.
        Fare: A float. Optional.
        Cabin: A string. Can be empty. Optional.
        Embarked: A string. Must be S, Q or C.

        Computes prediction regarding survival.
        """
        app.state.logger.info("Received a request for a prediction.")
        input_list_of_dicts = [element.dict() for element in input_data]
        batch_processing_samples, batch_y, current_scaler, current_median = preprocess_samples(pd.DataFrame(input_list_of_dicts), False, sample_scaler=app.state.scaler, sample_median=app.state.median)
        pred = app.state.loaded_model.predict(batch_processing_samples)
        ans = [int(x) for x in pred]
        return ans


