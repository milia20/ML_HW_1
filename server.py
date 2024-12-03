from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import numpy as np
from fastapi.responses import Response
import pickle
import re
import json

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Maruti Swift Dzire VDI",
                    "year": 2014,
                    "km_driven": 14500,
                    "fuel": "Diesel",
                    "seller_type": "Individual",
                    "transmission": "Manual",
                    "owner": "First Owner",
                    "mileage": "23.4 kmpl",
                    "engine": "1197 CC",
                    "max_power": "74 bhp",
                    "torque": "190Nm@ 2000rpm",
                    "seats": 5.0
                }
            ]
        }
    }


class Items(BaseModel):
    objects: List[Item]


def load_model():
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    return model


def load_encoder():
    with open('hot_encoder', 'rb') as f:
        encoder = pickle.load(f)
    return encoder


def preprocess_torque(dataframe: pd.DataFrame) -> pd.DataFrame:
    def extract_torque_value(torque_str: str):
        if pd.isna(torque_str):
            return np.nan, np.nan
        match = re.match(
            r'^([\d\.\,]+)\s*(?:\w\w|\w\w\w|@|\s)[@\s|\sat\s|/]*([\d\-\,\~\@\s\+\-\/]*)(?:rpm|RPM|\(kgm@ rpm\)|\(NM@ rpm\)|\s)?$',
            torque_str.strip())
        if not match:
            print(f"Cannot parse string: '{torque_str}'")
            return np.nan, np.nan

        value = float(match.group(1).replace(',', '.'))
        rpm_range = match.group(2).strip().replace('@', '').replace(' ', '').replace("'", '')
        rpm_range = rpm_range.replace(',', '')

        if '+/-' in rpm_range:
            rpm_start, rpm_end = map(int, rpm_range.split('+/-'))
            rpm_end += rpm_start
        elif '-' in rpm_range or '~' in rpm_range:
            rpm_start, rpm_end = rpm_range.split('-') if '-' in rpm_range else rpm_range.split('~')
            rpm_start, rpm_end = int(rpm_start), int(rpm_end)
        elif ',' in rpm_range:
            rpm_start, rpm_end = map(int, rpm_range.split(','))
        else:
            rpm_start = int(rpm_range) if rpm_range else np.nan
            rpm_end = rpm_start

        rpm = rpm_end if rpm_end else rpm_start
        return value, rpm

    torque_df = dataframe['torque'].apply(extract_torque_value).apply(pd.Series)
    torque_df.rename(columns={0: 'torque_value', 1: 'max_torque_rpm'}, inplace=True)
    torque_df['max_torque_rpm'] = torque_df['max_torque_rpm'].astype("Int64")
    return torque_df


def clean_and_convert_to_float(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].str.replace(r'[^\d\.]', '', regex=True).replace(r'', '0.0', regex=True).astype(float)
    return df


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    df_new = preprocess_torque(data[['torque']])
    data = pd.concat([data, df_new], axis=1).drop(columns=['torque'])

    columns_to_clean = ['mileage', 'engine', 'max_power']
    data = clean_and_convert_to_float(data, columns_to_clean)

    data['mark'] = data.name.str.split().str[0]
    data['model'] = data.name.str.split().str[1]

    cat_col = ['fuel', 'seller_type', 'transmission', 'owner', 'mark', 'model', 'seats']
    X = data[cat_col]
    X.loc[:, 'seats'] = X['seats'].astype('Int64').astype('O')

    ohe = load_encoder()
    X = pd.concat([X, pd.DataFrame(ohe.transform(X[cat_col]).toarray())], axis=1).drop(cat_col, axis=1)

    return X


@app.post("/predict_item")
async def predict_item(item: Item) -> dict[str, float]:
    model = load_model()
    df = pd.DataFrame([json.loads(item.model_dump_json())])
    df = transform_data(df)
    predicted_cost = model.predict(df)[0]
    return {"predicted_cost": predicted_cost}


@app.post("/predict_items")
async def predict_items(items: List[Item]) -> dict[str, List[float]]:
    model = load_model()
    df = pd.DataFrame([json.loads(item.model_dump_json()) for item in items])
    df = transform_data(df)
    predicted_costs = model.predict(df).tolist()
    return {"predicted_costs": predicted_costs}


@app.post('/upload_csv')
async def upload_csv(file: UploadFile = File(...)) -> Response:
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents), encoding='utf8')
    df = df.replace('<NA>', np.nan).replace('nan', np.nan).ffill().fillna(0)

    items = [
        Item(
            name=row['name'],
            year=int(row['year']),
            km_driven=int(row.get('km_driven', 0)),
            fuel=str(row['fuel']),
            seller_type=str(row['seller_type']),
            transmission=str(row.get('transmission', 'Manual')),
            owner=str(row.get('owner', 'First Owner')),
            mileage=str(row.get('mileage', '')),
            engine=str(row['engine']),
            max_power=str(row['max_power']),
            torque=str(row.get('torque', '')),
            seats=float(row.get('seats', 0))
        )
        for _, row in df.iterrows()
    ]

    predictions = await predict_items(items)
    df['predicted_cost'] = predictions['predicted_costs']

    response = df.to_csv(index=False)
    return Response(content=response, media_type="text/csv")