import pandas as pd
import dill
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Union

app = FastAPI()

model_data = None
pipeline = None
threshold = 0.35  # Пороговое значение для принятия решения о классе

# Разрешить все источники (по необходимости можно настроить)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/status')
def status():
    return "I'm alive!"

@app.get('/version')
def version():
    if model_data is None:
        raise HTTPException(status_code=500, detail="Model data is not loaded")
    return model_data['metadata']

# Обработчик события запуска приложения
@app.on_event("startup")
async def startup_event():
    global model_data, pipeline
    model_path = '/Users/yszam/Pycharm_Projects/car_leasing/models/catboost_model.pkl'
    with open(model_path, 'rb') as f:
        model_data = dill.load(f)  # Загружаем данные модели
        pipeline = model_data['model']  # Извлекаем модель из данных

class Observations(BaseModel):
    # Определение модели данных для наблюдений
    session_id: Dict[str, str]
    client_id: Dict[str, str]
    visit_date: Dict[str, str]
    visit_time: Dict[str, str]
    visit_number: Dict[str, int]
    utm_source: Dict[str, str]
    utm_medium: Dict[str, str]
    utm_campaign: Optional[Dict[str, Optional[str]]] = None
    utm_adcontent: Optional[Dict[str, Optional[str]]] = None
    utm_keyword: Optional[Dict[str, Optional[str]]] = None
    device_category: Dict[str, str]
    device_os: Optional[Dict[str, Optional[str]]] = None
    device_brand: Optional[Dict[str, Optional[str]]] = None
    device_model: Optional[Dict[str, Optional[str]]] = None
    device_screen_resolution: Dict[str, str]
    device_browser: Dict[str, str]
    geo_country: Dict[str, str]
    geo_city: Dict[str, str]

@app.post("/predict")
async def predict(observations: Observations):
    try:
        # Преобразуем данные в DataFrame
        data_dict = {key: list(value.values()) for key, value in observations.dict().items()}
        input_data = pd.DataFrame(data_dict)

        # Предобработка данных
        input_data = preprocess_input_data(input_data)

        # Выполнение предсказаний
        prediction_proba = pipeline.predict_proba(input_data.drop(columns=['session_id', 'client_id'], axis=1))[:, 1]
        predictions = (prediction_proba >= threshold).astype(int)

        # Форматируем результаты
        results = []
        for pred, proba, session_id, client_id in zip(predictions, prediction_proba, input_data['session_id'], input_data['client_id']):
            results.append({
                'session_id': session_id,
                'client_id': client_id,
                'prediction': int(pred),
                'probability': float(proba)
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def preprocess_input_data(input_data):
    # Удаление ненужных столбцов
    input_data = input_data.drop(columns=[
            'device_model', 'utm_keyword', 'device_os', 'visit_time', 'visit_date',
            'visit_number', 'utm_medium', 'device_category', 'device_brand'], errors='ignore')

    # Загрузка словарей конверсии
    cr_dict_path = '/Users/yszam/Pycharm_Projects/car_leasing/models/conversion_dict.json'
    with open(cr_dict_path, 'r') as f:
        conversion_dict = pd.read_json(f)

    def map_conversion_rates(df, conversion_dict):
        # Присваивание коэффициентов конверсии
        for col, cr_dict in conversion_dict.items():
            df[f'cr_{col}'] = df[col].map(cr_dict).fillna(0)
        return df

    columns_to_encode = [
            'utm_source', 'utm_campaign', 'geo_country', 'utm_adcontent',
            'device_browser', 'device_screen_resolution', 'geo_city']

    input_data = map_conversion_rates(input_data, conversion_dict)

    # Удаление столбцов после назначения коэффициентов конверсии
    input_data = input_data.drop(columns=columns_to_encode, errors='ignore')

    return input_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
