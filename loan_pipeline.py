import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from datetime import datetime
import dill
import json
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import catboost as cb

# Класс для загрузки данных
class LoadData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        hits_df = pd.read_feather('/Users/yszam/PyCharm_Projects/car_leasing/data/hits.feather')
        sessions_df = pd.read_feather('/Users/yszam/PyCharm_Projects/car_leasing/data/sessions.feather')
        return hits_df, sessions_df

# Класс для очистки данных
class CleanData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        hits_df, sessions_df = X

        # Очистка данных о кликах
        hits_df = hits_df.drop(columns=[
            'hit_time', 'hit_referer', 'event_label', 'event_value', 'hit_date',
            'hit_type', 'hit_number', 'hit_page_path'], errors='ignore')

        # Очистка данных о сессиях
        sessions_df = sessions_df.drop(columns=[
            'device_model', 'utm_keyword', 'device_os', 'visit_time', 'visit_date',
            'visit_number', 'utm_medium', 'device_category', 'device_brand'], errors='ignore')
        sessions_df = sessions_df.fillna(sessions_df.mode().iloc[0])

        return hits_df, sessions_df

# Класс для создания признаков
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        hits_df, sessions_df = X

        # Определение целевых действий
        target_actions = [
            'sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
            'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
            'sub_submit_success', 'sub_car_request_submit_click'
        ]

        # Создание целевой переменной
        hits_df['target'] = hits_df['event_action'].apply(lambda x: 1 if x in target_actions else 0)

        # Определение целевой переменной для клиентов
        client_target = hits_df.groupby('session_id')['target'].max().reset_index()
        client_sessions = sessions_df.merge(client_target, on='session_id', how='left')
        client_sessions['target'] = client_sessions.groupby('client_id')['target'].transform('max').fillna(0).astype(int)
        client_sessions = client_sessions.drop_duplicates(subset='client_id')

        # Функция для вычисления коэффициента конверсии
        def calculate_cr(df, groupby_column):
            cr = df.groupby(groupby_column)['target'].mean().reset_index()
            cr.columns = [groupby_column, 'conversion_rate']
            return cr

        # Функция для присвоения коэффициентов конверсии к данным
        def map_conversion_rates(df, conversion_dict):
            for col, cr_dict in conversion_dict.items():
                df[f'cr_{col}'] = df[col].map(cr_dict).fillna('error')
            return df

        columns_to_encode = [
            'utm_source', 'utm_campaign', 'geo_country', 'utm_adcontent',
            'device_browser', 'device_screen_resolution', 'geo_city']

        conversion_dict = {col: dict(zip(*calculate_cr(client_sessions, col).values.T)) for col in columns_to_encode}
        client_sessions = map_conversion_rates(client_sessions, conversion_dict)

        # Сохранение словаря для будушей работы модели
        with open('/Users/yszam/PyCharm_Projects/car_leasing/models/conversion_dict.json', 'w') as f:
            json.dump(conversion_dict, f)

        # Удаление ненужных колонок
        client_sessions = client_sessions.drop(columns=['session_id', 'client_id'] + columns_to_encode, errors='ignore')

        # Балансировка данных
        balanced = pd.concat([client_sessions[client_sessions['target'] == 1].sample(n=43839, random_state=42),
                              client_sessions[client_sessions['target'] == 0].sample(n=43839, replace=True, random_state=42)])
        return balanced

def main():
    try:
        print('Conversion Prediction Pipeline')

        # Определение этапов пайплайна
        pipeline_steps = [
            ('load_data', LoadData()),
            ('clean_data', CleanData()),
            ('feature_engineering', FeatureEngineering())
        ]

        pipeline = Pipeline(steps=pipeline_steps)

        # Выполнение пайплайна
        balanced = pipeline.fit_transform(None)

        # Проверка данных на наличие строковых значений
        for col in balanced.columns:
            if balanced[col].dtype == 'object':
                print(f"Column '{col}' contains non-numeric data")
                print(balanced[col].unique())
                raise ValueError(f"Non-numeric data found in column '{col}'")

        # Разделение данных на обучающую и тестовую выборки
        X = balanced.drop(['target'], axis=1)
        y = balanced['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Инициализация модели CatBoost с заданными параметрами
        model = cb.CatBoostClassifier(
            random_strength=10,
            learning_rate=0.05,
            l2_leaf_reg=7,
            iterations=500,
            depth=8,
            bagging_temperature=0.1,
            loss_function='Logloss',
            eval_metric='AUC',
            early_stopping_rounds=50,
            verbose=100
        )

        # Создание финального пайплайна
        final_pipeline = Pipeline(steps=[
            ('classifier', model)
        ])

        print("Pipeline steps:")
        print(final_pipeline)

        # Проведение кросс-валидации
        print("Performing cross-validation...")
        cv_data = cb.Pool(data=X_train, label=y_train)
        cv_scores = cb.cv(
            params=model.get_params(),
            pool=cv_data,
            fold_count=5,
            plot=False
        )
        mean_auc = cv_scores['test-AUC-mean'].iloc[-1]
        print(f"Mean ROC-AUC from CV: {mean_auc:.5f}")

        # Обучение модели на тестовом наборе данных
        final_pipeline.fit(X_test, y_test)

        # Прогноз вероятностей
        y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

        # Применение оптимального порога, по показателям f1-score и Best Balanced Accuracy
        best_threshold = 0.35
        y_pred = (y_pred_proba >= best_threshold).astype(int)

        # Расчет ROC-AUC с использованием вероятностей
        test_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC from unseen, test size: {test_auc:.5f}")

        # Оценка дополнительных метрик
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        specificity = tn / (tn + fp)

        print(f"Confusion matrix with threshold {best_threshold:.2f}:\n{cm}")
        print(f"Precision with the threshold: {precision:.3f}")
        print(f"Recall with the threshold: {recall:.3f}")
        print(f"F1 Score with the threshold: {f1:.3f}")
        print(f"Specificity with the threshold: {specificity:.3f}")

        # Обучение модели на всем наборе данных
        print("Fitting the pipeline on entire data...")
        final_pipeline.fit(X, y)

        # Оценка модели
        y_pred_proba = final_pipeline.predict_proba(X)[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)
        print(f"ROC-AUC for CatBoostClassifier: {roc_auc:.5f}")

        # Сохранение модели, если ROC-AUC выше порога
        if roc_auc >= 0.65:
            model_path = '/Users/yszam/PyCharm_Projects/car_leasing/models/catboost_model.pkl'
            with open(model_path, 'wb') as f:
                dill.dump({
                    'model': final_pipeline,
                    'metadata': {
                        'name': 'Conversion prediction model',
                        'author': 'Yaroslav Zamuruyev',
                        'version': 2.1,
                        'date': datetime.now(),
                        'type': type(final_pipeline.named_steps['classifier']).__name__,
                        'roc_auc': roc_auc
                    }
                }, f)
            print(f"Model saved to {model_path}")
        else:
            print("ROC-AUC score is below 0.65, model not saved.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
