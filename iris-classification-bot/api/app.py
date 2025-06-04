from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging
import os

app = Flask(__name__)

logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_PATH = 'iris_model.pkl'
FEATURE_INFO_PATH = 'feature_info.pkl'

if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_INFO_PATH)):
    raise FileNotFoundError('Модель и/или информация о признаках не найдены. Запустите model.py для тренировки модели.')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(FEATURE_INFO_PATH, 'rb') as f:
    feature_info = pickle.load(f)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        if not data or 'features_list' not in data:
            return jsonify({'error': 'Не переданы данные для предсказания'}), 400

        features_list = np.array(data['features_list'], dtype=float)
        
        # Проверка размерности
        if features_list.ndim != 2 or features_list.shape[1] != 4:
            return jsonify({'error': 'Неверный формат данных. Ожидается список из 4 признаков для каждого образца'}), 400

        # Валидация всех данных
        for i, features in enumerate(features_list):
            for j, (feat_value, (min_val, max_val)) in enumerate(zip(features, feature_info['feature_ranges'].values())):
                if not (min_val <= feat_value <= max_val):
                    return jsonify({'error': f'Строка {i+1}: значение признака {feature_info["feature_names"][j]} ({feat_value}) вне диапазона ({min_val} - {max_val})'}), 400

        predictions = model.predict(features_list)
        class_names = [feature_info['target_names'][p] for p in predictions]

        return jsonify({
            'classes': [int(p) for p in predictions],
            'class_names': class_names
        })

    except Exception as e:
        logging.error(f'Batch prediction error: {str(e)}')
        return jsonify({'error': 'Внутренняя ошибка сервера'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
