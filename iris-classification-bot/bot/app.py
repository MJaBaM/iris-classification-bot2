# bot/app.py
import telebot
from telebot import types
import requests
import sqlite3
import logging
from datetime import datetime
import os
from keyboards import main_menu, cancel_markup
import pandas as pd
import io
import time

# Конфиги
API_URL = "http://localhost:5000/predict"
BATCH_API_URL = "http://localhost:5000/batch_predict"
MAX_CSV_ROWS = 100  # Максимальное количество строк для обработки

bot = telebot.TeleBot("токен")
os.makedirs('temp_files', exist_ok=True)

# Логирование
logging.basicConfig(
    filename='bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('iris_bot.db')
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (user_id INTEGER PRIMARY KEY,
                  username TEXT,
                  first_name TEXT,
                  last_name TEXT,
                  registration_date TEXT,
                  last_active TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS requests
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  timestamp TEXT,
                  features TEXT,
                  prediction TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS batch_requests
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  timestamp TEXT,
                  file_name TEXT,
                  rows_processed INTEGER,
                  FOREIGN KEY(user_id) REFERENCES users(user_id))''')

    conn.commit()
    conn.close()

init_db()

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start(message):
    user = message.from_user
    conn = sqlite3.connect('iris_bot.db')
    c = conn.cursor()
    c.execute('INSERT OR IGNORE INTO users(user_id, username, first_name, last_name, registration_date, last_active) VALUES (?, ?, ?, ?, ?, ?)',
              (user.id, user.username, user.first_name, user.last_name, datetime.now().isoformat(), datetime.now().isoformat()))
    conn.commit()
    conn.close()

    bot.send_message(message.chat.id, 
                    "🌸 *Бот для классификации цветов ириса* 🌸\n\n"
                    "Отправьте 4 числа через запятую (длина и ширина чашелистика, длина и ширина лепестка)\n"
                    "Пример: _5.1, 3.5, 1.4, 0.2_\n\n"
                    "Или загрузите CSV-файл с данными для пакетной обработки.",
                    reply_markup=main_menu(),
                    parse_mode='Markdown')

# Обработчик команды /cancel
@bot.message_handler(commands=['cancel'])
def cancel(message):
    bot.send_message(message.chat.id, "Текущая операция отменена.", reply_markup=main_menu())

# Обработчик команды "Статистика"
@bot.message_handler(func=lambda message: message.text == "Статистика")
def show_statistics(message):
    conn = sqlite3.connect('iris_bot.db')
    c = conn.cursor()
    
    # Получаем общее количество запросов пользователя
    c.execute("SELECT COUNT(*) FROM requests WHERE user_id = ?", (message.from_user.id,))
    single_count = c.fetchone()[0]
    
    # Получаем количество пакетных обработок
    c.execute("SELECT COUNT(*), SUM(rows_processed) FROM batch_requests WHERE user_id = ?", (message.from_user.id,))
    batch_count, total_rows = c.fetchone()
    total_rows = total_rows or 0
    
    conn.close()
    
    stats_text = (
        f"📊 *Ваша статистика:*\n\n"
        f"• Одиночных классификаций: {single_count}\n"
        f"• Пакетных обработок: {batch_count}\n"
        f"• Всего классифицировано цветков: {single_count + total_rows}"
    )
    
    bot.send_message(message.chat.id, stats_text, parse_mode='Markdown')

# Обработчик команды "Помощь"
@bot.message_handler(func=lambda message: message.text == "Помощь")
def show_help(message):
    help_text = (
        "🌸 *Помощь по боту* 🌸\n\n"
        "Я могу классифицировать цветы ириса на 3 вида:\n"
        "- Setosa\n"
        "- Versicolor\n"
        "- Virginica\n\n"
        "*Как использовать:*\n"
        "1. Отправьте 4 числа через запятую:\n"
        "   `длина_чашелистика, ширина_чашелистика, длина_лепестка, ширина_лепестка`\n"
        "   Пример: _5.1, 3.5, 1.4, 0.2_\n\n"
        "2. Или загрузите CSV-файл с данными для пакетной обработки (макс. 100 строк)\n\n"
        "*Доступные команды:*\n"
        "- Статистика: ваша активность\n"
        "- Информация о модели: технические детали\n"
        "- Загрузить CSV: пакетная обработка данных"
    )
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown')

# Обработчик команды "Информация о модели"
@bot.message_handler(func=lambda message: message.text == "Информация о модели")
def show_model_info(message):
    model_info = (
        "🔍 *Информация о модели*\n\n"
        "• Алгоритм: Random Forest (100 деревьев)\n"
        "• Данные: Iris dataset (150 samples)\n"
        "• Точность: ~96%\n"
        "• Признаки:\n"
        "  1. Длина чашелистика (см)\n"
        "  2. Ширина чашелистика (см)\n"
        "  3. Длина лепестка (см)\n"
        "  4. Ширина лепестка (см)\n\n"
        "Модель проверяет, что входные значения находятся в допустимых диапазонах."
    )
    bot.send_message(message.chat.id, model_info, parse_mode='Markdown')

# Обработчик текста (предсказание)
@bot.message_handler(func=lambda message: True, content_types=['text'])
def classify(message):
    # Пропускаем команды из меню
    if message.text in ["Статистика", "Помощь", "Информация о модели", "Загрузить CSV"]:
        return
        
    try:
        features = [float(x.strip()) for x in message.text.split(',')]
        if len(features) != 4:
            bot.reply_to(message, "❌ Пожалуйста, отправьте ровно 4 числа через запятую.")
            return
    except:
        bot.reply_to(message, "❌ Не удалось распознать числа. Попробуйте ещё раз.")
        return

    try:
        # Показываем "ожидайте" сообщение
        wait_msg = bot.reply_to(message, "🔍 Анализирую цветок...", reply_markup=cancel_markup())
        
        res = requests.post(API_URL, json={'features': features})
        if res.status_code != 200:
            bot.delete_message(message.chat.id, wait_msg.message_id)
            bot.reply_to(message, f"❌ Ошибка API: {res.json().get('error', 'Неизвестная ошибка')}")
            return

        data = res.json()
        class_name = data['class_name']

        # Сохраняем запрос в БД
        conn = sqlite3.connect('iris_bot.db')
        c = conn.cursor()
        c.execute('INSERT INTO requests(user_id, timestamp, features, prediction) VALUES (?, ?, ?, ?)',
                  (message.from_user.id, datetime.now().isoformat(), str(features), class_name))
        conn.commit()
        conn.close()

        # Удаляем "ожидайте" и показываем результат
        bot.delete_message(message.chat.id, wait_msg.message_id)
        
        # Красивое оформление ответа
        response_text = (
            f"🌸 *Результат классификации* 🌸\n\n"
            f"• Вид: *{class_name}*\n"
            f"• Признаки:\n"
            f"  - Длина чашелистика: {features[0]:.1f} см\n"
            f"  - Ширина чашелистика: {features[1]:.1f} см\n"
            f"  - Длина лепестка: {features[2]:.1f} см\n"
            f"  - Ширина лепестка: {features[3]:.1f} см"
        )
        
        bot.send_message(message.chat.id, response_text, parse_mode='Markdown')

    except Exception as e:
        logging.error(f'Error processing message: {e}')
        bot.reply_to(message, "❌ Произошла ошибка при обработке запроса.")

# Обработчик команды "Загрузить CSV"
@bot.message_handler(func=lambda message: message.text == "Загрузить CSV")
def prompt_csv_upload(message):
    help_text = (
        "📁 *Загрузка CSV-файла*\n\n"
        "Отправьте CSV-файл с данными для классификации.\n"
        "Файл должен содержать 4 числовых столбца (в любом порядке):\n"
        "- sepal_length (длина чашелистика)\n"
        "- sepal_width (ширина чашелистика)\n"
        "- petal_length (длина лепестка)\n"
        "- petal_width (ширина лепестка)\n\n"
        "*Ограничения:*\n"
        "- Максимум 100 строк\n"
        "- Размер файла до 5MB\n\n"
        "Пример файла можно получить по команде /sample_csv"
    )
    bot.send_message(message.chat.id, help_text, parse_mode='Markdown', reply_markup=cancel_markup())

# Обработчик команды /sample_csv
@bot.message_handler(commands=['sample_csv'])
def send_sample_csv(message):
    # Создаем пример CSV файла в памяти
    sample_data = """sepal_length,sepal_width,petal_length,petal_width
5.1,3.5,1.4,0.2
6.7,3.0,5.2,2.3
4.9,3.1,1.5,0.1
7.0,3.2,4.7,1.4"""
    
    # Создаем файл в памяти
    csv_file = io.StringIO(sample_data)
    csv_file.name = "iris_sample.csv"
    
    # Отправляем файл пользователю
    bot.send_document(message.chat.id, (csv_file.name, csv_file.getvalue().encode()), 
                     caption="Пример CSV-файла для загрузки")









@bot.message_handler(content_types=['document'])
def handle_csv_file(message):
    try:
        # Проверка типа файла
        if not message.document.file_name.lower().endswith('.csv'):
            bot.reply_to(message, "❌ Пожалуйста, загрузите файл в формате CSV.")
            return

        # Отправка сообщения о начале обработки
        status_msg = bot.send_message(message.chat.id, "⏳ Начинаю обработку файла...")

        try:
            # Скачивание файла
            file_info = bot.get_file(message.document.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            
            # Чтение CSV
            try:
                df = pd.read_csv(io.BytesIO(downloaded_file))
            except Exception as e:
                bot.edit_message_text("❌ Ошибка чтения CSV файла", 
                                   chat_id=status_msg.chat.id, 
                                   message_id=status_msg.message_id)
                return

            # Проверка структуры
            required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                bot.edit_message_text(f"❌ Отсутствуют столбцы: {', '.join(missing_cols)}",
                                   chat_id=status_msg.chat.id,
                                   message_id=status_msg.message_id)
                return
                
            if len(df) > MAX_CSV_ROWS:
                bot.edit_message_text(f"❌ Слишком много строк (максимум {MAX_CSV_ROWS})",
                                   chat_id=status_msg.chat.id,
                                   message_id=status_msg.message_id)
                return

            # Обновление статуса
            try:
                bot.edit_message_text("⏳ Отправляю данные на анализ...",
                                   chat_id=status_msg.chat.id,
                                   message_id=status_msg.message_id)
            except:
                pass

            # Подготовка данных
            features = df[required_cols].values.tolist()
            
            # Проверка доступности API
            try:
                requests.get(API_URL, timeout=3)
            except:
                bot.edit_message_text("❌ Сервис анализа недоступен. Попробуйте позже.",
                                   chat_id=status_msg.chat.id,
                                   message_id=status_msg.message_id)
                return
            
            # Отправка в API
            try:
                response = requests.post(
                    BATCH_API_URL,
                    json={'features_list': features},
                    timeout=30
                )
                
                if response.status_code == 404:
                    bot.edit_message_text("❌ Функция пакетной обработки недоступна",
                                       chat_id=status_msg.chat.id,
                                       message_id=status_msg.message_id)
                    return
                
                response.raise_for_status()
                results = response.json()
            except requests.exceptions.RequestException as e:
                bot.edit_message_text(f"❌ Ошибка при анализе данных: {str(e)}",
                                    chat_id=status_msg.chat.id,
                                    message_id=status_msg.message_id)
                return
            
            # Обработка результатов
            df['prediction'] = results['class_names']
            
            # Сохранение в БД
            try:
                conn = sqlite3.connect('iris_bot.db')
                c = conn.cursor()
                c.execute('''INSERT INTO batch_requests 
                           (user_id, timestamp, file_name, rows_processed) 
                           VALUES (?, ?, ?, ?)''',
                         (message.from_user.id, datetime.now().isoformat(), 
                          message.document.file_name, len(df)))
                conn.commit()
                conn.close()
            except Exception as e:
                logging.error(f"Database error: {str(e)}")

            # Подготовка результата
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            # Удаление статус-сообщения
            try:
                bot.delete_message(chat_id=status_msg.chat.id, 
                                 message_id=status_msg.message_id)
            except:
                pass

            # Отправка результатов
            bot.send_message(message.chat.id, f"✅ Успешно обработано {len(df)} строк")
            bot.send_document(
                message.chat.id,
                (f"results_{message.document.file_name}", output.getvalue().encode()),
                caption="Результаты классификации"
            )
            
        except Exception as e:
            try:
                bot.edit_message_text(f"❌ Ошибка обработки: {str(e)}",
                                   chat_id=status_msg.chat.id,
                                   message_id=status_msg.message_id)
            except:
                bot.send_message(message.chat.id, f"❌ Ошибка обработки: {str(e)}")
            logging.error(f"CSV processing error: {str(e)}")
            
    except Exception as e:
        bot.reply_to(message, f"❌ Критическая ошибка: {str(e)}")
        logging.error(f"Critical error in handle_csv_file: {str(e)}")



if __name__ == '__main__':
    logging.info("Starting bot...")
    bot.infinity_polling()