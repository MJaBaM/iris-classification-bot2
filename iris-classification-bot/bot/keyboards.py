# bot/keyboards.py
from telebot import types

def main_menu():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    buttons = [
        "Классифицировать цветок",
        "Статистика",
        "Загрузить CSV",
        "Помощь",
        "Информация о модели"
    ]
    markup.add(*buttons)
    return markup

def cancel_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("Отмена")
    return markup
