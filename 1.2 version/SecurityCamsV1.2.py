import telebot
import time
import requests
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model

bot = telebot.TeleBot('')
url_person = ''

# Папка для временных файлов
temp_dir = 'temp_photos'
os.makedirs(temp_dir, exist_ok=True)

# Глобальные переменные для управления видеопотоком
video_processing = False
current_chat_id = None

model = load_model('final.keras')


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Здравствуйте! Добро пожаловать в систему камер видеонаблюдения: SecurityCams!")
    time.sleep(1.5)
    bot.send_message(message.chat.id,
                     "Наш телеграм бот поможет предотвращать инциденты, связанные с уголовным деянием.")
    time.sleep(1.5)
    bot.send_message(message.chat.id, "Для того чтобы узнать функционал бота, введите команду: /help")


@bot.message_handler(commands=['help'])
def send_help(message):
    help_text = """
Доступные команды:
/setcamera - Добавить камеру для анализа;
/cameras - Просмотреть подключенный камеры;
/startanalyzecamera - Начать анализ камеры;
/stopanalyzecamera - Закончить анализ камеры; 
"""
    bot.send_message(message.chat.id, help_text)


@bot.message_handler(commands=['setcamera'])
def start_analyze_camera(message):
    bot.send_message(message.chat.id, r'Введите URL камеры видеонаблюдения. К примеру http://{ip}/snapshot.jpg')
    bot.register_next_step_handler(message, main_analyze_camera)


def main_analyze_camera(message):
    global url_person
    url = message.text
    if message.text == '/Отмена':
        return
    try:
        url_person = url
        bot.send_message(message.chat.id, "Камера успешно добавлена!")
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка {e}. Попробуйте еще раз, либо напишите \"Отмена\" для выхода")
        bot.register_next_step_handler(message, main_analyze_camera)


@bot.message_handler(commands=['cameras'])
def save_cameras(message):
    global url_person
    bot.send_message(message.chat.id, f"Ваши камеры: {url_person}")


@bot.message_handler(commands=['stopanalyzecamera'])
def stop_analyze(message):
    global video_processing
    video_processing = False
    bot.send_message(message.chat.id, "Анализ камеры завершён!")


@bot.message_handler(commands=['startanalyzecamera'])
def analyze_camera(message):
    global url_person, video_processing, current_chat_id
    if url_person == '':
        bot.send_message(message.chat.id, "У вас нет ни одной подключенной камеры")
        return

    current_chat_id = message.chat.id
    video_processing = True
    frame_counter = 0
    analysis_interval = 5  # Анализировать каждый 5-й кадр

    bot.send_message(message.chat.id, "Начинаю анализ видеопотока...")

    while video_processing:
        try:
            # Загрузка изображения с камеры
            response = requests.get(url_person, timeout=10)
            if response.status_code != 200:
                raise Exception(f"Ошибка получения кадра: HTTP {response.status_code}")

            frame_counter += 1
            if frame_counter % analysis_interval != 0:
                time.sleep(0.1)  # Небольшая задержка между кадрами
                continue

            # Сохраняем кадр для анализа
            frame_path = os.path.join(temp_dir, 'current_frame.jpg')
            with open(frame_path, 'wb') as f:
                f.write(response.content)

            # Анализируем кадр
            result, processed_img = analyze_image(frame_path)

            if "ОРУЖИЕ" in result:
                # Если обнаружено оружие, отправляем кадр с результатом
                result_path = os.path.join(temp_dir, 'result_frame.jpg')
                cv2.imwrite(result_path, processed_img)

                with open(result_path, 'rb') as photo:
                    bot.send_photo(current_chat_id, photo, caption=result)
                    bot.send_message(current_chat_id, "Внимание! Обнаружено оружие!")

        except Exception as e:
            bot.send_message(current_chat_id, f"Ошибка обработки кадра: {str(e)}")
            time.sleep(5)  # Пауза при ошибке

    bot.send_message(current_chat_id, "Анализ видеопотока остановлен")


def analyze_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")

        # Сохраняем оригинальное изображение для визуализации
        original_img = img.copy()

        # Подготовка изображения для модели
        img = cv2.resize(img, (256, 256)) / 255.0
        img = np.expand_dims(img, axis=0)

        # Предсказание
        prediction = model.predict(img)

        # Визуализация результата на оригинальном изображении
        if prediction[0][0] > 0.65:
            cv2.putText(original_img,
                        f"WEAPON DETECTED: {prediction[0][0] * 100:.2f}%",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)
            result = f"Обнаружено ОРУЖИЕ с вероятностью {prediction[0][0] * 100:.2f}%!!!"
        else:
            cv2.putText(original_img, "No threats detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            result = "Опасных предметов не обнаружено"

        return result, original_img

    except Exception as e:
        return f"Ошибка анализа изображения: {str(e)}", None


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        image_path = os.path.join(temp_dir, 'photo.jpg')
        with open(image_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        result, processed_img = analyze_image(image_path)

        if processed_img is not None:
            result_path = os.path.join(temp_dir, 'processed_photo.jpg')
            cv2.imwrite(result_path, processed_img)

            with open(result_path, 'rb') as photo:
                bot.send_photo(message.chat.id, photo, caption=result)
        else:
            bot.reply_to(message, result)

    except Exception as e:
        bot.reply_to(message, f'Ошибка обработки фото: {str(e)}')


bot.polling(none_stop=True, interval=0)