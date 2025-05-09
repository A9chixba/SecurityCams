import telebot
import time
import logging
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from threading import Thread

bot = telebot.TeleBot('TOKEN')

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# Папка для временных файлов
temp_dir = 'temp_photos'
os.makedirs(temp_dir, exist_ok=True)

# Глобальные переменные для управления видеопотоком
video_processing = False
current_chat_id = None

try:
    model = load_model('final.keras')
    logger.info("Модель успешно загружена")
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")
    exit()


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
/startvideo - Начать анализ видео с камеры
/stopvideo - Остановить анализ видео
"""
    bot.send_message(message.chat.id, help_text)


@bot.message_handler(commands=['startvideo'])
def start_video_processing(message):
    global video_processing, current_chat_id
    if video_processing:
        bot.send_message(message.chat.id, "Видеоанализ уже запущен!")
        return

    video_processing = True
    current_chat_id = message.chat.id
    bot.send_message(message.chat.id, "Запускаю анализ видео с камеры...")

    # Запускаем обработку видео в отдельном потоке
    video_thread = Thread(target=process_video_stream)
    video_thread.start()


@bot.message_handler(commands=['stopvideo'])
def stop_video_processing(message):
    global video_processing
    video_processing = False
    bot.send_message(message.chat.id, "Анализ видео остановлен")


def process_video_stream():
    global video_processing, current_chat_id

    # Открываем камеру (0 - встроенная камера)
    cap = cv2.VideoCapture(0)

    # Проверяем, открыта ли камера
    if not cap.isOpened():
        bot.send_message(current_chat_id, "Ошибка: не удалось открыть камеру!")
        return

    frame_counter = 0
    analysis_interval = 5  # Анализировать каждый 5-й кадр

    while video_processing:
        ret, frame = cap.read()
        if not ret:
            bot.send_message(current_chat_id, "Ошибка: не удалось получить кадр с камеры!")
            break

        frame_counter += 1
        if frame_counter % analysis_interval != 0:
            continue

        try:
            # Сохраняем кадр для анализа
            frame_path = os.path.join(temp_dir, 'current_frame.jpg')
            cv2.imwrite(frame_path, frame)

            # Анализируем кадр
            result, processed_img = analyze_image(frame_path)

            if "ОРУЖИЕ" in result:
                # Если обнаружено оружие, отправляем кадр с результатом
                result_path = os.path.join(temp_dir, 'result_frame.jpg')
                cv2.imwrite(result_path, processed_img)

                with open(result_path, 'rb') as photo:
                    bot.send_photo(current_chat_id, photo, caption=result)

        except Exception as e:
            logger.error(f"Ошибка обработки кадра: {e}")

    # Освобождаем камеру
    cap.release()
    bot.send_message(current_chat_id, "Видеоанализ завершен")


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
        logger.error(f"Ошибка анализа изображения: {e}")
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
        logger.error(f"Ошибка обработки фото: {e}")
        bot.reply_to(message, f'Ошибка обработки фото: {str(e)}')

bot.polling(none_stop=True, interval=0)
