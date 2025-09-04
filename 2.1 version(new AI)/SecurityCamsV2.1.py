import telebot, time, requests, cv2, os
from threading import Thread, Event
from queue import Queue
import numpy as np
from ultralytics import YOLO

bot = telebot.TeleBot('TOKEN')
url_person = ''

temp_dir = 'temp_photos'
os.makedirs(temp_dir, exist_ok=True)
model = YOLO('best.onnx')

with open(file="InstructionV1.3.txt", encoding="utf-8") as a:
    instruction = a.read()

video_processing = False
current_chat_id = None
frame_queue = Queue(maxsize=10)
stop_event = Event()


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
    bot.send_message(message.chat.id,
"""
Доступные команды:
/setcamera - Добавить камеру для анализа;
/mycameras - Просмотреть подключенные камеры;
/startanalyzecamera - Начать анализ камеры;
/stopanalyzecamera - Закончить анализ камеры; 
Также вы можете скинуть фотографию и наш ИИ проанализирует ее на наличие оружия. 
""")


@bot.message_handler(commands=['setcamera'])
def start_analyze_camera(message):
    bot.send_message(message.chat.id, 'Введите URL видеопотока. Примеры:\n'
                                      '- Для IP-камер: rtsp://admin:password@192.168.1.64:554/stream1\n'
                                      '- Для веб-камер: http://192.168.1.64:8080/video\n'
                                      '- Для локальной камеры: 0\n'
                                      '- Для выхода введите "Отмена"')
    bot.register_next_step_handler(message, main_analyze_camera)


def main_analyze_camera(message):
    global url_person
    url = message.text
    if message.text == 'Отмена':
        return
    try:
        url_person = url
        bot.send_message(message.chat.id, "Камера успешно добавлена!")
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка {e}. Попробуйте еще раз, либо напишите \"Отмена\" для выхода")
        bot.register_next_step_handler(message, main_analyze_camera)


@bot.message_handler(commands=['mycameras'])
def save_cameras(message):
    global url_person
    bot.send_message(message.chat.id, f"Ваши камеры: {url_person}")


@bot.message_handler(commands=['stopanalyzecamera'])
def stop_analyze(message):
    global video_processing
    if not video_processing:
        bot.send_message(message.chat.id, "Анализ камеры не ведётся!")
    else:
        video_processing = False
        stop_event.set()
        bot.send_message(message.chat.id, "Анализ камеры завершён!")


@bot.message_handler(commands=['startanalyzecamera'])
def analyze_camera(message):
    global url_person, video_processing, current_chat_id
    if not url_person:
        bot.send_message(message.chat.id, "У вас нет ни одной подключенной камеры")
        return

    current_chat_id = message.chat.id
    video_processing = True
    frame_counter = 0
    analysis_interval = 5

    bot.send_message(message.chat.id, "Начинаю анализ видеопотока...")

    if url_person.endswith((".jpg", ".jpeg", ".png")):
        while video_processing:
            try:
                response = requests.get(url_person, timeout=10)
                if response.status_code != 200:
                    raise Exception(f"Ошибка получения кадра: HTTP {response.status_code}")

                frame_counter += 1
                if frame_counter % analysis_interval != 0:
                    time.sleep(0.1)
                    continue

                frame_path = os.path.join(temp_dir, 'current_frame.jpg')
                with open(frame_path, 'wb') as f:
                    f.write(response.content)

                result, processed_img = analyze_image(frame_path)
                if "ОРУЖИЕ" in result:
                    result_path = os.path.join(temp_dir, 'result_frame.jpg')
                    cv2.imwrite(result_path, processed_img)
                    with open(result_path, 'rb') as photo:
                        bot.send_photo(current_chat_id, photo, caption=result)
                        bot.send_message(current_chat_id, instruction)

            except Exception as e:
                bot.send_message(current_chat_id, f"Ошибка обработки кадра: {str(e)}")
                time.sleep(5)
    else:
        stop_event.clear()
        capture_thread = Thread(target=video_capture)
        processing_thread = Thread(target=frame_processing)
        capture_thread.start()
        processing_thread.start()


def video_capture():
    global video_processing, url_person
    cap = None
    try:
        if url_person.startswith(('rtsp://', 'http://', 'https://')):
            cap = cv2.VideoCapture(url_person)
        else:
            cap = cv2.VideoCapture(int(url_person) if url_person.isdigit() else 0)

        if not cap.isOpened():
            raise Exception("Не удалось подключиться к видеопотоку")

        while not stop_event.is_set() and video_processing:
            ret, frame = cap.read()
            if not ret:
                raise Exception("Ошибка чтения кадра")

            if not frame_queue.full():
                frame_queue.put(frame)
            time.sleep(0.1)

    except Exception as e:
        if current_chat_id:
            bot.send_message(current_chat_id, f"Ошибка видеопотока: {str(e)}")
    finally:
        if cap is not None:
            cap.release()


def frame_processing():
    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                result, processed_img = analyze_image(frame)
                if "ОРУЖИЕ" in result:
                    result_path = os.path.join(temp_dir, 'result_frame.jpg')
                    cv2.imwrite(result_path, processed_img)
                    with open(result_path, 'rb') as photo:
                        bot.send_photo(current_chat_id, photo, caption=result)
                        bot.send_message(current_chat_id, instruction)

            except Exception as e:
                if current_chat_id:
                    bot.send_message(current_chat_id, f"Ошибка обработки кадра: {str(e)}")
        time.sleep(0.1)


def analyze_image(image_path):
    try:
        confidence_threshold = 0.65

        if isinstance(image_path, np.ndarray):
            img = image_path.copy()
        else:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Не удалось загрузить изображение")

        original_img = img.copy()
        results = model(img)
        weapon_detected = False

        for result in results:
            for box in result.boxes:
                if int(box.cls) == 0 and box.conf[0] >= confidence_threshold:
                    weapon_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(original_img,
                               f"WEAPON {box.conf[0]:.2f}",
                               (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX,
                               0.7,
                               (0, 0, 255),
                               2)

        if weapon_detected:
            result = f"Обнаружено ОРУЖИЕ! (Уверенность: {box.conf[0]:.2f})"
        else:
            cv2.putText(original_img,
                       "No threats detected",
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)
            result = "Опасных предметов не обнаружено"

        return result, original_img

    except Exception as e:
        error_msg = f"Ошибка анализа изображения: {str(e)}"
        return error_msg, None


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
        error_msg = f'Ошибка обработки фото: {str(e)}'
        bot.reply_to(message, error_msg)


bot.polling(none_stop=True, interval=0)