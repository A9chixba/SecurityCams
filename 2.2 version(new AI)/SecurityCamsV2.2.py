import telebot, sqlite3, time, requests, cv2, os
from threading import Thread, Event
from queue import Queue
import numpy as np
from ultralytics import YOLO

bot = telebot.TeleBot('')

temp_dir = 'temp_photos'
os.makedirs(temp_dir, exist_ok=True)
model = YOLO('best.onnx')

with sqlite3.connect('SecurityCamsDatabase.db') as conn:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY,
        camera_link TEXT,
        is_analyzing BOOLEAN DEFAULT 0
    )
    """)

instruction = """
На записи с вашей камеры было обнаружено оружие. Не выходите из квартиры/дома! В случае необходимости вызовите полицию:
-02 — со стационарного телефона,
-102 — с мобильного.
Назовите ваш адрес и сообщите о вооружённом человеке в подъезде/возле дома."
"""


frame_queue = Queue(maxsize=10)
stop_event = Event()


def get_user_data(user_id):
    with sqlite3.connect('SecurityCamsDatabase.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cursor.fetchone()


def update_camera_link(user_id, url):
    with sqlite3.connect('SecurityCamsDatabase.db') as conn:
        conn.execute("INSERT OR REPLACE INTO users (id, camera_link, is_analyzing) VALUES (?, ?, ?)",
                     (user_id, url, False))
        conn.commit()


def set_analyzing_status(user_id, status):
    with sqlite3.connect('SecurityCamsDatabase.db') as conn:
        conn.execute("UPDATE users SET is_analyzing = ? WHERE id = ?", (status, user_id))
        conn.commit()


def analyze_mjpg_stream(chat_id, stream_url):
    try:
        stream = requests.get(stream_url, stream=True)
        bytes_data = bytes()

        while True:
            user_data = get_user_data(chat_id)
            if not user_data or not user_data[2]:
                break

            bytes_data += stream.raw.read(1024)
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')

            if a != -1 and b != -1:
                jpg = bytes_data[a:b + 2]
                bytes_data = bytes_data[b + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                result, processed_img = analyze_image(img)
                if "ОРУЖИЕ" in result:
                    result_path = os.path.join(temp_dir, f'result_{time.time()}.jpg')
                    cv2.imwrite(result_path, processed_img)
                    with open(result_path, 'rb') as photo:
                        bot.send_photo(chat_id, photo, caption=result)
                        bot.send_message(chat_id, instruction)
                time.sleep(1)

    except Exception as e:
        bot.send_message(chat_id, f"Ошибка MJPG-потока: {str(e)}")

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
    bot.send_message(message.chat.id, """
Доступные команды:
/setcamera - Добавить камеру для анализа
/mycameras - Просмотреть подключенные камеры
/startanalyzecamera - Начать анализ камеры
/stopanalyzecamera - Закончить анализ камеры
Также вы можете скинуть фотографию и наш ИИ проанализирует ее на наличие оружия.""")


@bot.message_handler(commands=['setcamera'])
def set_camera(message):
    bot.send_message(message.chat.id, 'Введите URL видеопотока или "Отмена"')
    bot.register_next_step_handler(message, process_camera_url)


def process_camera_url(message):
    if message.text.lower() == 'отмена':
        bot.send_message(message.chat.id, "Добавление камеры отменено!")
        return

    user_data = get_user_data(message.chat.id)
    if user_data and user_data[1]:
        msg = bot.send_message(message.chat.id, f"У вас уже есть камера: {user_data[1]}\nЗаменить? (Да/Нет)")
        bot.register_next_step_handler(msg, lambda m: handle_camera_replace(m, message.text))
    else:
        update_camera_link(message.chat.id, message.text)
        bot.send_message(message.chat.id, "Камера успешно добавлена!")


def handle_camera_replace(message, new_url):
    if message.text.lower() == 'да':
        update_camera_link(message.chat.id, new_url)
        bot.send_message(message.chat.id, "Камера успешно обновлена!")
    else:
        bot.send_message(message.chat.id, "Изменение отменено.")


@bot.message_handler(commands=['mycameras'])
def show_cameras(message):
    user_data = get_user_data(message.chat.id)
    if user_data:
        status = "активен" if user_data[2] else "не активен"
        bot.send_message(message.chat.id, f"Ваша камера: {user_data[1]}\nСтатус анализа: {status}")
    else:
        bot.send_message(message.chat.id, "У вас нет подключенных камер")

@bot.message_handler(commands=['startanalyzecamera'])
def start_analysis(message):
    user_data = get_user_data(message.chat.id)
    if not user_data or not user_data[1]:
        bot.send_message(message.chat.id, "У вас нет подключенной камеры")
        return

    if user_data[2]:
        bot.send_message(message.chat.id, "Анализ уже запущен")
        return

    set_analyzing_status(message.chat.id, True)
    bot.send_message(message.chat.id, "Анализ начат")

    Thread(target=analyze_stream, args=(message.chat.id, user_data[1])).start()

@bot.message_handler(commands=['stopanalyzecamera'])
def stop_analysis(message):
    user_data = get_user_data(message.chat.id)
    if not user_data or not user_data[2]:
        bot.send_message(message.chat.id, "Анализ не запущен")
        return

    set_analyzing_status(message.chat.id, False)
    stop_event.set()
    bot.send_message(message.chat.id, "Анализ остановлен")


def analyze_stream(chat_id, stream_url):
    if stream_url.endswith((".jpg", ".jpeg", ".png")):
        analyze_image_stream(chat_id, stream_url)
    elif "mjpg" in stream_url.lower() or "video.cgi" in stream_url.lower():
        analyze_mjpg_stream(chat_id, stream_url)
    else:
        analyze_video_stream(chat_id, stream_url)

def analyze_image_stream(chat_id, image_url):
    frame_counter = 0
    analysis_interval = 5

    while True:
        user_data = get_user_data(chat_id)
        if not user_data or not user_data[2]:
            break

        try:
            response = requests.get(image_url, timeout=10)
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
                    bot.send_photo(chat_id, photo, caption=result)
                    bot.send_message(chat_id, instruction)

        except Exception as e:
            bot.send_message(chat_id, f"Ошибка обработки кадра: {str(e)}")
            time.sleep(5)


def analyze_video_stream(chat_id, stream_url):
    cap = None
    try:
        if stream_url.startswith(('rtsp://', 'http://', 'https://')):
            cap = cv2.VideoCapture(stream_url)
        else:
            cap = cv2.VideoCapture(int(stream_url) if stream_url.isdigit() else 0)

        if not cap.isOpened():
            raise Exception("Не удалось подключиться к видеопотоку")

        stop_event.clear()
        while not stop_event.is_set():
            user_data = get_user_data(chat_id)
            if not user_data or not user_data[2]:
                break

            ret, frame = cap.read()
            if not ret:
                raise Exception("Ошибка чтения кадра")

            if not frame_queue.full():
                frame_queue.put(frame)

            time.sleep(0.1)

    except Exception as e:
        bot.send_message(chat_id, f"Ошибка видеопотока: {str(e)}")
    finally:
        if cap is not None:
            cap.release()

    Thread(target=process_frames, args=(chat_id,)).start()


def process_frames(chat_id):
    while not stop_event.is_set() or not frame_queue.empty():
        user_data = get_user_data(chat_id)
        if not user_data or not user_data[2]:
            break

        if not frame_queue.empty():
            frame = frame_queue.get()
            try:
                result, processed_img = analyze_image(frame)
                if "ОРУЖИЕ" in result:
                    result_path = os.path.join(temp_dir, 'result_frame.jpg')
                    cv2.imwrite(result_path, processed_img)
                    with open(result_path, 'rb') as photo:
                        bot.send_photo(chat_id, photo, caption=result)
                        bot.send_message(chat_id, instruction)

            except Exception as e:
                bot.send_message(chat_id, f"Ошибка обработки кадра: {str(e)}")

        time.sleep(0.1)


def analyze_image(image_path):
    try:
        confidence_threshold = 0.25
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
            if "ОРУЖИЕ" in result:
                bot.send_message(message.chat.id,instruction)
        else:
            bot.reply_to(message, result)

    except Exception as e:
        error_msg = f'Ошибка обработки фото: {str(e)}'
        bot.reply_to(message, error_msg)


bot.polling(none_stop=True)