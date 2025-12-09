import os
import warnings
import io
from dotenv import load_dotenv
import telebot
from telebot import types
import time
import logging

# Импортируем InferenceClient
from huggingface_hub import InferenceClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Отключаем предупреждения
warnings.filterwarnings("ignore")

load_dotenv()
TOKEN = os.getenv("TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    raise RuntimeError("❌ В .env нет TOKEN")

if not HF_TOKEN:
    logger.error("❌ HF_TOKEN не найден в .env файле")
    logger.info("ℹ️ Получите бесплатный токен на https://huggingface.co/settings/tokens")
    logger.info("ℹ️ Добавьте в .env: HF_TOKEN=ваш_токен")
else:
    logger.info("✅ Hugging Face токен найден")

# Инициализируем клиент для FLUX.1-dev через Nebius
try:
    client = InferenceClient(
        provider="nebius",  # Используем Nebius как провайдера
        api_key=HF_TOKEN,
    )
    logger.info("✅ InferenceClient инициализирован с провайдером Nebius")
except Exception as e:
    logger.error(f"❌ Ошибка инициализации InferenceClient: {e}")
    client = None

bot = telebot.TeleBot(TOKEN)

# Хранилище состояний пользователей
user_states = {}
user_data = {}

# Словарь для маппинга стилей на промпты
STYLE_PROMPTS = {
    "Минимализм": "minimalist tattoo design, clean thin lines, simple elegant design, single needle style, delicate, subtle",
    "Традишнл": "traditional tattoo, american traditional style, bold black outlines, limited color palette, sailor jerry style, tattoo flash",
    "Реализм": "realistic tattoo, photorealistic, detailed shading, 3D effect, skin texture, hyperrealistic tattoo art",
    "Акварель": "watercolor tattoo, paint splashes effect, soft edges, blended colors, artistic, painterly style",
    "Геометрия": "geometric tattoo, sacred geometry, mandala pattern, symmetrical, precise lines, dotwork, intricate patterns",
    "Блэкворк": "blackwork tattoo, solid black areas, heavy black fill, ornamental patterns, bold contrast",
    "Лайнворк": "linework tattoo, continuous line drawing, single line art, elegant contours, minimalist line art",
    "Трайбл": "tribal tattoo, polynesian tattoo patterns, maori design, cultural motifs, flowing black lines",
    "Биомеханика": "biomechanical tattoo, H.R. Giger style, mechanical parts integrated with flesh, cyborg, industrial",
    "Олдскул": "old school tattoo, vintage flash, classic designs, bold lines, roses, anchors, swallows",
    "Японский": "japanese irezumi tattoo, traditional japanese style, koi fish, dragons, waves, chrysanthemums",
    "Скетч": "sketch style tattoo, pencil drawing style, rough lines, artistic sketch, hand-drawn look",
    "Киберпанк": "cyberpunk tattoo, neon colors, glitch effect, digital art style, futuristic, techwear"
}

class UserState:
    NONE = 0
    WAITING_FOR_STYLE = 1
    WAITING_FOR_BODY_PART = 2
    WAITING_FOR_SUBJECT = 3
    WAITING_FOR_COLOR = 4

def ensure_user_data(chat_id):
    """Проверяет и создает словарь данных для пользователя, если его нет"""
    if chat_id not in user_data:
        user_data[chat_id] = {}

def reset_user_state(chat_id):
    """Сброс состояния пользователя"""
    user_states[chat_id] = UserState.NONE
    ensure_user_data(chat_id)

def generate_prompt(user_data_dict):
    """Создает промпт для FLUX.1-dev на основе данных пользователя"""
    style = user_data_dict.get('style', 'Минимализм')
    subject = user_data_dict.get('subject', 'abstract design')
    color = user_data_dict.get('color', 'Черно-белая')
    body_part = user_data_dict.get('body_part', 'arm')

    # Базовый стиль
    style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["Минимализм"])

    # Цветовая схема
    if color == "Черно-белая":
        color_prompt = "black and white, monochrome, grayscale, no color"
    elif color == "Цветная":
        color_prompt = "vibrant colors, colorful, saturated, rich colors"
    elif color == "Монохром":
        color_prompt = "monochromatic, single color, tonal variation"
    else:  # С акцентами цвета
        color_prompt = "black and white with color accents, color highlights, mostly monochrome"

    # Описание части тела
    body_part_mapping = {
        "Плечо": "shoulder tattoo, upper arm placement",
        "Предплечье": "forearm tattoo, arm placement",
        "Запястье": "wrist tattoo, delicate placement",
        "Кисть": "hand tattoo, knuckle tattoo",
        "Грудь": "chest tattoo, sternum tattoo, chest piece",
        "Ребра": "rib tattoo, side body, underboob tattoo",
        "Спина": "back tattoo, full back piece, back artwork",
        "Живот": "stomach tattoo, abdomen tattoo",
        "Шея": "neck tattoo, nape tattoo, throat tattoo",
        "За ухом": "behind ear tattoo, ear tattoo",
        "Лодыжка": "ankle tattoo, foot tattoo",
        "Бедро": "thigh tattoo, leg tattoo",
        "Икра": "calf tattoo, leg tattoo",
        "Лопатка": "shoulder blade tattoo, scapula",
        "Ключица": "collarbone tattoo, clavicle tattoo"
    }

    body_prompt = body_part_mapping.get(body_part, "tattoo design")

    # Собираем промпт для FLUX.1-dev
    prompt = f"{style_prompt}, {subject}, {body_prompt}, {color_prompt}, tattoo design, high quality, detailed, professional tattoo art, 8k resolution"

    # Негативный промпт
    negative_prompt = "blurry, low quality, ugly, deformed, distorted, watermark, text, signature, bad anatomy, extra limbs, missing limbs"

    return prompt, negative_prompt

def generate_image_with_flux(prompt, negative_prompt=""):
    """Генерация изображения через FLUX.1-dev с InferenceClient"""
    try:
        if not client:
            logger.error("InferenceClient не инициализирован")
            return None, "InferenceClient не инициализирован. Проверьте HF_TOKEN."

        if not HF_TOKEN:
            logger.error("HF_TOKEN не настроен")
            return None, "HF_TOKEN не настроен"

        logger.info(f"🚀 Генерация изображения через FLUX.1-dev")
        logger.info(f"📝 Промпт: {prompt[:100]}...")

        start_time = time.time()

        # Генерируем изображение через FLUX.1-dev
        try:
            logger.info("📤 Отправка запроса к FLUX.1-dev...")

            # Используем параметры для лучшего качества татуировок
            image = client.text_to_image(
                prompt,
                model="black-forest-labs/FLUX.1-dev",
                negative_prompt=negative_prompt,
                guidance_scale=3.5,  # Для FLUX лучше 3.5-4.0
                num_inference_steps=20,  # FLUX быстрая, 20 шагов достаточно
                height=1024,  # FLUX поддерживает высокое разрешение
                width=1024,
                seed=None  # Случайный сид для разнообразия
            )

            generation_time = time.time() - start_time
            logger.info(f"⏱️ Генерация заняла: {generation_time:.1f} секунд")

            if image:
                # Сохраняем для отладки
                debug_dir = "generated_tattoos"
                os.makedirs(debug_dir, exist_ok=True)
                timestamp = int(time.time())
                debug_path = os.path.join(debug_dir, f"tattoo_flux_{timestamp}.png")
                image.save(debug_path)
                logger.info(f"💾 Изображение сохранено: {debug_path}")

                # Конвертируем в байты
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG', quality=95)
                img_byte_arr.seek(0)

                return img_byte_arr, None
            else:
                logger.error("❌ FLUX.1-dev вернул пустое изображение")
                return None, "Пустое изображение"

        except Exception as e:
            logger.error(f"❌ Ошибка FLUX.1-dev: {str(e)}")

            # Пробуем альтернативный подход без провайдера
            logger.info("🔄 Пробую альтернативный метод...")
            try:
                # Пробуем через стандартный API
                from huggingface_hub import InferenceClient as StandardClient
                alt_client = StandardClient(
                    api_key=HF_TOKEN,
                )

                image = alt_client.text_to_image(
                    prompt,
                    model="black-forest-labs/FLUX.1-dev",
                    negative_prompt=negative_prompt,
                )

                if image:
                    # Сохраняем изображение
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG', quality=95)
                    img_byte_arr.seek(0)
                    return img_byte_arr, None
                else:
                    return None, f"Ошибка FLUX.1-dev: {str(e)}"

            except Exception as alt_error:
                logger.error(f"❌ Альтернативный метод тоже не сработал: {alt_error}")
                return None, f"Ошибка генерации: {str(e)}"

    except Exception as e:
        logger.error(f"❌ Неизвестная ошибка в generate_image_with_flux: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Ошибка: {str(e)[:100]}"

@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == UserState.WAITING_FOR_STYLE)
def handle_style_selection(message):
    try:
        chat_id = message.chat.id
        ensure_user_data(chat_id)

        if message.text in STYLE_PROMPTS:
            user_data[chat_id]['style'] = message.text
        else:
            user_data[chat_id]['style'] = "Минимализм"

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
        body_parts = ["Плечо", "Предплечье", "Запястье", "Кисть", "Грудь", "Ребра",
                      "Спина", "Живот", "Шея", "За ушем", "Лодыжка", "Бедро", "Икра"]

        for i in range(0, len(body_parts), 2):
            markup.row(*body_parts[i:i + 2])

        bot.send_message(
            chat_id,
            f"✅ <b>Стиль:</b> {user_data[chat_id]['style']}\n\n"
            "📍 <b>Выбери часть тела для тату:</b>",
            reply_markup=markup,
            parse_mode='HTML'
        )
        user_states[chat_id] = UserState.WAITING_FOR_BODY_PART
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_style_selection: {e}")
        bot.send_message(chat_id, "❌ Ошибка. Попробуйте снова /generate")

@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == UserState.WAITING_FOR_BODY_PART)
def handle_body_part_selection(message):
    try:
        chat_id = message.chat.id
        ensure_user_data(chat_id)

        user_data[chat_id]['body_part'] = message.text

        markup = types.ReplyKeyboardRemove()
        bot.send_message(
            chat_id,
            f"✅ <b>Часть тела:</b> {user_data[chat_id]['body_part']}\n\n"
            "🎨 <b>Что должно быть изображено на тату?</b>\n\n"
            "🌍 <b>Для лучшего результата пиши описание на английском!</b>\n"
            "🤖 FLUX.1-dev лучше понимает английский язык\n\n"
            "<i>Опиши детально:</i>\n"
            "• <b>wolf with moon light</b> (волк с лунным светом)\n"
            "• <b>lotus flower with roots</b> (цветок лотоса с корнями)\n"
            "• <b>dragon wrapping around a sword</b> (дракон, обвивающий меч)\n"
            "• <b>compass and old map</b> (компас и старая карта)\n"
            "• <b>phoenix with spread wings</b> (феникс с расправленными крыльями)\n\n"
            "<b>Чем детальнее описание, тем лучше результат!</b>",
            reply_markup=markup,
            parse_mode='HTML'
        )
        user_states[chat_id] = UserState.WAITING_FOR_SUBJECT
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_body_part_selection: {e}")

@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == UserState.WAITING_FOR_SUBJECT)
def handle_subject_description(message):
    try:
        chat_id = message.chat.id
        ensure_user_data(chat_id)

        user_data[chat_id]['subject'] = message.text

        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
        markup.row("Черно-белая", "Цветная")
        markup.row("Монохром", "С акцентами цвета")

        bot.send_message(
            chat_id,
            f"✅ <b>Изображение:</b> {user_data[chat_id]['subject']}\n\n"
            "🌈 <b>Выбери цветовую схему:</b>",
            reply_markup=markup,
            parse_mode='HTML'
        )
        user_states[chat_id] = UserState.WAITING_FOR_COLOR
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_subject_description: {e}")

@bot.message_handler(func=lambda message: user_states.get(message.chat.id) == UserState.WAITING_FOR_COLOR)
def handle_color_selection(message):
    try:
        chat_id = message.chat.id
        ensure_user_data(chat_id)

        color_options = ["Черно-белая", "Цветная", "Монохром", "С акцентами цвета"]

        if message.text in color_options:
            user_data[chat_id]['color'] = message.text

            summary_text = (
                f"✨ <b>Параметры эскиза:</b>\n\n"
                f"🤖 <b>Генератор:</b> FLUX.1-dev\n"
                f"🎨 <b>Стиль:</b> {user_data[chat_id]['style']}\n"
                f"📍 <b>Место:</b> {user_data[chat_id]['body_part']}\n"
                f"🖼 <b>Изображение:</b> {user_data[chat_id]['subject']}\n"
                f"🌈 <b>Цвет:</b> {user_data[chat_id]['color']}\n\n"
                f"⏳ <i>Генерирую эскиз... Это займет 15-45 секунд.</i>"
            )

            remove_markup = types.ReplyKeyboardRemove()
            msg = bot.send_message(chat_id, summary_text,
                                   reply_markup=remove_markup, parse_mode='HTML')

            # Запускаем генерацию
            generate_and_send_tattoo(chat_id, msg.message_id)

        else:
            bot.send_message(
                chat_id,
                "❌ Выбери вариант цвета из предложенных.",
                parse_mode='HTML'
            )
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_color_selection: {e}")

def generate_and_send_tattoo(chat_id, message_id=None):
    """Функция генерации и отправки эскиза через FLUX.1-dev"""
    try:
        data = user_data.get(chat_id, {})

        if not data:
            bot.send_message(chat_id, "❌ Не удалось найти данные. Попробуйте снова /generate")
            return

        # Обновляем сообщение
        if message_id:
            try:
                bot.edit_message_text(
                    "🎨 <b>FLUX.1-dev запущен...</b>\n"
                    "⏳ Генерация через Nebius\n"
                    "<i>Это займет 5-30 секунд</i>",
                    chat_id=chat_id,
                    message_id=message_id,
                    parse_mode='HTML'
                )
            except:
                pass

        # Генерируем промпт
        prompt, negative_prompt = generate_prompt(data)

        logger.info(f"📝 Генерация с промптом: {prompt[:100]}...")

        # Генерируем изображение через FLUX.1-dev
        image_bytes, error_message = generate_image_with_flux(prompt, negative_prompt)

        if image_bytes:
            # Обновляем сообщение
            if message_id:
                try:
                    bot.edit_message_text(
                        "✅ <b>Эскиз готов!</b>\n"
                        "Отправляю изображение...",
                        chat_id=chat_id,
                        message_id=message_id,
                        parse_mode='HTML'
                    )
                except:
                    pass

            # Отправляем изображение
            try:
                image_bytes.seek(0)

                bot.send_photo(
                    chat_id,
                    photo=image_bytes,
                    caption=f"🎨 <b>Твой эскиз татуировки</b>\n"
                            f"🤖 <b>Генератор:</b> FLUX.1-dev (Nebius)\n"
                            f"📏 <b>Разрешение:</b> 1024x1024\n\n"
                            f"<b>Стиль:</b> {data.get('style', 'Не указан')}\n"
                            f"<b>Место:</b> {data.get('body_part', 'Не указано')}\n"
                            f"<b>Изображение:</b> {data.get('subject', 'Не указано')}\n"
                            f"<b>Цвет:</b> {data.get('color', 'Не указан')}\n\n"
                            f"💡 <i>Сохрани для консультации с тату-мастером!</i>",
                    parse_mode='HTML'
                )
            except Exception as e:
                logger.error(f"❌ Ошибка отправки фото: {e}")

            bot.send_message(
                chat_id,
                f"💭 <b>Использованный промпт:</b>\n"
                f"<code>{prompt[:700]}</code>\n\n"
                f"🔄 Новый эскиз: /generate\n"
                f"🤖 Модель: FLUX.1-dev через Nebius",
                parse_mode='HTML'
            )

        else:
            # Если не удалось сгенерировать
            if message_id:
                try:
                    bot.edit_message_text(
                        f"⚠️ <b>Не удалось сгенерировать изображение</b>\n\n"
                        f"Причина: {error_message}",
                        chat_id=chat_id,
                        message_id=message_id,
                        parse_mode='HTML'
                    )
                except:
                    pass

            # Предлагаем альтернативу
            bot.send_message(
                chat_id,
                f"💡 <b>Попробуй:</b>\n"
                f"• Упростить описание\n"
                f"• Проверить токен: /status\n"
                f"• Подождать минуту и попробовать снова\n"
                f"• Использовать более простой запрос\n\n"
                f"🔄 Новый эскиз: /generate",
                parse_mode='HTML'
            )

        reset_user_state(chat_id)

    except Exception as e:
        logger.error(f"❌ Ошибка в generate_and_send_tattoo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        try:
            bot.send_message(
                chat_id,
                "❌ <b>Произошла ошибка при генерации</b>\n"
                "Попробуйте еще раз или используйте более простое описание.\n\n"
                "🔄 Напиши /generate чтобы попробовать снова",
                parse_mode='HTML'
            )
        except:
            pass
        reset_user_state(chat_id)

def test_generation(message):
    """Тестовая команда для проверки FLUX.1-dev"""
    chat_id = message.chat.id

    if not HF_TOKEN:
        bot.send_message(
            chat_id,
            "❌ <b>Hugging Face токен не настроен</b>\n\n"
            "Добавьте в .env файл:\n"
            "<code>HF_TOKEN=ваш_токен</code>\n\n"
            "Получите токен на https://huggingface.co/settings/tokens",
            parse_mode='HTML'
        )
        return

    if not client:
        bot.send_message(
            chat_id,
            "❌ <b>InferenceClient не инициализирован</b>\n\n"
            "Проверьте ваш HF_TOKEN и перезапустите бота.",
            parse_mode='HTML'
        )
        return

    bot.send_message(
        chat_id,
        "🧪 <b>Тестирую FLUX.1-dev через Nebius...</b>\n"
        "⏳ Генерация тестового изображения...",
        parse_mode='HTML'
    )

    try:
        # Простой тестовый промпт
        test_prompt = "minimalist black and white tattoo of a simple geometric wolf, clean lines, elegant design, tattoo art, high quality, 8k"
        negative_prompt = "blurry, low quality, watermark, text"

        image_bytes, error = generate_image_with_flux(test_prompt, negative_prompt)

        if image_bytes:
            bot.send_photo(
                chat_id,
                photo=image_bytes,
                caption="✅ <b>FLUX.1-dev работает через Nebius!</b>\n"
                        "🎨 Генерация успешна\n"
                        "🤖 Провайдер: Nebius\n"
                        "⚡ Модель: FLUX.1-dev\n\n"
                        "Создайте свой эскиз: /generate",
                parse_mode='HTML'
            )
        else:
            bot.send_message(
                chat_id,
                f"❌ <b>Тест не пройден</b>\n\n"
                f"Ошибка: {error}\n\n"
                "Проверьте:\n"
                "1. HF_TOKEN в .env файле\n"
                "2. Правильный ли токен (нужен Inference API токен)\n"
                "3. Доступность Nebius провайдера",
                parse_mode='HTML'
            )

    except Exception as e:
        logger.error(f"❌ Ошибка теста: {e}")
        bot.send_message(
            chat_id,
            f"❌ <b>Ошибка тестирования:</b> {str(e)[:200]}",
            parse_mode='HTML'
        )

## КОМАНДЫ

@bot.message_handler(commands=['start'])
def start(message):
    welcome_text = (
        "🎨 <b>Добро пожаловать в TattooKaterokBot!</b>\n\n"
        "🤖 Я создаю эскизы татуировок с помощью <b>FLUX.1-dev</b> (новейшая модель!)\n\n"
        "⚡ <b>Преимущества FLUX.1-dev:</b>\n"
        "• 🚀 Быстрая генерация (5-30 секунд)\n"
        "• 🎨 Высокое качество (1024x1024 пикселей)\n"
        "• 💰 Полностью бесплатно\n"
        "• 🌐 Провайдер: Nebius\n\n"
        "✨ <b>Основные команды:</b>\n"
        "/generate - Создать эскиз тату\n"
        "/test - Проверить работу FLUX.1-dev\n"
        "/status - Статус API\n"
        "/help - Помощь\n"
        "/about - О боте\n\n"
        "Напиши /generate чтобы начать!"
    )
    try:
        bot.reply_to(message, welcome_text, parse_mode='HTML')
    except Exception as e:
        logger.error(f"❌ Ошибка отправки приветствия: {e}")
    reset_user_state(message.chat.id)

@bot.message_handler(commands=['about'])
def about_bot(message):
    about_text = (
        "👩🏻‍🦰 <b>Автор:</b> Верниковская Екатерина Андреевна\n\n"
        "🤖 <b>TattooKaterokBot v1.0.0</b>\n\n"
        "🎨 <b>Стек технологий:</b>\n"
        "• <b>Backend:</b> Python 3.10+\n"
        "• <b>Telegram API:</b> PyTelegramBotAPI 4.19+\n"
        "• <b>ИИ-модель:</b> FLUX.1-dev от Black Forest Labs\n"
        "• <b>API провайдер:</b> Nebius Inference\n"
        "• <b>Хостинг модели:</b> Hugging Face Hub\n\n"
        "⚙️ <b>Технические характеристики:</b>\n"
        "• Разрешение: 1024×1024 пикселей\n"
        "• Время генерации: 5-30 секунд\n"
        "• Стилей тату: 13 вариантов\n"
        "• Частей тела: 15 локаций\n"
        "• Цветовых схем: 4 варианта\n\n"
        "⚠️ <b>Ограничения:</b>\n"
        "• Nebius может иметь лимиты запросов\n"
        "• FLUX.1-dev — тестовая модель\n"
        "• Требуется стабильный интернет\n\n"
        "🎯 <b>Цель проекта:</b>\n"
        "Предоставить бесплатный инструмент для визуализации\n"
        "идей татуировок перед визитом к мастеру.\n\n"
        "📝 <b>Важно:</b>\n"
        "• Эскизы создаются искусственным интеллектом и являются концептами.\n"
        "• Для финального дизайна проконсультируйтесь с профессиональным тату-мастером!\n\n"
    )
    bot.reply_to(message, about_text, parse_mode='HTML')

@bot.message_handler(commands=['generate'])
def start_generation(message):
    try:
        chat_id = message.chat.id
        ensure_user_data(chat_id)

        # Проверяем наличие токена
        if not HF_TOKEN:
            bot.send_message(
                chat_id,
                "❌ <b>Hugging Face токен не настроен!</b>\n\n"
                "Добавьте в .env файл:\n"
                "<code>HF_TOKEN=ваш_токен</code>\n\n"
                "Получите бесплатный токен:\n"
                "1. Зайдите на https://huggingface.co\n"
                "2. Зарегистрируйтесь\n"
                "3. Настройки → Access Tokens\n"
                "4. Создайте новый токен\n\n"
                "После добавления перезапустите бота.",
                parse_mode='HTML'
            )
            return

        if not client:
            bot.send_message(
                chat_id,
                "❌ <b>InferenceClient не инициализирован</b>\n\n"
                "Проверьте ваш HF_TOKEN и перезапустите бота.",
                parse_mode='HTML'
            )
            return

        # Начинаем процесс генерации
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
        styles = list(STYLE_PROMPTS.keys())

        for i in range(0, len(styles), 3):
            markup.row(*styles[i:i + 3])

        bot.send_message(
            chat_id,
            "🤖 <b>Используется FLUX.1-dev</b>\n"
            "🚀 Быстрая генерация через Nebius\n\n"
            "🎨 <b>Выбери стиль татуировки:</b>",
            reply_markup=markup,
            parse_mode='HTML'
        )
        user_states[chat_id] = UserState.WAITING_FOR_STYLE

    except Exception as e:
        logger.error(f"❌ Ошибка в start_generation: {e}")
        bot.reply_to(message, "❌ Ошибка. Попробуйте еще раз.")

@bot.message_handler(commands=['test'])
def test_generation(message):
    """Тестовая команда для проверки FLUX.1-dev"""
    chat_id = message.chat.id

    if not HF_TOKEN:
        bot.send_message(
            chat_id,
            "❌ <b>Hugging Face токен не настроен</b>\n\n"
            "Добавьте в .env файл:\n"
            "<code>HF_TOKEN=ваш_токен</code>\n\n"
            "Получите токен на https://huggingface.co/settings/tokens",
            parse_mode='HTML'
        )
        return

    if not client:
        bot.send_message(
            chat_id,
            "❌ <b>InferenceClient не инициализирован</b>\n\n"
            "Проверьте ваш HF_TOKEN и перезапустите бота.",
            parse_mode='HTML'
        )
        return

    bot.send_message(
        chat_id,
        "🧪 <b>Тестирую FLUX.1-dev через Nebius...</b>\n"
        "⏳ Генерация тестового изображения...",
        parse_mode='HTML'
    )

    try:
        # Простой тестовый промпт
        test_prompt = "minimalist black and white tattoo of a simple geometric wolf, clean lines, elegant design, tattoo art, high quality, 8k"
        negative_prompt = "blurry, low quality, watermark, text"

        image_bytes, error = generate_image_with_flux(test_prompt, negative_prompt)

        if image_bytes:
            bot.send_photo(
                chat_id,
                photo=image_bytes,
                caption="✅ <b>FLUX.1-dev работает через Nebius!</b>\n"
                        "🎨 Генерация успешна\n"
                        "🤖 Провайдер: Nebius\n"
                        "⚡ Модель: FLUX.1-dev\n\n"
                        "Создайте свой эскиз: /generate",
                parse_mode='HTML'
            )
        else:
            bot.send_message(
                chat_id,
                f"❌ <b>Тест не пройден</b>\n\n"
                f"Ошибка: {error}\n\n"
                "Проверьте:\n"
                "1. HF_TOKEN в .env файле\n"
                "2. Правильный ли токен (нужен Inference API токен)\n"
                "3. Доступность Nebius провайдера",
                parse_mode='HTML'
            )

    except Exception as e:
        logger.error(f"❌ Ошибка теста: {e}")
        bot.send_message(
            chat_id,
            f"❌ <b>Ошибка тестирования:</b> {str(e)[:200]}",
            parse_mode='HTML'
        )

@bot.message_handler(commands=['status'])
def show_status(message):
    """Показывает статус FLUX.1-dev API"""
    status_text = (
        "📊 <b>Статус FLUX.1-dev API</b>\n\n"
        f"🔑 <b>Токен настроен:</b> {'✅ Да' if HF_TOKEN else '❌ Нет'}\n"
        f"🤖 <b>Клиент инициализирован:</b> {'✅ Да' if client else '❌ Нет'}\n"
        f"🚀 <b>Провайдер:</b> Nebius\n"
        f"⚡ <b>Модель:</b> black-forest-labs/FLUX.1-dev\n"
        f"📏 <b>Разрешение:</b> 1024x1024 пикселей\n"
        f"⏱️ <b>Скорость:</b> 5-30 секунд\n"
        f"💳 <b>Оплата:</b> Nebius может иметь лимиты\n"
        f"🌐 <b>VPN:</b> Не требуется\n\n"
        "💡 <b>Если не работает:</b>\n"
        "1. Убедитесь что токен правильный\n"
        "2. Nebius может быть временно недоступен\n"
        "3. Попробуйте позже\n\n"
        "🔄 Новый эскиз: /generate"
    )

    bot.reply_to(message, status_text, parse_mode='HTML')

@bot.message_handler(commands=['styles'])
def show_styles(message):
    styles_text = (
        "🎭 <b>Доступные стили татуировок:</b>\n\n"
        "• <b>Минимализм</b> - тонкие линии, элегантность\n"
        "• <b>Традишнл</b> - яркие цвета, четкие контуры\n"
        "• <b>Реализм</b> - как фотография, максимальная детализация\n"
        "• <b>Акварель</b> - эффект размытых красок, художественный\n"
        "• <b>Геометрия</b> - симметрия, узоры, мандалы\n"
        "• <b>Блэкворк</b> - сплошная черная заливка, контраст\n"
        "• <b>Лайнворк</b> - только контуры, минимализм\n"
        "• <b>Трайбл</b> - этнические узоры, черный цвет\n"
        "• <b>Биомеханика</b> - тело + механизмы, индустриальный\n"
        "• <b>Олдскул</b> - классические морские мотивы\n"
        "• <b>Японский</b> - иредзуми, драконы, кои\n"
        "• <b>Скетч</b> - набросок карандашом, эскиз\n"
        "• <b>Киберпанк</b> - неон, технологии, будущее\n\n"
        "Используй /generate чтобы выбрать стиль!"
    )
    bot.reply_to(message, styles_text, parse_mode='HTML')

@bot.message_handler(commands=['bodyplace'])
def body_placement(message):
    placement_text = (
        "📍 <b>Где разместить тату?</b>\n\n"

        "💪 <b>Плечо/предплечье</b> - Классика, мало боли\n"
        "🎯 <b>Запястье</b> - Для небольших тату\n"
        "🦵 <b>Лодыжка/голень</b> - Женственный вариант\n"
        "🖐 <b>Ребра</b> - Болезненно, но эффектно\n"
        "🔙 <b>Спина</b> - Для крупных работ\n"
        "🫀 <b>Грудь</b> - Символичные тату\n"
        "👂 <b>За ухом</b> - Минималистичные\n"
        "🎗 <b>Шея</b> - Смелый выбор\n\n"

        "⚠️ <b>Самые болезненные зоны:</b>\n"
        "• Ребра\n• Колени\n• Локти\n• Голова\n\n"

        "Используй /pain для подробной шкалы боли"
    )
    bot.reply_to(message, placement_text, parse_mode='HTML')

@bot.message_handler(commands=['pain'])
def body_pain(message):
    pain_text = (
        "😖 <b>Шкала боли от 1 до 10:</b>\n\n"
        "1-2: Плечи, предплечья, бедра\n"
        "3-4: Грудь, спина, икры\n"
        "5-6: Запястья, шея, ключицы\n"
        "7-8: Ребра, позвоночник, живот\n"
        "9-10: Колени, локти, голова, пальцы\n\n"
        "💡 <b>Совет:</b> Первую тату лучше делать в менее болезненной зоне!"
    )
    bot.reply_to(message, pain_text, parse_mode='HTML')

@bot.message_handler(commands=['care'])
def tattoo_care(message):
    care_text = (
        "🩹 <b>Уход за новой татуировкой:</b>\n\n"

        "1️⃣ <b>Первые 2-3 часа:</b>\n"
        "   • Снять пленку\n"
        "   • Промыть теплой водой с мылом\n"
        "   • Промокнуть салфеткой\n\n"

        "2️⃣ <b>Первые 3 дня:</b>\n"
        "   • Мазать тонким слоем мази (Бепантен, Д-Пантенол)\n"
        "   • 2-3 раза в день\n"
        "   • Не мочить длительно\n\n"

        "3️⃣ <b>Неделя 1-2:</b>\n"
        "   • Использовать увлажняющий крем\n"
        "   • Не чесать, не сдирать корочки\n"
        "   • Избегать солнца, бани, бассейна\n\n"

        "⚠️ <b>Что нельзя:</b>\n"
        "• Солнце\n• Хлорка\n• Тесная одежда\n• Расчесывать\n\n"

        "Полное заживление: 2-4 недели"
    )
    bot.reply_to(message, care_text, parse_mode='HTML')

@bot.message_handler(commands=['help'])
def help_cmd(message):
    help_text = (
        "🆘 <b>Помощь по командам TattooKaterokBot</b>\n\n"

        "🎨 <b>Генерация эскизов:</b>\n"
        "/generate - Создать эскиз тату\n"
        "/test - Проверить работу FLUX.1-dev\n"
        "/status - Статус API\n"
        "/styles - Стили татуировок\n\n"

        "⚙️ <b>Как это работает:</b>\n"
        "1. Выбираешь стиль тату\n"
        "2. Указываешь часть тела\n"
        "3. Описываешь изображение\n"
        "4. Выбираешь цветовую схему\n"
        "5. Получаешь уникальный эскиз!\n\n"

        "📐 <b>Информация:</b>\n"
        "/bodyplace - Где лучше разместить тату\n"
        "/pain - Шкала боли для разных зон\n"
        "/care - Уход после нанесения\n"
        "/about - О боте\n\n"

        "🤖 <b>Технология:</b> FLUX.1-dev (Hugging Face)\n"
        "⚡ <b>Для начала:</b> /generate"
    )
    bot.reply_to(message, help_text, parse_mode='HTML')

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    """Обработка всех сообщений"""
    try:
        chat_id = message.chat.id

        if user_states.get(chat_id) != UserState.NONE:
            return

        if message.text.startswith('/'):
            bot.reply_to(message,
                         "Неизвестная команда 😕\n"
                         "Используй /help чтобы увидеть все команды\n"
                         "Или /generate чтобы создать эскиз тату!",
                         parse_mode='HTML')
        else:
            bot.reply_to(message,
                         "Привет! Я бот для создания эскизов татуировок. 🎨\n"
                         "🤖 Использую FLUX.1-dev от Hugging Face\n\n"
                         "Напиши /help чтобы увидеть все команды\n"
                         "Или /generate чтобы начать создание!\n\n"
                         f"🔑 Hugging Face: {'✅ Настроен' if HF_TOKEN else '⚠️ Требуется токен'}",
                         parse_mode='HTML')
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_all_messages: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("🤖 TattooKaterokBot - FLUX.1-dev через InferenceClient")
    print("=" * 60)
    print(f"🔑 Hugging Face токен: {'✅ Найден' if HF_TOKEN else '❌ Не найден'}")
    print(f"🤖 InferenceClient: {'✅ Инициализирован' if client else '❌ Не инициализирован'}")
    print(f"🚀 Провайдер: Nebius")
    print(f"⚡ Модель: black-forest-labs/FLUX.1-dev")

    if not HF_TOKEN:
        print("\n⚠️  Для работы бота нужен токен Hugging Face:")
        print("1. Зайдите на https://huggingface.co")
        print("2. Зарегистрируйтесь")
        print("3. Settings → Access Tokens")
        print("4. Создайте новый токен (бесплатно)")
        print("5. Добавьте в .env файл:")
        print("   HF_TOKEN=ваш_токен")
    elif not client:
        print("\n⚠️  Не удалось инициализировать InferenceClient")
        print("Проверьте ваш токен HF_TOKEN")

    print("=" * 60)

    # Создаем необходимые директории
    directories = ["generated_tattoos"]
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"📁 Создана директория: {dir_name}/")

    print("=" * 60)

    print("\n🚀 Запускаю бота...")

    try:
        bot.infinity_polling(timeout=30, long_polling_timeout=30)
    except Exception as e:
        print(f"❌ Ошибка бота: {e}")
        print("🔄 Перезапустите бота вручную")