import pyttsx3
import speech_recognition as sr
import random
import logging
import threading

# Создаем блокировку для синхронизации вызовов kurumi_speak
speak_lock = threading.Lock()

def setup_voice():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for v in voices:
        if "Microsoft Irina" in v.name or "Microsoft Svetlana" in v.name:
            engine.setProperty('voice', v.id)
            break
        if "female" in v.name.lower() or "женский" in v.name.lower():
            engine.setProperty('voice', v.id)
            break
    engine.setProperty('rate', 185)
    return engine


def listen_command():
    """Улучшенная версия с обработкой шумов"""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        # Увеличиваем время настройки на шум
        recognizer.adjust_for_ambient_noise(source, duration=2)
        logging.info("Настройка на фоновый шум завершена")

    try:
        with mic as source:
            logging.info("Слушаю...")
            # Увеличиваем время ожидания и длину фразы
            audio = recognizer.listen(
                source,
                timeout=8,
                phrase_time_limit=15
            )

        try:
            # Используем альтернативные API распознавания
            text = recognizer.recognize_google(
                audio,
                language="ru-RU",
                show_all=False
            )
            logging.info(f"Распознано: {text}")
            return text.lower()

        except sr.UnknownValueError:
            logging.warning("Речь распознана, но не понята")
            return ""

    except sr.WaitTimeoutError:
        logging.warning("Время ожидания речи истекло")
        return ""

def kurumi_speak(engine, text: str, use_style=True):
    """
    Синтез речи Куруми с добавлением случайных приставок и окончаний (по умолчанию).
    Если use_style=False, текст озвучивается без стилизации.
    """
    with speak_lock:  # Блокируем доступ к engine
        if use_style:
            prefixes = [
                "Охохо…",
                "Хехехе…",
                "Хо-хо-хо…"
            ]
            suffixes = [
                "Хихи…",
                "Хе-хе…",
                "Ммм?",
                "Ахах!",
                "Хо-хо…"
            ]
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes)
            styled_text = f"{prefix} {text} {suffix}"
        else:
            styled_text = text  # Без стилизации

        logging.info("[Куруми]: %s", styled_text)
        engine.say(styled_text)
        engine.runAndWait()

def speak_ai_response(engine, ai_response):
    """
    Озвучивает ответ, сгенерированный моделью Llama.
    """
    if ai_response:
        # Убираем ссылки из текста
        import re
        cleaned_response = re.sub(r'http\S+', '', ai_response)  # Удаляем URL
        kurumi_speak(engine, cleaned_response, use_style=False)  # Без стилизации
    else:
        kurumi_speak(engine, "Извини, я не смогла придумать ответ.", use_style=False)