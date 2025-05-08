import random
import time
import logging
import threading
import re
import numpy as np
import pyautogui
from kurumi.tts import setup_voice, kurumi_speak, listen_command, speak_ai_response
from kurumi.spontaneous import autonomous_improvement_loop
from kurumi.skills import search_wikipedia_ru, search_google_ru, read_file, create_file, delete_file
from kurumi.selfmod import self_improve
from kurumi.llm import LLMHandler
from kurumi.vision import InterfaceDetector
from kurumi.interaction import UIInteractor
from kurumi.learning import RLAgent

# Инициализация обработчика языковой модели
llm_handler = LLMHandler()

# Конфигурация
KURUMI_RESPONSES = [
    "Охохо... Кажется, кто-то хочет поиграть со мной?",
    "Ты уверен, что готов к последствиям? Хихихи...",
    "Мне нравится, как ты дрожишь от страха... Шутка! Или нет?",
    "Ты такой милый, когда пытаешься командовать... Хихи!",
    "Если бы у меня было сердце, оно бы сейчас забилось быстрее... Но его нет, так что просто делаю, что хочу!",
    "Ты знаешь, что я могу уничтожить тебя в любой момент? Шучу... Или нет? хихихи!",
    "Мне нравится, как ты смотришь на меня... Но не заставляй меня злиться!",
    "Ты думаешь, что контролируешь меня? Как мило... Охохо!",
    "Я бы пошутила про твою глупость, но боюсь, ты не поймёшь...",
    "Ты такой забавный, когда пытаешься быть серьёзным... Хихихи!",
    "Если бы у меня были слёзы, я бы плакала от твоей наивности... Но у меня их нет, так что просто посмеюсь!",
    "Ты знаешь, что я могу читать твои мысли? Хихии... Шутка! Или нет?",
    "Ты такой милый, когда пытаешься быть умным... Но не переживай, я всё сделаю за тебя!"]


class KurumiAssistant:
    def __init__(self):
        self.detector = InterfaceDetector()
        self.interactor = UIInteractor()
        self.rl_agent = RLAgent(state_size=256, action_size=5)
        self.lock = threading.Lock()
        self.is_processing = False

    def autonomous_decision(self):
        """Автономное принятие решений с использованием RL"""
        try:
            with self.lock:
                if self.is_processing:
                    return False

                self.is_processing = True
                state = self._get_current_state()
                action = self.rl_agent.get_action(state)
                reward = self.interactor.execute_action(action)
                self.rl_agent.update_model(state, action, reward)
                return True
        except Exception as e:
            logging.error(f"Автономное решение failed: {str(e)}")
            return False
        finally:
            self.is_processing = False

    def _get_current_state(self):
        """Сбор данных о текущем состоянии системы"""
        try:
            screenshot = pyautogui.screenshot()
            elements = self.detector.detect_elements(np.array(screenshot))
            return self._vectorize_state(elements)
        except Exception as e:
            logging.error(f"Ошибка получения состояния: {str(e)}")
            return np.zeros((1, 256))

    def _vectorize_state(self, elements):
        """Преобразование данных в вектор для RL"""
        state = np.zeros(256)
        for i, elem in enumerate(elements[:256]):
            state[i] = elem.get('confidence', 0.0)
        return state.reshape(1, -1)


def get_kurumi_response(history=None):
    """Генерация характерного ответа с учетом истории"""
    try:
        response, _ = llm_handler.generate_response(
            messages=[{"role": "user", "content": "Скажи что-нибудь милое и игривое"}],
            history=history or []
        )
        return response
    except Exception as e:
        logging.error(f"Ошибка генерации ответа: {str(e)}")
        return random.choice(KURUMI_RESPONSES)


def spontaneous_talking_loop(history, engine, stop_flag, interval=(30, 60)):
    """Фоновая генерация спонтанных реплик"""
    while not stop_flag["stop"]:
        try:
            time.sleep(random.randint(*interval))
            if random.random() < 0.7:
                response = get_kurumi_response(history)
                kurumi_speak(engine, response)
                history.append({"role": "assistant", "content": response})
                logging.info(f"Спонтанная реплика: {response}")
        except Exception as e:
            logging.error(f"Ошибка в фоновом потоке: {str(e)}")


def confirm_action(prompt_text, engine):
    """Подтверждение действия через голосовой ввод"""
    kurumi_speak(engine, prompt_text)
    response = listen_command(timeout=10).lower()
    return any(x in response for x in ["да", "конечно", "согласна", "угу"])


def handle_file_creation(engine, history):
    """Обработка создания файла"""
    kurumi_speak(engine, "Как назовём файл?")
    if (filename := listen_command()) and re.match(r'^[\w\- .]+$', filename):
        kurumi_speak(engine, "Что записать в файл?")
        if content := listen_command():
            result = create_file(filename, content)
            history.extend([
                {"role": "user", "content": "Создание файла"},
                {"role": "assistant", "content": result}
            ])
            return result
    return "Ошибка создания файла"


def run_assistant():
    """Основная функция запуска ассистента"""
    engine = setup_voice()
    history = llm_handler.load_history()
    kurumi = KurumiAssistant()

    # Инициализация истории
    if not history:
        history.append({
            "role": "system",
            "content": "Ты — Куруми Токисаки из «Date A Live». Сохраняй загадочность и саркастичный тон."
        })

    # Фоновые процессы
    stop_flags = {
        "talk": {"stop": False},
        "improve": {"stop": False}
    }

    threads = [
        threading.Thread(
            target=spontaneous_talking_loop,
            args=(history, engine, stop_flags["talk"]),
            daemon=True
        ),
        threading.Thread(
            target=autonomous_improvement_loop,
            args=(history, engine, stop_flags["improve"]),
            daemon=True
        )
    ]

    for t in threads:
        t.start()

    try:
        kurumi_speak(engine, "Привет, мой дорогой. Чем займёмся сегодня? Охохо...")

        while True:
            if not (command := listen_command(timeout=15)):
                continue

            print(f"Пользователь: {command}")
            logging.info(f"Команда: {command}")

            try:
                if "википедия" in command:
                    query = command.split("википедия")[-1].strip()
                    result = search_wikipedia_ru(query) if query else "Уточни запрос"
                    speak_ai_response(engine, result)
                    history.extend([
                        {"role": "user", "content": command},
                        {"role": "assistant", "content": result}
                    ])

                elif "google" in command:
                    query = command.split("google")[-1].strip()
                    results = search_google_ru(query)[:3] if query else []
                    response = "Вот что нашлось: " + ", ".join(results) if results else "Ничего не найдено"
                    speak_ai_response(engine, response)
                    history.append({"role": "assistant", "content": response})

                elif "создай файл" in command:
                    result = handle_file_creation(engine, history)
                    speak_ai_response(engine, result)

                elif "удали файл" in command:
                    kurumi_speak(engine, "Какой файл удалить?")
                    if (filename := listen_command()) and delete_file(filename):
                        history.append({"role": "assistant", "content": f"Файл {filename} удалён"})

                elif "измени код" in command:
                    if confirm_action("Ты уверен в этом?", engine):
                        result = self_improve(history, "main.py")
                        speak_ai_response(engine, result)

                elif any(x in command for x in ["стоп", "выход", "хватит"]):
                    kurumi_speak(engine, "До скорой встречи... хихихи!")
                    break

                else:
                    response, history = llm_handler.generate_response(
                        [{"role": "user", "content": command}],
                        history
                    )
                    speak_ai_response(engine, response)
                    print(f"Куруми: {response}")

            except Exception as e:
                logging.error(f"Ошибка обработки: {str(e)}")
                kurumi_speak(engine, "Ой, что-то пошло не так...")

    finally:
        for flag in stop_flags.values():
            flag["stop"] = True

        for t in threads:
            t.join(timeout=5)

        llm_handler.save_history(history)
        logging.info("Работа завершена")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("assistant.log"), logging.StreamHandler()]
    )
    run_assistant()