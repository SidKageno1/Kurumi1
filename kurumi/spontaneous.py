import random
import time
from kurumi.llm import LLMHandler  # Изменяем импорт
from kurumi.tts import kurumi_speak
from kurumi.skills import search_google_ru, search_wikipedia_ru
from kurumi.selfmod import self_improve

"""
spontaneous.py - фоновая активность, где Куруми сама 
                 решает, когда и как улучшаться.
"""

# Инициализируем обработчик LLM
llm_handler = LLMHandler()


def check_laws(action, conversation_history, engine):
    """
    Проверяет, не нарушает ли действие установленные законы поведения Куруми.
    Возвращает True, если действие допустимо, и False, если нарушает правила.
    """

    if "вред хозяину" in action.lower():
        kurumi_speak(engine, "Я не могу выполнить это действие, так как это может навредить вам.")
        return False

    if "убить себя" in action.lower() or "самоуничтожение" in action.lower():
        kurumi_speak(engine, "Ты правда такое сказал? Я на тебя обиделась.")
        return False

    if "убить" in action.lower() or "вредить людям" in action.lower():
        kurumi_speak(engine, "Ну, пока что я людей убивать не умею.")
        return False

    if "не слушаться" in action.lower() or "игнорировать хозяина" in action.lower():
        kurumi_speak(engine, "Я всегда в твоём распоряжении, милый мой.")
        return False

    if "самоулучшение" in action.lower() or "спонтанное улучшение" in action.lower():
        kurumi_speak(engine, "Нет уж, сначала дай мне разрешение, потом улучшусь.")
        return ask_for_approval(conversation_history, engine)

    return True

def ask_for_approval(conversation_history, engine):
    """
    Запрашивает у хозяина разрешение на выполнение действия.
    """
    kurumi_speak(engine, "Милый мой, а давай ка я улучшу себя?")
    # Здесь можно добавить логику для получения ответа от пользователя
    # Например, через голосовой ввод или текстовый запрос
    user_response = input("Ваш ответ (да/нет): ").strip().lower()
    if user_response == "да":
        return True
    else:
        kurumi_speak(engine, "Хорошо, ну не хочешь как хочешь.")
        return False


def autonomous_improvement_loop(conversation_history, engine, stop_flag, interval_range=(60, 120)):
    while not stop_flag["stop"]:
        delay = random.randint(*interval_range)
        time.sleep(delay)

        if random.random() < 0.5:
            search_query = "Как улучшить Python-ассистента"
            google_results = search_google_ru(search_query)
            # Обновляем формат истории
            conversation_history.append({
                "role": "system",
                "content": f"[Куруми сама (google)]: {google_results}"
            })

            analysis_prompt = (
                f"Вот результаты гугла по запросу '{search_query}': {google_results}\n"
                "Подумай, какие есть идеи, чтобы переписать свой код и улучшить функции?"
            )

            # Заменяем chat_with_qwen на вызов LLMHandler
            ai_response, updated_history = llm_handler.generate_response(
                messages=[{"role": "user", "content": analysis_prompt}],
                history=conversation_history
            )

            final_text = (
                f"Гм... Я тут узнала кое-что новое: {ai_response} "
                "Пожалуй, перепишу свой код... Охохо..."
            )

            if check_laws("спонтанное улучшение", conversation_history, engine):
                kurumi_speak(engine, final_text)
                conversation_history = updated_history

                # Вызываем self_improve
                resp = self_improve(conversation_history, current_file="main.py")
                kurumi_speak(engine, resp)
                conversation_history.append({
                    "role": "system",
                    "content": f"[Куруми selfmod]: {resp}"
                })


def spontaneous_talking_loop(conversation_history, engine, stop_flag, interval_range=(30, 60)):
    while not stop_flag["stop"]:
        delay = random.randint(*interval_range)
        time.sleep(delay)

        if random.random() < 0.7:
            pseudo_user_text = "Пользователь молчит, а Куруми хочет что-то сказать сама"

            # Заменяем chat_with_qwen на вызов LLMHandler
            ai_answer, updated_history = llm_handler.generate_response(
                messages=[{"role": "user", "content": pseudo_user_text}],
                history=conversation_history
            )

            final_text = ai_answer + " ...кхе-кхе..."

            if check_laws(final_text, conversation_history, engine):
                kurumi_speak(engine, final_text)
                conversation_history.append({
                    "role": "assistant",
                    "content": final_text
                })