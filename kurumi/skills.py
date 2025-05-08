import os
import wikipedia
from googlesearch import search


"""
skills.py - набор функций (скиллов) для ассистента:
   - поиск в Википедии
   - поиск в Google
   - работа с файлами
   - обучение PPO       
"""

def search_wikipedia_ru(query):
    wikipedia.set_lang("ru")
    try:
        result = wikipedia.summary(query, sentences=2)
        return result
    except wikipedia.exceptions.DisambiguationError:
        return "Слишком много значений. Уточни, пожалуйста."
    except wikipedia.exceptions.PageError:
        return "Ничего не нашла, извини."

def search_google_ru(query):
    return [url for url in search(query, num_results=5)]

def create_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Файл {filename} создан!"

def read_file(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return "Вот что там написано:\n" + f.read()
    else:
        return f"Файл {filename} не найден."

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        return f"Файл {filename} удалён."
    else:
        return f"Не вижу такого файла."
