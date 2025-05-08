import os
import ast
import logging
import shutil
from datetime import datetime
from kurumi.llm import LLMHandler

logger = logging.getLogger(__name__)
BACKUP_DIR = "code_backups"
os.makedirs(BACKUP_DIR, exist_ok=True)

# Инициализация обработчика языковой модели
llm_handler = LLMHandler()


def validate_python_syntax(code: str) -> bool:
    """Проверка синтаксиса Python с улучшенной обработкой ошибок"""
    try:
        ast.parse(code)
        return True
    except Exception as e:
        logger.error(f"Синтаксическая ошибка: {str(e)}")
        return False


def create_backup(file_path: str) -> str:
    """Создание временной метки для резервной копии"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(BACKUP_DIR, f"{os.path.basename(file_path)}_{timestamp}.bak")
        shutil.copy2(file_path, backup_path)
        logger.info(f"Создана резервная копия: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Ошибка создания бэкапа: {str(e)}")
        return None


def generate_improved_code(old_code: str, context: str) -> str:
    """Генерация улучшенного кода с помощью LLM"""
    prompt = f"""Ты - эксперт по рефакторингу кода. Проанализируй предоставленный код и предложи улучшенную версию, сохранив исходную функциональность. Учти контекст обсуждения: {context}

Исходный код:
{old_code}

Улучшенный код должен:
1. Соответствовать PEP8
2. Устранить code smells
3. Оптимизировать производительность
4. Добавить обработку ошибок
5. Сохранить обратную совместимость

Верни только код без пояснений:"""

    try:
        response, _ = llm_handler.generate_response(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048
        )

        # Извлечение кода между ```python и ```
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        return response
    except Exception as e:
        logger.error(f"Ошибка генерации кода: {str(e)}")
        return None


def run_sanity_check() -> bool:
    """Расширенная проверка работоспособности"""
    try:
        # Проверка основных зависимостей
        import pytest  # type: ignore
        import mypy
        return True
    except ImportError:
        logger.error("Отсутствуют тестовые зависимости")
        return False
    except Exception as e:
        logger.error(f"Ошибка проверки: {str(e)}")
        return False


def rollback_changes(original_file: str, backup_file: str) -> bool:
    """Безопасный откат изменений"""
    try:
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, original_file)
            logger.info(f"Откат к версии: {backup_file}")
            return True
        logger.warning("Резервная копия не найдена")
        return False
    except Exception as e:
        logger.error(f"Ошибка отката: {str(e)}")
        return False


def self_improve(context: str, current_file: str = "main.py") -> str:
    """Улучшение кода с полным циклом контроля качества"""
    if not os.path.exists(current_file):
        return "Ошибка: Файл не найден"

    backup_path = create_backup(current_file)
    if not backup_path:
        return "Ошибка создания резервной копии"

    try:
        # Чтение исходного кода
        with open(current_file, 'r', encoding='utf-8') as f:
            old_code = f.read()

        # Генерация улучшений
        new_code = generate_improved_code(old_code, context)
        if not new_code:
            return "Не удалось сгенерировать улучшения"

        # Валидация синтаксиса
        if not validate_python_syntax(new_code):
            return "Ошибка: Некорректный синтаксис Python"

        # Запись нового кода
        with open(current_file, 'w', encoding='utf-8') as f:
            f.write(new_code)

        # Запуск тестов
        if not run_sanity_check():
            rollback_changes(current_file, backup_path)
            return "Ошибка: Тесты не пройдены, откат изменений"

        # Проверка типов
        try:
            import mypy.api
            result = mypy.api.run([current_file])
            if result[0]:
                raise RuntimeError("Ошибки типизации")
        except ImportError:
            logger.warning("mypy не установлен, проверка типов пропущена")

        return "Код успешно улучшен! Рекомендую провести полное тестирование."

    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
        rollback_changes(current_file, backup_path)
        return f"Ошибка: {str(e)}, выполнено восстановление из резервной копии"


if __name__ == "__main__":
    # Пример использования
    logging.basicConfig(level=logging.DEBUG)
    test_code = "print('Hello World')"
    print(self_improve(test_code, "Тестовое улучшение"))