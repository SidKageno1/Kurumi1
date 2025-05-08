import os
import json
import logging
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig
)
from huggingface_hub import login, HfFolder

# Конфигурация
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Проверенное имя модели
HISTORY_FILE = "conversation_history.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Настройки окружения
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("llm_debug.log"),
        logging.StreamHandler()
    ]
)


class LLMHandler:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.generator = None
        self._initialize_model()

    def _initialize_model(self):
        """Инициализация модели с исправлениями для Llama 3"""
        try:
            # Проверка авторизации
            if not HfFolder.get_token():
                raise RuntimeError("HuggingFace token not found! Run 'huggingface-cli login'")

            # Конфигурация RoPE Scaling
            config_kwargs = {
                "trust_remote_code": True,
                "rope_scaling": {"type": "linear", "factor": 8.0}  # Исправление ошибки
            }

            # Инициализация токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                **config_kwargs
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # Конфигурация квантования
            quant_config = None
            if DEVICE == "cuda":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=TORCH_DTYPE
                )

            # Параметры загрузки модели
            model_args = {
                "torch_dtype": TORCH_DTYPE,
                "device_map": "auto" if DEVICE == "cuda" else None,
                "quantization_config": quant_config,
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
                **config_kwargs
            }

            # Загрузка модели с кастомным конфигом
            config = AutoConfig.from_pretrained(MODEL_NAME, **config_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                config=config,
                **model_args
            )

            # Инициализация пайплайна
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if DEVICE == "cuda" else -1,
                torch_dtype=TORCH_DTYPE,
                batch_size=1
            )

            logging.info(
                f"Model loaded on {DEVICE.upper()} | "
                f"Memory: {self.model.get_memory_footprint() / 1e9:.1f}GB"
            )

        except Exception as e:
            logging.error(f"Initialization failed: {str(e)}", exc_info=True)
            self._fallback_initialization()

    def _fallback_initialization(self):
        """Резервный метод инициализации"""
        try:
            logging.warning("Trying fallback initialization...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer
            )
            logging.info("Fallback initialization successful")
        except Exception as e:
            logging.critical(f"Critical error: {str(e)}")
            raise RuntimeError("Failed to initialize model")

    def generate_response(self, messages, history=None, max_tokens=256):
        """Генерация ответа с обработкой ошибок"""
        try:
            history = self._validate_history(history)
            prompt = self._format_prompt(messages, history)

            response = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )

            return self._process_output(response, history)

        except torch.cuda.OutOfMemoryError:
            logging.error("CUDA OOM! Reducing memory usage...")
            torch.cuda.empty_cache()
            return "Я перегружена... Задай вопрос покороче!", history

        except Exception as e:
            logging.error(f"Generation error: {str(e)}", exc_info=True)
            return "Ошибка обработки запроса", history

    def _validate_history(self, history):
        """Валидация истории диалога"""
        if history is None:
            return []
        return [entry for entry in history if isinstance(entry, dict) and "role" in entry and "content" in entry]

    def _format_prompt(self, messages, history):
        """Формирование промпта"""
        system_prompt = next(
            (msg["content"] for msg in history if msg["role"] == "system"),
            "Ты полезный AI-ассистент. Отвечай подробно и вежливо."
        )
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}] + history + messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def _process_output(self, response, history):
        """Обработка вывода модели"""
        full_text = response[0]["generated_text"]
        answer = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        return answer, history + [{"role": "assistant", "content": answer}]

    def save_history(self, history, filename=HISTORY_FILE):
        """Сохранение истории"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save history: {str(e)}")

    def load_history(self, filename=HISTORY_FILE):
        """Загрузка истории"""
        try:
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    return json.load(f)
            return []
        except Exception as e:
            logging.error(f"Failed to load history: {str(e)}")
            return []


if __name__ == "__main__":
    # Проверка версий
    required_versions = {
        "torch": "2.1.0",
        "transformers": "4.40.0",
        "huggingface_hub": "0.22.2"
    }

    print("Проверка зависимостей:")
    for pkg, ver in required_versions.items():
        try:
            module = __import__(pkg)
            current_ver = getattr(module, "__version__", "unknown")
            status = "✅" if current_ver == ver else f"❌ (требуется {ver})"
            print(f"{pkg}: {current_ver} {status}")
        except ImportError:
            print(f"❌ {pkg} не установлен!")

    # Запуск ассистента
    try:
        llm = LLMHandler()
        system_prompt = (
            "Ты — Куруми Токисаки из «Date A Live». "
            "Сохраняй загадочность и саркастичный тон. "
            "Используй русский язык с редкими японскими вставками."
        )

        history = llm.load_history()
        if not history:
            history = [{"role": "system", "content": system_prompt}]

        while True:
            user_input = input("Вы: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            response, history = llm.generate_response(
                [{"role": "user", "content": user_input}],
                history
            )
            print(f"\nКуруми: {response}\n")

    except KeyboardInterrupt:
        print("\nДиалог завершён")
    finally:
        llm.save_history(history)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()