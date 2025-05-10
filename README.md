# Kurumi — Self-Improving AI Desktop Assistant  
> “A little **Jarvis-style** helper that listens, thinks and speaks back.”

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)  
![Status](https://img.shields.io/badge/status-experimental-orange)  
![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen)

Kurumi aims to be a **voice-controlled, on-device AI companion** that can:

1. Recognise your spoken commands  
2. Reason on top of an LLM (OpenAI by default)  
3. Speak the answer back with high-quality TTS  
4. Trigger local PC actions (open apps, search the web, etc.)  
5. *Coming soon*: learn from its own logs and refactor itself

---

## ✨ Features
| Area | Modules / Assets | What it does |
|------|------------------|--------------|
| **Speech I/O** | `main.py`, `tts-env/` | Hot-word detection, Whisper/STT, Edge-TTS or VoiceVox output |
| **Core Brain** | `kurumi/` package | Dialogue loop, tool-calling, simple memory store |
| **Computer Vision** | `yolov8n.pt` | Optional object-detection for “What’s on my desk?” queries |
| **Self-Update** | experimental | Analyses its own logs ➜ suggests code patches (disabled by default) |



---

## 🚀 Quick Start

```bash
# 1. Clone and enter
git clone https://github.com/SidKageno1/Kurumi1.git
cd Kurumi1

# 2. Create an isolated env (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Provide your keys 
well here i'm using hugging_face and Llama-3.1 (it is located in llm.py)
but feel free to change it

# 5. Run!
python main.py



P.S: As i sad before the project is raw and it nedds more cooking so fi u are intrested feel free to jump in!
