"""
Docstring for chat_bot_voice_to_voice

Create Python or Conda evnironment ------- 
for python 
python -m venv .venv

.venv\Scripts\activate.bat

for conda 
conda create .venv
conda activate .venv

install following library 

pip install gradio faster_whisper pyttsx3 ollama

"""



import os
import shutil
import requests
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import gradio as gr
import pyttsx3
from faster_whisper import WhisperModel

# =========================================================
# CONFIGURATION
# =========================================================

OLLAMA_API_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mistral"

AUDIO_SAVE_DIR = "recordings"
WHISPER_MODEL_SIZE = "small"

os.makedirs(AUDIO_SAVE_DIR, exist_ok=True)

# =========================================================
# MODEL INITIALIZATION
# =========================================================

class SpeechToText:
    def __init__(self, model_size: str):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(segment.text for segment in segments).strip()


class TextToSpeech:
    def __init__(self, rate: int = 175):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def speak(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()


class OllamaClient:
    def __init__(self, api_url: str, model: str):
        self.api_url = api_url
        self.model = model

    def chat(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        response = requests.post(self.api_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["message"]["content"]


# =========================================================
# SERVICE INSTANCES
# =========================================================

stt_service = SpeechToText(WHISPER_MODEL_SIZE)
tts_service = TextToSpeech()
llm_service = OllamaClient(OLLAMA_API_URL, OLLAMA_MODEL)

# =========================================================
# CORE PIPELINE
# =========================================================

def save_audio(audio_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = os.path.join(AUDIO_SAVE_DIR, f"user_{timestamp}.wav")
    shutil.copy(audio_path, destination)
    return destination


def format_conversation(history: List[Dict[str, str]]) -> str:
    formatted = []
    for message in history:
        prefix = "User" if message["role"] == "user" else "Assistant"
        formatted.append(f"{prefix}: {message['content']}")
    return "\n\n".join(formatted)


def handle_voice_interaction(
    audio_path: Optional[str],
    history: List[Dict[str, str]]
) -> Tuple[List[Dict[str, str]], str]:

    if audio_path is None:
        return history, "No audio submitted."

    try:
        # Save audio
        saved_audio = save_audio(audio_path)

        # Speech to text
        user_text = stt_service.transcribe(saved_audio)
        if not user_text:
            return history, "Could not understand speech."

        history = history or []
        history.append({"role": "user", "content": user_text})

        # LLM response
        assistant_text = llm_service.chat(history)
        history.append({"role": "assistant", "content": assistant_text})

        # Text to speech
        tts_service.speak(assistant_text)

        # UI output
        conversation_view = format_conversation(history)
        return history, conversation_view

    except Exception as error:
        return history, f"Error: {str(error)}"


# =========================================================
# UI LAYER (GRADIO)
# =========================================================

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Professional Voice Assistant") as demo:

        gr.Markdown("## Voice Assistant (Start / Stop / Submit)")
        gr.Markdown(
            "• Click **Start Recording** → Speak\n"
            "• Click **Stop Recording**\n"
            "• Click **Submit** to send audio to the assistant\n\n"
            "Conversation continues until the browser is closed."
        )

        conversation_state = gr.State([])

        audio_input = gr.Audio(
            sources=["microphone"],
            type="filepath",
            visible=False,
            label="Microphone Input"
        )

        with gr.Row():
            start_btn = gr.Button(" Start Recording")
            stop_btn = gr.Button("Stop Recording")
            submit_btn = gr.Button("Submit")

        conversation_box = gr.Textbox(
            label="Conversation",
            lines=14
        )

        # Button bindings
        start_btn.click(
            lambda: gr.update(visible=True),
            outputs=audio_input
        )

        stop_btn.click(
            lambda: gr.update(visible=False),
            outputs=audio_input
        )

        submit_btn.click(
            fn=handle_voice_interaction,
            inputs=[audio_input, conversation_state],
            outputs=[conversation_state, conversation_box]
        )

    return demo


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(inbrowser=True)
