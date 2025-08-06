'''import streamlit as st
import tempfile
import os

from gemma3n_utils import transcribe_audio, summarize_text

st.title("Tamil ASR and Literary Summarization with Gemma 3n")

audio_file = st.file_uploader("Upload Tamil audio (*.wav, *.flac, *.mp3)", type=["wav", "flac", "mp3"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_path = temp_audio.name

    st.audio(temp_path)

    with st.spinner("Transcribing..."):
        transcription = transcribe_audio(temp_path)
    st.subheader("Transcription")
    st.write(transcription)

    with st.spinner("Summarizing..."):
        summary = summarize_text(transcription)
    st.subheader("Summary")
    st.write(summary)
#
    os.remove(temp_path)
else:
    st.info("Please upload a Tamil audio file.")


import gradio as gr
from gemma3n_utils import transcribe_audio, summarize_text

def full_pipeline(audio_path):
    transcription = transcribe_audio(audio_path)
    summary = summarize_text(transcription)
    return transcription, summary

demo = gr.Interface(
    fn=full_pipeline,
    inputs=gr.Audio(type="filepath", label="Upload Tamil audio (wav, flac, mp3)"),
    outputs=[gr.Textbox(label="Transcription"), gr.Textbox(label="Summary")],
    title="Tamil ASR and Literary Summarization with Gemma 3n",
    description="Upload an audio file with Tamil speech to transcribe and summarize."
)

if __name__ == "__main__":
    demo.launch(share=True)
'''


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import torchaudio
import tempfile
import librosa
import soundfile as sf

# Load Whisper model for Tamil audio transcription
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if torch.cuda.is_available() else -1)

# Load Gemma 3n model for summarization
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

# Transcription function
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        audio_array, sr = torchaudio.load(tmp.name)
        audio_array = audio_array[0].numpy()  # Convert from tensor to numpy
        audio_16k = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        transcription = asr(audio_16k)
        return transcription["text"]

# Summarization function
def summarize_text(text):
    prompt = f"தமிழில் இந்த உரையை சுருக்கவும்:\n{text}\n\nசுருக்கம்:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.replace(prompt, "").strip()
