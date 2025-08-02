import os
import streamlit as st

# Unsloth must be imported before transformers for optimal patching!
from unsloth import FastModel

# Optional: Hide Streamlit warning when running outside browser
os.environ["STREAMLIT_RUN_ON_SAVE"] = "1"

import torch
from transformers import pipeline
import librosa
import tempfile

st.set_page_config(page_title="Tamil Oral Literature ASR + Summary")
st.title("Tamil Oral Literature ASR + Unsloth Summarizer")

@st.cache_resource
def load_models():
    # Load ASR (Whisper) pipeline (uses CPU or GPU 0)
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if torch.cuda.is_available() else -1)
    
    # Load Unsloth Gemma model and tokenizer
    # NOTE: Make sure you have a suitable NVIDIA GPU available for Unsloth.
    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3n-E4B-it",
        dtype = None,
        max_seq_length = 1024,
        load_in_4bit = True,
        full_finetuning = False,
    )
    device = next(model.parameters()).device
    return asr, model, tokenizer, device

asr, model, tokenizer, model_device = load_models()

st.markdown("Upload a `wav` or `mp3` sample from the [SPRINGLab/IndicTTS_Tamil](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Tamil) dataset or your own Tamil audio.")

uploaded_audio = st.file_uploader("Upload Tamil audio (.wav/.mp3)", type=["wav", "mp3"])
if uploaded_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(uploaded_audio.read())
        tmp_filepath = tmp_file.name

    # Audio preprocess
    audio_array, orig_sr = librosa.load(tmp_filepath, sr=None)
    audio_16k = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=16000)

    with st.spinner("Running Tamil ASR (Whisper)..."):
        asr_result = asr(audio_16k)
        transcript = asr_result.get("text", "").strip()
        st.subheader("Transcription")
        st.write(transcript if transcript else "No speech detected.")

    if transcript:
        prompt = f"Summarize the following Tamil oral literature:\n\n{transcript}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model_device) for k,v in inputs.items()}

        with st.spinner("Generating summary with Gemma 3n..."):
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=256,          # increase token cap for longer output
                num_beams=3,                 # optional: better output quality
                no_repeat_ngram_size=3,      # optional: less repetition
                early_stopping=True
            )
            summary = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        st.subheader("Summary")
        st.write(summary)

    # Remove temp file
    os.remove(tmp_filepath)
