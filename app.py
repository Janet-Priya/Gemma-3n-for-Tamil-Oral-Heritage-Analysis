import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import librosa
import tempfile
import os

st.set_page_config(page_title="Tamil Oral Literature ASR + Summary Demo (CPU)")
st.title("Tamil Oral Literature ASR + Summarizer Demo (CPU)")

@st.cache_resource(show_spinner=False)
def load_models():
    # Load Whisper ASR pipeline (CPU)
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=-1  # CPU
    )
    # Load summarization model and tokenizer with use_fast=False to avoid tokenizer errors
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=-1  # CPU
    )
    return asr, summarizer

asr, summarizer = load_models()

st.markdown("""
Upload a Tamil speech audio file (.wav or .mp3). The app will transcribe the speech and generate a short summary.

*Note:* This demo runs fully on CPU and uses a small summarization model as a proxy for Gemma 3n.
""")

uploaded_audio = st.file_uploader("Upload Tamil Audio", type=["wav", "mp3"])

if uploaded_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(uploaded_audio.read())
        tmp_filepath = tmp_file.name

    # Load and resample audio to 16 kHz as expected by Whisper
    audio_array, original_sr = librosa.load(tmp_filepath, sr=None)
    audio_16k = librosa.resample(audio_array, orig_sr=original_sr, target_sr=16000)

    with st.spinner("Transcribing audio with Whisper..."):
        asr_result = asr(audio_16k)
        transcript = asr_result.get("text", "").strip()

    st.subheader("Transcription")
    if transcript:
        st.write(transcript)
    else:
        st.write("No speech detected or transcription failed.")

    if transcript:
        with st.spinner("Generating summary with mT5..."):
            summary_result = summarizer(
                transcript,
                max_length=80,
                min_length=15,
                do_sample=False
            )
            summary = summary_result[0]["summary_text"]

        st.subheader("Summary")
        st.write(summary)

    # Clean up temporary file
    os.remove(tmp_filepath)
