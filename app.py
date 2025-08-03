import streamlit as st
from transformers import pipeline
import librosa
import tempfile

st.set_page_config(page_title="Tamil Oral Literature ASR + Summary")
st.title("Tamil Oral Literature ASR + Summarizer Demo (Public Web)")

@st.cache_resource
def load_models():
    # ASR: Whisper (smallest suitable for CPU hosting)
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",  # or "openai/whisper-tiny"
        device=-1,                    # -1 = CPU
    )
    # Summarization: MT5 Small (multilingual, CPU friendly)
    summarizer = pipeline(
        "summarization",
        model="google/mt5-small",
        tokenizer="google/mt5-small",
        device=-1,
    )
    return asr, summarizer

asr, summarizer = load_models()

st.markdown("Upload a `wav` or `mp3` Tamil sample to transcribe and summarize.")

uploaded_audio = st.file_uploader("Upload Tamil audio", type=["wav", "mp3"])
if uploaded_audio is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
        tmp_file.write(uploaded_audio.read())
        tmp_filepath = tmp_file.name

    audio_array, orig_sr = librosa.load(tmp_filepath, sr=None)
    audio_16k = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=16000)

    with st.spinner("Transcribing (Whisper)..."):
        asr_result = asr(audio_16k)
        transcript = asr_result.get("text", "").strip()
        st.subheader("Transcription")
        st.write(transcript if transcript else "No speech detected.")

    if transcript:
        with st.spinner("Summarizing (MT5)..."):
            # MT5 expects at least a few words; may need to adjust prompt/length for best results
            summary_result = summarizer(
                transcript, max_length=80, min_length=15, do_sample=False
            )
            summary = summary_result[0]['summary_text']
        st.subheader("Summary")
        st.write(summary)

    # Remove temp file
    import os; os.remove(tmp_filepath)
