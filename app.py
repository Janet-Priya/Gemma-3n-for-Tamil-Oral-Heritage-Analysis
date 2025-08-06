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
'''

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
    demo.launch()
