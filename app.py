# app.py
import gradio as gr
from gemma3n_utils import transcribe_audio, translate_to_english, deeper_analysis

def process_audio(audio_path):
    transcription = transcribe_audio(audio_path)
    translation = translate_to_english(transcription)
    analysis = deeper_analysis(translation)
    return transcription, translation, analysis

iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="🎙️ Upload Audio"),
    outputs=[
        gr.Textbox(label=" Tamil Transcription"),
        gr.Textbox(label=" English Translation"),
        gr.Textbox(label=" Deeper Analysis")
    ],
    title="Tamil Audio Transcriber & Analyzer",
    description="Upload a Tamil audio file. Get transcription, English translation, and deeper content analysis."
)

iface.launch(server_name="0.0.0.0", server_port=7860)


