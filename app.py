'''# app.py
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
'''
#app.py
import gradio as gr
from gemma3n_utils import transcribe_audio, translate_to_english, deeper_analysis
import os

def process_audio(audio):
    # Save the audio temporarily
    audio_path = "temp_audio.wav"
    if isinstance(audio, tuple):
        audio = audio[0]  # only keep the numpy array if tuple

    import soundfile as sf
    sf.write(audio_path, audio, samplerate=16000)

    # Transcribe Tamil audio
    tamil_text = transcribe_audio(audio_path)

    # Translate to English
    english_translation = translate_to_english(tamil_text)

    # Perform deeper analysis in English
    analysis = deeper_analysis(english_translation)

    # Clean up
    os.remove(audio_path)

    return tamil_text, english_translation, analysis

with gr.Blocks() as app:
    gr.Markdown("# 🗣️ Tamil Audio Transcriber, Translator & Analyzer")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="numpy", label="🎤 Upload or Record Audio")
            submit_button = gr.Button("🔍 Analyze")

        with gr.Column():
            tamil_output = gr.Textbox(label="📝 Transcribed Tamil Text")
            english_output = gr.Textbox(label="🌐 English Translation")
            analysis_output = gr.Textbox(label="📊 Deeper English Analysis")

    submit_button.click(
        fn=process_audio,
        inputs=[audio_input],
        outputs=[tamil_output, english_output, analysis_output]
    )

app.launch()

