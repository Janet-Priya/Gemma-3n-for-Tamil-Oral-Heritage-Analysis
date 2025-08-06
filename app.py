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

'''
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

'''

# app.py
import gradio as gr
import os
from gemma3n_utils import transcribe_audio, translate_to_english, deeper_analysis
import torchaudio

def process_audio(audio):
    if audio is None:
        return "No audio provided", "", ""

    print("Audio received:", audio)

    if isinstance(audio, tuple):
        audio_np, sample_rate = audio
    else:
        audio_path = audio if isinstance(audio, str) else audio.name
        audio_np, sample_rate = torchaudio.load(audio_path)

    # Transcribe
    transcription = transcribe_audio(audio_np.numpy(), sample_rate)

    # Translate
    translation = translate_to_english(transcription)

    # Deeper Analysis
    analysis = deeper_analysis(translation)

    return transcription, translation, analysis

# UI
custom_css = """
footer {visibility: hidden}
h1 {text-align: center;}
.gradio-container {font-family: 'Segoe UI', sans-serif; background-color: #f5f5f5;}
"""

with gr.Blocks(css=custom_css) as interface:
    gr.Markdown("# 🎧 Tamil Audio Translator + Analyzer")
    with gr.Row():
        audio_input = gr.Audio(type="numpy", label="🎙️ Record or Upload Audio")
    with gr.Row():
        trans_output = gr.Textbox(label="📝 Transcription (Tamil)", lines=2)
        trans_en_output = gr.Textbox(label="🌐 Translated (English)", lines=2)
    with gr.Row():
        analysis_output = gr.Textbox(label="🧠 Deeper Analysis", lines=4)

    submit_btn = gr.Button("Process Audio")
    submit_btn.click(fn=process_audio, inputs=[audio_input], outputs=[trans_output, trans_en_output, analysis_output])

interface.launch(share=True)

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

iface.launch(share=True)

import gradio as gr
import os
from gemma3n_utils import transcribe_audio, translate_to_english, deeper_analysis

def process_audio(audio_file):
    if audio_file is None:
        return "Please upload an audio file.", "", ""

    # Save the uploaded file to a temporary location
    audio_path = "temp_audio.wav"
    audio_file.save(audio_path)

    # Transcribe
    transcription = transcribe_audio(audio_path)

    # Translate
    translation = translate_to_english(transcription)

    # Analyze
    analysis = deeper_analysis(translation)

    # Clean up temp file
    os.remove(audio_path)

    return transcription, translation, analysis

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🎙️ Tamil Audio Summarizer & Translator</h1>")

    with gr.Row():
        audio_input = gr.Audio(label="Upload Tamil Audio File", type="file")
        submit_btn = gr.Button("Submit")

    with gr.Row():
        tamil_output = gr.Textbox(label="📝 Tamil Transcription")
        english_output = gr.Textbox(label="🌐 English Translation")
        analysis_output = gr.Textbox(label="🔍 Deeper Analysis")

    submit_btn.click(fn=process_audio, inputs=[audio_input], outputs=[tamil_output, english_output, analysis_output])

demo.launch()



'''


