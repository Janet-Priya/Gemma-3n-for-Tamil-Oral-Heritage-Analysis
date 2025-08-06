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

# app.py
import gradio as gr
from gemma3n_utils import transcribe_audio, translate_to_english, deeper_analysis
import torch
import torchaudio
import tempfile

def process_audio(audio):
    print(f"Received audio input: {audio}")

    if isinstance(audio, tuple):
        # Microphone input: (sample_rate, numpy_array)
        sample_rate, data = audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        torchaudio.save(temp_file.name, torch.tensor(data).unsqueeze(0), sample_rate=sample_rate)
        audio_path = temp_file.name
    elif isinstance(audio, str):
        # File upload
        audio_path = audio
    else:
        raise ValueError("Unsupported audio input type")

    print(f"Transcribing: {audio_path}")
    transcription = transcribe_audio(audio_path)
    translated = translate_to_english(transcription)
    analysis = deeper_analysis(translated)

    return transcription, translated, analysis

with gr.Blocks(title="Tamil Audio Transcriber & Translator") as demo:
    gr.Markdown("""
    # 🎙️ Tamil Audio Transcriber
    Upload or record your Tamil audio.
    Get instant transcription 📝, translation 🌍, and deep analysis 📊 in English!
    """)

    with gr.Row():
        audio_input = gr.Audio(source="microphone", type="numpy", label="Upload or Record Audio")

    with gr.Row():
        transcribed_text = gr.Textbox(label="Tamil Transcription")
        translated_text = gr.Textbox(label="English Translation")
        analysis_text = gr.Textbox(label="Deeper Analysis")

    submit_btn = gr.Button("🔍 Transcribe & Analyze")
    submit_btn.click(fn=process_audio, inputs=audio_input, outputs=[transcribed_text, translated_text, analysis_text])

if __name__ == "__main__":
    demo.launch()
