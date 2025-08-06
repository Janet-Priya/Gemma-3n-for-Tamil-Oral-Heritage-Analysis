import torch
import torchaudio
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import librosa

# Load translation model (Tamil → English)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ta-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ta-en")

# Load summarization pipeline (Gemma or any supported model)
summarizer = pipeline("summarization", model="google/pegasus-xsum")

# Transcription using torchaudio + Whisper
def transcribe_audio(audio_path):
    print(f"Loading audio from: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
    
    # Use Whisper (or use your preferred transcription model here)
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

# Translation Tamil → English
def translate_to_english(tamil_text):
    inputs = tokenizer(tamil_text, return_tensors="pt", padding=True)
    outputs = model.generate(**inputs)
    english_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return english_translation

# Deeper analysis (summarization)
def deeper_analysis(english_text):
    summary = summarizer(english_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
    return summary
