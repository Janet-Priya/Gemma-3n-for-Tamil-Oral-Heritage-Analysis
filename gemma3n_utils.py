'''# gemma3n_utils.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import torchaudio
import librosa
import soundfile as sf
import tempfile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Whisper model for Tamil transcription
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if torch.cuda.is_available() else -1)

# Load Gemma model for translation + analysis
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-2b-it",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
model.eval()

# Transcribe audio using Whisper
def transcribe_audio(audio_path):
    print(f"Transcribing: {audio_path}")
    audio_array, sr = torchaudio.load(audio_path)
    audio_array = audio_array[0].numpy()
    audio_16k = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    transcription = asr(audio_16k)
    return transcription["text"]

# Translate Tamil to English using Gemma
def translate_to_english(text):
    print("Translating Tamil to English...")
    prompt = f"Translate this to English:\n{text}\n\nEnglish:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

# Deeper analysis in English using Gemma
def deeper_analysis(text):
    print("Performing deeper analysis...")
    prompt = f"Analyze this English content in depth:\n{text}\n\nAnalysis:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()











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
'''
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Gemma model for translation/analysis
tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model_gemma = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it").to("cuda" if torch.cuda.is_available() else "cpu")

def gemma_generate(prompt):
    inputs = tokenizer_gemma(prompt, return_tensors="pt").to(model_gemma.device)
    outputs = model_gemma.generate(**inputs, max_new_tokens=100)
    return tokenizer_gemma.decode(outputs[0], skip_special_tokens=True)

def translate_to_english(tamil_text):
    prompt = f"Translate the following Tamil text to English:\n{tamil_text}\nTranslation:"
    return gemma_generate(prompt)

def deeper_analysis(english_text):
    prompt = f"Give a deep analysis of the following statement:\n{english_text}\nAnalysis:"
    return gemma_generate(prompt)
