# gemma3n_utils.py

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

