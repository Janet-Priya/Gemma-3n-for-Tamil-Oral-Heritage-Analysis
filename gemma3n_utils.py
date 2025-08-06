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


# gemma3n_utils.py
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
import torchaudio

# Load models once at the start
processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
model.to("cuda" if torch.cuda.is_available() else "cpu")

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if torch.cuda.is_available() else -1,
)

def transcribe_audio(audio_path):
    print(f"Transcribing file: {audio_path}")
    result = pipe(audio_path)
    return result["text"]

def translate_to_english(tamil_text):
    # Dummy placeholder for actual translation logic (e.g. using NLLB or MarianMT)
    print(f"Translating: {tamil_text}")
    return f"[Translated] {tamil_text}"

def deeper_analysis(english_text):
    # Dummy placeholder for NLP techniques (summarization, emotion, sentiment, etc.)
    print(f"Analyzing: {english_text}")
    return f"[Analysis] This appears to be a sample Tamil audio about literature or conversation."

# gemma3n_utils.py
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torch
import torchaudio

# Load models once
processor = AutoProcessor.from_pretrained("openai/whisper-small")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
model.to("cuda" if torch.cuda.is_available() else "cpu")

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
)

def transcribe_audio(audio_np, sample_rate):
    print("Transcribing uploaded audio")
    waveform = torch.tensor(audio_np).float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
    temp_file = "/tmp/temp.wav"
    torchaudio.save(temp_file, resampled, 16000)
    result = pipe(temp_file)
    return result["text"]

def translate_to_english(tamil_text):
    print(f"Translating: {tamil_text}")
    # Replace with your translation model or API
    return f"[Translated to English] {tamil_text}"

def deeper_analysis(english_text):
    print(f"Analyzing: {english_text}")
    # You can add keyword extraction, sentiment, tone, etc.
    return (
        f"[Deeper Analysis]\n"
        f"- Summary: This might be a Tamil literature excerpt or conversation.\n"
        f"- Sentiment: Neutral or descriptive tone.\n"
        f"- Insight: Rich in cultural context, potentially poetic."
    )


'''


# gemma3n_utils.py
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import torchaudio
import os

# Load the Gemma model and processor (adjust to your actual model path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
processor = AutoProcessor.from_pretrained("google/gemma-1.1-2b-it")
model.to(device)

def transcribe_audio(audio_path):
    print(f"Transcribing: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    # Resample if needed
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    inputs = processor(text="Transcribe this Tamil audio:", audio=waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    transcription = processor.decode(outputs[0], skip_special_tokens=True)
    return transcription

def translate_to_english(text):
    print("Translating Tamil to English...")
    inputs = processor(text="Translate this to English: " + text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    translation = processor.decode(outputs[0], skip_special_tokens=True)
    return translation

def deeper_analysis(text):
    print("Performing deeper analysis...")
    prompt = f"Analyze this English content in depth:\n{text}"
    inputs = processor(text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    analysis = processor.decode(outputs[0], skip_special_tokens=True)
    return analysis


