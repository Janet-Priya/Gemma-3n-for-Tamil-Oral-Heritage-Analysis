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

'''
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





