'''import os
import sys
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForImageTextToText
from accelerate import disk_offload

# Get HF token from environment variable (set as secret in Spaces for security)
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    sys.stderr.write("Warning: HF_TOKEN not set, model loading may fail for gated models.\n")

GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"

# Load processor and model (no manual .to(device) used)
processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=HF_TOKEN
)
model = AutoModelForImageTextToText.from_pretrained(...)  # without device_map
model = disk_offload(model, offload_dir="./your_offload_folder")


def transcribe_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True).numpy()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": waveform},
                {"type": "text", "text": "Transcribe this audio in Tamil."}
            ]
        }
    ]
    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    # Get device from (part of) model under accelerate
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device, dtype=model.dtype)

    outputs = model.generate(**input_ids, max_new_tokens=128)
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

def summarize_text(text):
    prompt = f"Summarize the following Tamil literature: {text}\nSummary:"
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    model_device = next(model.parameters()).device
    input_ids = input_ids.to(model_device, dtype=model.dtype)

    outputs = model.generate(**input_ids, max_new_tokens=150)
    summary = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return summary

import os
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForImageTextToText
import torchaudio

# Get HuggingFace token from environment variable or None
HF_TOKEN = os.environ.get("HF_TOKEN", None)
GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"

# Load the processor and model - no disk_offload to avoid complexity on Colab
processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=HF_TOKEN
)
model.eval()  # set to eval mode

def transcribe_audio(audio_path):
    waveform, sample_rate = sf.read(audio_path)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)  # convert to mono if stereo

    waveform = np.asarray(waveform, dtype=np.float32)[None, :]  # shape (1, samples)

    # Resample if needed
    if sample_rate != 16000:
        waveform_t = torch.from_numpy(waveform)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform_t = resampler(waveform_t)
        waveform = waveform_t.numpy()

    # Prepare messages for the processor
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": waveform},
                {"type": "text", "text": "Transcribe this audio in Tamil."}
            ]
        }
    ]

    # Get input tokens
    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )

    # Move inputs to model device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Run generation
    outputs = model.generate(**input_ids, max_new_tokens=128)
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription

def summarize_text(text):
    prompt = f"Summarize the following Tamil literature: {text}\nSummary:"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    outputs = model.generate(**input_ids, max_new_tokens=150)
    summary = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return summary


import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForImageTextToText

# Set model ID and HF token
HF_TOKEN = os.environ.get("HF_TOKEN", None)
GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"

# Load processor and model
processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=HF_TOKEN,
)
model.eval()

def transcribe_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError(f"Audio file does not exist or path is invalid: {audio_path}")

    try:
        waveform, sample_rate = sf.read(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    # Convert stereo to mono if needed
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {waveform.shape}")

    # Convert to float32 and make it 2D: (batch_size=1, time)
    waveform = np.asarray(waveform, dtype=np.float32)
    waveform = waveform[None, :]  # shape = (1, time)

    # Resample to 16kHz
    if sample_rate != 16000:
        waveform_tensor = torch.from_numpy(waveform)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform_tensor = resampler(waveform_tensor)
    else:
        waveform_tensor = torch.from_numpy(waveform)

    # Pad/truncate to 30s (480k samples)
    max_samples = 30 * 16000
    waveform_tensor = torch.nn.functional.pad(waveform_tensor, (0, max(0, max_samples - waveform_tensor.shape[1])))
    waveform_tensor = waveform_tensor[:, :max_samples]

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    waveform_tensor = waveform_tensor.to(device)

    # Convert to numpy (batch, time)
    waveform_np = waveform_tensor.cpu().numpy()

    # Prompt Gemma 3n to transcribe
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": waveform_np},
                {"type": "text", "text": "Transcribe this audio in Tamil."},
            ],
        }
    ]
    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(device)

    outputs = model.generate(**input_ids, max_new_tokens=128)
    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription


def summarize_text(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("Input text to summarize must be a non-empty string.")

    prompt = f"Summarize the following Tamil literature: {text}\nSummary:"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to(device)

    outputs = model.generate(**input_ids, max_new_tokens=150)
    summary = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return summary

'''


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import torchaudio
import tempfile
import librosa
import soundfile as sf

# Load Whisper model for Tamil audio transcription
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0 if torch.cuda.is_available() else -1)

# Load Gemma 3n model for summarization
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
model.eval()

if torch.cuda.is_available():
    model = model.to("cuda")

# Transcription function
def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()

        audio_array, sr = torchaudio.load(tmp.name)
        audio_array = audio_array[0].numpy()  # Convert from tensor to numpy
        audio_16k = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        transcription = asr(audio_16k)
        return transcription["text"]

# Summarization function
def summarize_text(text):
    prompt = f"தமிழில் இந்த உரையை சுருக்கவும்:\n{text}\n\nசுருக்கம்:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.replace(prompt, "").strip()
