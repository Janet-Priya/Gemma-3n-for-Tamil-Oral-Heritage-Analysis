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
'''



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
