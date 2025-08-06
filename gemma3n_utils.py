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
'''

import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForImageTextToText


# Get HF token from environment variable (ensure you set this before running)
HF_TOKEN = os.environ.get("HF_TOKEN", None)
GEMMA_MODEL_ID = "google/gemma-3n-E4B-it"

# Load model and processor once on import
processor = AutoProcessor.from_pretrained(GEMMA_MODEL_ID, token=HF_TOKEN)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=HF_TOKEN,
)
model.eval()  # set model to eval mode


def transcribe_audio(audio_path):
    """
    Transcribes Tamil speech audio into text using Gemma 3n.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        str: Transcribed text or error message.
    """
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError(f"Audio file does not exist or path is invalid: {audio_path}")

    try:
        waveform, sample_rate = sf.read(audio_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    # Convert stereo to mono if needed by averaging channels
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError(f"Unsupported audio shape: {waveform.shape}")

    # Convert to float32 numpy array with shape (1, num_samples)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[None, :]  # Add batch dimension
    elif waveform.ndim != 2 or waveform.shape[0] != 1:
        # If unexpected shape, convert to mono and reshape explicitly
        waveform = waveform.mean(axis=0, keepdims=True)

    # Resample audio to 16 kHz if needed
    if sample_rate != 16000:
        waveform_torch = torch.from_numpy(waveform)
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform_torch = resampler(waveform_torch)
        waveform = waveform_torch.numpy()

    # Pad or truncate to max 30 seconds (480,000 samples at 16 kHz)
    max_samples = 30 * 16000
    num_samples = waveform.shape[1]
    if num_samples < max_samples:
        pad_width = max_samples - num_samples
        waveform = np.pad(waveform, ((0, 0), (0, pad_width)), mode="constant")
    else:
        waveform = waveform[:, :max_samples]

    # Final checks before passing to processor
    assert waveform.dtype == np.float32, f"Expected float32 dtype, got {waveform.dtype}"
    assert waveform.ndim == 2 and waveform.shape[0] == 1, f"Waveform shape must be (1, samples), got {waveform.shape}"

    # Prepare chat messages with audio and transcription instruction
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": waveform},
                {"type": "text", "text": "Transcribe this audio in Tamil."},
            ],
        }
    ]

    input_ids = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Generate transcription tokens
    outputs = model.generate(**input_ids, max_new_tokens=128)

    transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return transcription


def summarize_text(text):
    """
    Summarizes Tamil text using Gemma 3n.

    Args:
        text (str): Input Tamil text to summarize.

    Returns:
        str: Summary text.
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise ValueError("Input text to summarize must be a non-empty string.")

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

