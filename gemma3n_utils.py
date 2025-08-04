import os
import sys
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForImageTextToText

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
    device_map="auto",
    token=HF_TOKEN
)
# DO NOT CALL model.to(device) here!

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
