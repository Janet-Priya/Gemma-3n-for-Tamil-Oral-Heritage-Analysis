# Unsloth-Finetuned Gemma 3n for Tamil Oral Heritage: Transcription, Translation, and Thematic Analysis
This project is a foundational prototype that demonstrates the use of artificial intelligence for preserving and analyzing Tamil oral literature. It enables users to upload Tamil audio recordings, automatically transcribe them, translate the transcription into English, and generate a deeper literary or contextual analysis of the content.

The Kaggle notebook comprises of the project and the main finetunes model using UnSloth and it is built using Gemma 3n
https://www.kaggle.com/code/janetanand/gemma-3n-for-tamil-oral-heritage-analysis

---

##  Project Objective

The goal of this project is to create a base system that can support the preservation of Tamil oral heritage by:
- Transcribing spoken Tamil into written text
- Translating Tamil content into English
- Offering deeper insights and meaning through contextual analysis and sometimes the hidden meaning behind it

This tool can serve as a digital assistant for linguists, researchers, and cultural preservationists working with oral literature.


## Features

- ** Tamil Audio Transcription**: Converts spoken Tamil audio into text using deep learning.
- ** English Translation**: Automatically translates the transcription into English.
- ** Deeper Analysis**: Generates an analytical summary to explore the meaning or message behind the content.
- **Supports**: Most of the Tamil dialects and provides indepth analysis of the Tamil literature

---

## Model Information

The project uses the `google/gemma-1.1-2b-it` language model for all tasks:
- Implemented via Hugging Face Transformers and Gradio
- Model handles multilingual tasks including instruction-following, summarization, and translation

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Janet-Priya/Tamil-Literature-summarize.git
cd Tamil-Literature-summarize
```
## Sample Use Case
Recording an elder reciting a Tamil poem or story

Uploading the audio to this tool

Receiving a transcription, English translation, and a concise analytical summary

Helping researchers understand oral literature without requiring deep knowledge of Tamil



