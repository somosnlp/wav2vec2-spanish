---
language: es
tags:
- audio
- automatic-speech-recognition
datasets:
- common_voice
---

# Wav2Vec2 Spanish

Spanish Wav2Vec2 model pre-trained using the Spanish portion of the Common Voice dataset.

Part of the [Flax x Hugging Face](https://discss.huggingface.co/t/open-to-the-community-community-week-using-jax-flax-for-nlp-cv/7104) community event.

Team:
[@mariagrandury](https://github.com/mariagrandury),
[@mrm8488](https://github.com/mrm8488),
[@edugp](https://github.com/edugp) and
[@pcuenq](https://github.com/pcuenq).

## Model description

The model used for training is [Wav2Vec2] by FacebookAI. It was introduced in the paper 
"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, and Michael Auli (https://arxiv.org/abs/2006.11477).

This model is available in the 🤗 [Model Hub](https://huggingface.co/facebook/wav2vec2-base-960h).

## Intended uses & limitations

### How to use (TODO)

### Limitations and bias (TODO)

## Training data

Spanish portion of [Common Voice](https://commonvoice.mozilla.org/en/datasets). Common Voice is an open source, multi-language dataset of voices part of Mozilla's initiative to help teach machines how real people speak.

The dataset is also available in the 🤗 [Datasets](https://huggingface.co/datasets/common_voice) library.

### Training procedure (TODO: update)

The script used for training (`train.sh`) is based on [this training script](https://github.com/huggingface/transformers/blob/master/examples/research_projects/jax-projects/wav2vec2/run_wav2vec2_pretrain_flax.py) and was modified as explained in `training_script_evolution.md`.

### Eval results (TODO)
