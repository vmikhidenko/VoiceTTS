run a file:

python train.py --config_path configs/vits_custom_voice_config.json
separate terminal:
tensorboard --logdir=models/vits_custom_voice/logs

http://localhost:6006/

run main.py:
python3 main.py --config_path configs/default_config.json --model_path models/vits_custom_voice/model_epoch_235.pth \
                --output_dir output --verbosity 1 --output_filename audio.wav



# Custom Text-to-Speech (TTS) Generator

This project implements a **Custom Text-to-Speech (TTS) Generator** that allows users to create personalized voice models using their own voice recordings. Leveraging advanced machine learning techniques and the [VITS](https://github.com/jaywalnut310/vits) architecture, the model can convert input text into speech that mimics the provided voice samples.

## 🚀 Features

- **Custom Voice Training**: Train the model with your own voice recordings for personalized speech synthesis.
- **High-Quality Audio Output**: Generate natural-sounding speech from text input.
- **Flexible Configuration**: Easily adjust model and training parameters via `config.json`.
- **Command-Line Interface (CLI)**: User-friendly CLI for training and generating speech.
- **Logging and Error Handling**: Comprehensive logging for monitoring and debugging.
- **Configuration Validation**: Ensures `config.json` adheres to the defined schema using `jsonschema`.

## 🛠 Technologies Used

- **Programming Language**: Python 3.8+
- **Machine Learning Framework**: [PyTorch](https://pytorch.org/)
- **TTS Library**: [TTS](https://github.com/coqui-ai/TTS) (`TTS==0.14.3`)
- **Additional Libraries**:
  - `jsonschema`: For configuration validation
  - `numpy`, `pandas`: For data manipulation
  - `soundfile`, `librosa`: For audio processing
  - `argparse`, `logging`: For CLI and logging

## 📁 Project Structure

custom_tts_project/
├── data/
│   ├── metadata.csv
│   └── wavs/
│       ├── sample1.wav
│       ├── sample2.wav
│       └── ...
├── models/
│   └── vits_custom_voice/
│       └── config.json
├── schemas/
│   └── config_schema.json
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── voice_trainer.py
│   ├── tts_generator.py
│   ├── tts_dataset.py
│   ├── audio_processor.py
│   ├── characters_config.py
│   ├── tokenizer.py
│   ├── speaker_manager.py
│   └── model.py
├── requirements.txt
├── train.py
└── main.py




## 📦 Setup

### 1. Prerequisites

- **Python**: Version 3.8 or higher.
- **pip**: Ensure you have pip installed.

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/custom-tts-generator.git
cd custom-tts-generator
