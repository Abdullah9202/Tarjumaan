# 🎙️ Tarjumaan - Urdu Voice Translator

An AI-powered multilingual voice translation system that converts Urdu speech to text and translates it into multiple languages with audio output capabilities.

## 🌟 Features

- **🎤 Voice Recording**: Record Urdu speech directly through your browser
- **📝 Speech-to-Text**: Accurate Urdu transcription using Google Speech Recognition
- **🌐 Multi-Language Translation**: Translate to English, Arabic, Chinese, Spanish, French, German, Turkish, and Persian
- **🔊 Text-to-Speech**: Generate audio output in the target language
- **📱 RTL Support**: Proper right-to-left text display for Urdu, Arabic, and Persian
- **📚 Dataset Integration**: Information and utilities for training custom models

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Internet connection (for Google's free APIs)

### Installation

All dependencies are already installed in this Replit environment:

- streamlit
- SpeechRecognition (Google Speech API)
- deep-translator
- gtts (Google Text-to-Speech)
- librosa
- numpy
- pandas
- scipy

### Running the Application

The application is configured to run automatically. If you need to start it manually:

```bash
streamlit run app.py --server.port 5000
```

## 🔑 API Keys

**No API keys required!** This application uses free Google services for speech recognition, translation, and text-to-speech.

## 📖 How to Use

1. **Select Target Language**: Choose the language you want to translate to (English, Arabic, Chinese, etc.)
2. **Record Audio**: Click "Start Recording" and speak in Urdu
3. **Play Recording**: (Optional) Verify your recording by playing it back
4. **Translate**: Click the "Translate" button to process your speech
5. **Listen to Translation**: Generate and play the translated audio output

## 📊 Datasets for Training

Tarjumaan includes utilities and information for the following datasets:

### Speech Recognition Datasets
- **Mozilla Common Voice Urdu**: Open-source dataset with validated Urdu recordings
- **UrduSpeech**: Academic dataset for accent and dialect variation

### Translation Datasets
- **OPUS-100**: Multilingual parallel corpus (Urdu ↔ 100+ languages)
- **IndicTrans**: Specialized for Indian languages including Urdu
- **Tatoeba**: Community-contributed multilingual sentences

## 🛠️ Dataset Utilities

The `dataset_utils.py` module provides tools for:

- Audio normalization and preprocessing
- Data augmentation (noise, speed, pitch variations)
- Text cleaning for Urdu script
- Parallel corpus creation
- Train/validation/test splits
- Dataset statistics calculation

### Example Usage

```python
from dataset_utils import DatasetUtilities

utils = DatasetUtilities()

# Normalize audio file
normalized_path = utils.normalize_audio('input.wav', 'output.wav')

# Clean Urdu text
clean_text = utils.clean_urdu_text(raw_urdu_text)

# Get dataset information
cv_info = utils.get_common_voice_info()
print(cv_info)
```

## 🏗️ Architecture

```
Tarjumaan/
│
├── app.py                 # Main Streamlit application
├── dataset_utils.py       # Dataset preprocessing utilities
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # Project documentation
```

## 🔧 Technical Stack

- **Frontend**: Streamlit
- **Speech Recognition**: Google Speech Recognition API (free)
- **Translation**: Google Translate (via deep-translator)
- **Text-to-Speech**: Google Text-to-Speech (gTTS)
- **Audio Processing**: SpeechRecognition, librosa

## 🎯 Example Workflow

```
🎙️ User speaks: "میرا نام عبداللہ ہے"
📝 Transcribed: "میرا نام عبداللہ ہے"
🌐 Translated to English: "My name is Abdullah"
🔊 Audio output: [English speech audio]
```

## 🌍 Supported Languages

- English
- Arabic (عربي)
- Chinese Simplified (简体中文)
- Chinese Traditional (繁體中文)
- Spanish (Español)
- French (Français)
- German (Deutsch)
- Turkish (Türkçe)
- Persian (فارسی)

## 📈 Future Enhancements

- Fine-tuned custom models using collected datasets
- Batch translation for multiple audio files
- Translation history and export functionality
- Dialect and accent customization
- Real-time streaming translation
- Support for additional languages

## 🤝 Contributing

This is an educational project demonstrating:
- Machine Learning & NLP
- Speech Recognition
- Multilingual Translation
- Web Application Development

## 📝 License

This project is built for educational purposes and uses the following free services:
- Google Speech Recognition API
- Google Translate
- Google Text-to-Speech

## 🙏 Acknowledgments

- **Google** for providing free speech recognition, translation, and TTS APIs
- **Mozilla Foundation** for the Common Voice dataset
- **OPUS Project** for multilingual parallel corpora
- **Google** for translation and TTS services

---

**Built with ❤️ using Streamlit and Python**

*Empowering multilingual communication through AI*
