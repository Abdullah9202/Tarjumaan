# Tarjumaan - Urdu Voice Translator

## Project Overview
Tarjumaan is an AI-powered multilingual voice translation system that converts Urdu speech to text and translates it into multiple languages (English, Arabic, Chinese, Spanish, French, German, Turkish, and Persian) with optional audio output.

## Recent Changes (November 11, 2025)

### Initial Implementation
- Created Streamlit-based web application with audio recording and file upload capabilities
- Integrated Google Speech Recognition for Urdu speech-to-text (free, no API key required)
- Implemented multi-language translation using deep-translator (Google Translate)
- Added text-to-speech functionality using gTTS for all supported languages
- Built comprehensive dataset utilities module for Common Voice and OPUS datasets
- Implemented proper RTL text support for Urdu, Arabic, and Persian scripts
- Fixed gTTS language code compatibility for Chinese variants (zh-cn, zh-tw)
- Preserved original file extensions for uploaded audio files

## Project Architecture

### Main Components
1. **app.py** - Main Streamlit application
   - Audio input via recording or file upload (WAV, MP3, M4A, OGG, FLAC)
   - Google Speech Recognition integration for Urdu (ur-PK)
   - Multi-language translation engine
   - Text-to-speech generation
   - RTL text display support

2. **dataset_utils.py** - Dataset preprocessing utilities
   - Audio normalization and augmentation
   - Urdu text cleaning and preprocessing
   - Parallel corpus creation
   - Dataset splitting and statistics
   - Information about Common Voice, OPUS, and Tatoeba datasets

3. **.streamlit/config.toml** - Streamlit configuration
   - Server settings (port 5000, headless mode)

### Technical Stack
- **Frontend**: Streamlit
- **Speech Recognition**: Google Speech Recognition API (free)
- **Translation**: Google Translate via deep-translator
- **Text-to-Speech**: gTTS (Google Text-to-Speech)
- **Audio Processing**: SpeechRecognition, librosa, numpy, pandas, scipy

### Key Features
- No API keys required - uses free Google services
- Browser-based audio recording with st.audio_input
- File upload support for multiple audio formats
- Real-time speech-to-text transcription
- Multi-language translation (9 languages)
- Text-to-speech audio generation
- RTL text support for Arabic-script languages
- Dataset integration information

## User Preferences
None documented yet.

## Known Issues
None currently.

## Future Enhancements
- Fine-tuned custom models using collected datasets
- Batch translation for multiple audio files
- Translation history with save/export functionality
- Dialect and accent customization
- Real-time streaming translation
- Support for additional languages
