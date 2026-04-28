import streamlit as st
import os
import numpy as np
from gtts import gTTS
from deep_translator import GoogleTranslator
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
import psycopg2
from datetime import datetime

st.set_page_config(
    page_title="Tarjumaan - Urdu Voice Translator",
    page_icon="🎙️",
    layout="wide"
)

st.markdown("""
    <style>
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-size: 24px;
        font-weight: bold;
        padding: 20px;
        background-color: #f0f2f6;
        color: #1a1a1a;
        border-radius: 10px;
        margin: 10px 0;
    }
    .arabic-text {
        direction: rtl;
        text-align: right;
        font-size: 20px;
        padding: 15px;
        background-color: #e8f4f8;
        color: #1a1a1a;
        border-radius: 10px;
        margin: 10px 0;
    }
    .translated-text {
        font-size: 20px;
        padding: 15px;
        background-color: #e8f4f8;
        color: #1a1a1a;
        border-radius: 10px;
        margin: 10px 0;
    }
    .recording-indicator {
        color: red;
        font-weight: bold;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

if 'urdu_text' not in st.session_state:
    st.session_state.urdu_text = ""
if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ""
if 'target_language' not in st.session_state:
    st.session_state.target_language = "English"
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

def save_translation_to_db(urdu_text, translated_text, target_language, filename=None, translation_type='single'):
    try:
        conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO translation_history (urdu_text, translated_text, target_language, filename, translation_type) VALUES (%s, %s, %s, %s, %s)",
            (urdu_text, translated_text, target_language, filename, translation_type)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        st.warning(f"Could not save to history: {str(e)}")

def get_translation_history(limit=50):
    try:
        conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
        cur = conn.cursor()
        cur.execute(
            "SELECT id, created_at, filename, urdu_text, translated_text, target_language, translation_type FROM translation_history ORDER BY created_at DESC LIMIT %s",
            (limit,)
        )
        results = cur.fetchall()
        cur.close()
        conn.close()
        return results
    except Exception as e:
        st.error(f"Could not retrieve history: {str(e)}")
        return []

st.title("🎙️ Tarjumaan - Urdu Voice Translator")
st.markdown("### AI-Powered Multilingual Translation System")

tab1, tab2, tab3 = st.tabs(["🎤 Single Translation", "📁 Batch Translation", "📜 History"])

with tab1:
    st.markdown("---")

col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("⚙️ Settings")
    
    target_language = st.selectbox(
        "Select Target Language",
        ["English", "Arabic", "Chinese (Simplified)", "Chinese (Traditional)", "Spanish", "French", "German", "Turkish", "Persian"],
        key="lang_selector"
    )
    
    language_codes = {
        "English": "en",
        "Arabic": "ar",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Turkish": "tr",
        "Persian": "fa"
    }
    
    tts_language_codes = {
        "English": "en",
        "Arabic": "ar",
        "Chinese (Simplified)": "zh-cn",
        "Chinese (Traditional)": "zh-tw",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Turkish": "tr",
        "Persian": "fa"
    }
    
    st.markdown("---")
    st.info("💡 **Tip**: Speak clearly in Urdu for best results")
    
    st.markdown("### 📌 Supported Audio Formats")
    st.caption("WAV, MP3, M4A, OGG, FLAC")

with col1:
    st.subheader("🎤 Audio Input")
    
    st.markdown("**Option 1: Record Audio**")
    audio_bytes = st.audio_input("Click to record your Urdu speech")
    
    if audio_bytes:
        st.session_state.audio_bytes = audio_bytes
        st.success("✅ Audio recorded successfully!")
        st.audio(audio_bytes)
    
    st.markdown("---")
    st.markdown("**Option 2: Upload Audio File**")
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'm4a', 'ogg', 'flac'])
    
    if uploaded_file:
        st.session_state.audio_bytes = uploaded_file
        st.success("✅ Audio file uploaded successfully!")
        st.audio(uploaded_file)
    
    if st.button("🔄 Clear All", use_container_width=True):
        st.session_state.audio_bytes = None
        st.session_state.urdu_text = ""
        st.session_state.translated_text = ""
        if 'tts_audio' in st.session_state:
            del st.session_state.tts_audio
        st.rerun()

st.markdown("---")

if st.button("🚀 Translate", use_container_width=True, type="primary"):
    if st.session_state.audio_bytes is None:
        st.error("❌ Please record or upload audio first!")
    else:
        try:
            with st.spinner("🎯 Transcribing Urdu speech..."):
                audio_data = st.session_state.audio_bytes
                
                if hasattr(audio_data, 'read'):
                    audio_data.seek(0)
                    audio_content = audio_data.read()
                    file_extension = getattr(audio_data, 'name', 'audio.wav').split('.')[-1].lower()
                else:
                    audio_content = audio_data
                    file_extension = 'wav'
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_input:
                    temp_input.write(audio_content)
                    temp_input.flush()
                    input_path = temp_input.name
                
                wav_path = None
                try:
                    if file_extension in ['wav', 'flac']:
                        wav_path = input_path
                    else:
                        with st.spinner(f"🔄 Converting {file_extension.upper()} to WAV format..."):
                            audio_segment = AudioSegment.from_file(input_path, format=file_extension)
                            wav_path = input_path.replace(f'.{file_extension}', '.wav')
                            audio_segment.export(wav_path, format='wav')
                            if os.path.exists(input_path):
                                os.unlink(input_path)
                    
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(wav_path) as source:
                        audio = recognizer.record(source)
                        text = recognizer.recognize_google(audio, language="ur-PK")
                    
                    st.session_state.urdu_text = text
                    st.success("✅ Transcription completed!")
                finally:
                    if wav_path and os.path.exists(wav_path):
                        os.unlink(wav_path)
                    if input_path != wav_path and os.path.exists(input_path):
                        os.unlink(input_path)
            
            if st.session_state.urdu_text:
                with st.spinner(f"🌐 Translating to {target_language}..."):
                    translator = GoogleTranslator(source='ur', target=language_codes[target_language])
                    st.session_state.translated_text = translator.translate(st.session_state.urdu_text)
                    
                    save_translation_to_db(
                        st.session_state.urdu_text,
                        st.session_state.translated_text,
                        target_language,
                        filename=None,
                        translation_type='single'
                    )
                    
                    st.success("✅ Translation completed!")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if st.session_state.urdu_text or st.session_state.translated_text:
    st.markdown("---")
    st.subheader("📝 Results")
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        st.markdown("**Original Urdu Text:**")
        if st.session_state.urdu_text:
            st.markdown(f'<div class="urdu-text">{st.session_state.urdu_text}</div>', unsafe_allow_html=True)
    
    with col_result2:
        st.markdown(f"**Translated to {target_language}:**")
        if st.session_state.translated_text:
            if target_language in ["Arabic", "Persian"]:
                st.markdown(f'<div class="arabic-text">{st.session_state.translated_text}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="translated-text">{st.session_state.translated_text}</div>', unsafe_allow_html=True)
    
    if st.session_state.translated_text:
        st.markdown("---")
        st.subheader("🔊 Text-to-Speech")
        
        col_tts1, col_tts2 = st.columns([1, 3])
        
        with col_tts1:
            if st.button("🎵 Generate & Play Audio", use_container_width=True):
                try:
                    with st.spinner("🔊 Generating speech..."):
                        tts = gTTS(text=st.session_state.translated_text, lang=tts_language_codes[target_language], slow=False)
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_tts:
                            tts.save(temp_tts.name)
                            audio_file = open(temp_tts.name, 'rb')
                            audio_bytes = audio_file.read()
                            audio_file.close()
                            os.unlink(temp_tts.name)
                        
                        st.session_state.tts_audio = audio_bytes
                        st.success("✅ Audio generated!")
                except Exception as e:
                    st.error(f"❌ TTS error: {str(e)}")
        
        if 'tts_audio' in st.session_state and st.session_state.tts_audio:
            with col_tts2:
                st.audio(st.session_state.tts_audio, format='audio/mp3')
                
                st.download_button(
                    label="⬇️ Download Audio",
                    data=st.session_state.tts_audio,
                    file_name=f"tarjumaan_translation_{target_language.lower().replace(' ', '_')}.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )

    st.markdown("---")

    with st.expander("📚 Dataset Information & Training Resources"):
        st.markdown("""
    ### 🗃️ Recommended Datasets for Model Training
    
    #### Speech Recognition (Urdu ASR)
    - **Mozilla Common Voice Urdu**: Open-source dataset with thousands of validated Urdu voice recordings
        - Access: [commonvoice.mozilla.org](https://commonvoice.mozilla.org/)
        - Contains: Native speaker recordings with transcriptions
        - Use case: Training custom ASR models or improving recognition accuracy
    
    - **UrduSpeech**: Academic dataset for Urdu speech recognition
        - Contains: Urdu sentences spoken by multiple speakers
        - Use case: Accent and dialect variation training
    
    #### Translation Datasets
    - **OPUS-100**: Multilingual parallel corpus including Urdu pairs
        - Languages: Urdu ↔ English, Arabic, Chinese, and 97+ languages
        - Access: [opus.nlpl.eu](https://opus.nlpl.eu/)
        - Use case: Training/fine-tuning translation models
    
    - **IndicTrans Dataset**: Specialized for Indian languages including Urdu
        - Contains: High-quality parallel sentences
        - Use case: Urdu-English translation improvements
    
    - **Tatoeba**: Community-contributed multilingual sentences
        - Contains: 100K+ Urdu sentences with translations
        - Use case: Evaluation and testing
    
    #### 🔧 Dataset Preprocessing Tips
    1. **Audio Normalization**: Use librosa to normalize audio files to consistent sample rates
    2. **Text Cleaning**: Remove special characters while preserving Urdu script
    3. **Data Augmentation**: Add background noise and speed variations
    4. **Train/Val/Test Split**: Use 80/10/10 ratio for balanced evaluation
    
    #### 💻 Quick Start with Datasets
    ```python
    # Example: Loading Common Voice dataset (requires datasets library)
    # from datasets import load_dataset
    # cv_dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ur")
    
    # Example: Processing OPUS parallel corpus
    # import pandas as pd
    # opus_data = pd.read_csv("opus_urdu_english.tsv", sep="\t")
        ```
        
        **Note**: For production use with these datasets, you may need to fine-tune the models using dedicated ML frameworks like PyTorch or TensorFlow.
        """)

with tab2:
    st.markdown("---")
    st.subheader("📁 Batch Translation")
    st.markdown("Upload multiple audio files and process them all at once")
    
    batch_target_lang = st.selectbox(
        "Select Target Language for Batch",
        ["English", "Arabic", "Chinese (Simplified)", "Chinese (Traditional)", "Spanish", "French", "German", "Turkish", "Persian"],
        key="batch_lang_selector"
    )
    
    uploaded_files = st.file_uploader(
        "Upload audio files (multiple files supported)",
        type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    col_batch1, col_batch2, col_batch3 = st.columns(3)
    
    with col_batch1:
        if st.button("🚀 Process All Files", use_container_width=True, type="primary", disabled=not uploaded_files):
            st.session_state.batch_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                
                try:
                    uploaded_file.seek(0)
                    audio_content = uploaded_file.read()
                    file_extension = uploaded_file.name.split('.')[-1].lower()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as temp_input:
                        temp_input.write(audio_content)
                        temp_input.flush()
                        input_path = temp_input.name
                    
                    wav_path = None
                    try:
                        if file_extension in ['wav', 'flac']:
                            wav_path = input_path
                        else:
                            audio_segment = AudioSegment.from_file(input_path, format=file_extension)
                            wav_path = input_path.replace(f'.{file_extension}', '.wav')
                            audio_segment.export(wav_path, format='wav')
                            if os.path.exists(input_path):
                                os.unlink(input_path)
                        
                        recognizer = sr.Recognizer()
                        with sr.AudioFile(wav_path) as source:
                            audio = recognizer.record(source)
                            urdu_text = recognizer.recognize_google(audio, language="ur-PK")
                        
                        translator = GoogleTranslator(source='ur', target=language_codes[batch_target_lang])
                        translated_text = translator.translate(urdu_text)
                        
                        save_translation_to_db(
                            urdu_text,
                            translated_text,
                            batch_target_lang,
                            filename=uploaded_file.name,
                            translation_type='batch'
                        )
                        
                        st.session_state.batch_results.append({
                            'filename': uploaded_file.name,
                            'status': 'Success',
                            'urdu_text': urdu_text,
                            'translated_text': translated_text,
                            'target_language': batch_target_lang
                        })
                    
                    except Exception as e:
                        st.session_state.batch_results.append({
                            'filename': uploaded_file.name,
                            'status': 'Failed',
                            'urdu_text': '',
                            'translated_text': '',
                            'error': str(e),
                            'target_language': batch_target_lang
                        })
                    
                    finally:
                        if wav_path and os.path.exists(wav_path):
                            os.unlink(wav_path)
                        if input_path != wav_path and os.path.exists(input_path):
                            os.unlink(input_path)
                
                except Exception as e:
                    st.session_state.batch_results.append({
                        'filename': uploaded_file.name,
                        'status': 'Failed',
                        'urdu_text': '',
                        'translated_text': '',
                        'error': str(e),
                        'target_language': batch_target_lang
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("✅ Batch processing complete!")
            st.success(f"Processed {len(uploaded_files)} files")
    
    with col_batch2:
        if st.button("🔄 Clear Results", use_container_width=True):
            st.session_state.batch_results = []
            st.rerun()
    
    with col_batch3:
        if st.session_state.batch_results:
            import pandas as pd
            import io
            
            df = pd.DataFrame(st.session_state.batch_results)
            csv = df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Export CSV",
                data=csv,
                file_name="tarjumaan_batch_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    if st.session_state.batch_results:
        st.markdown("---")
        st.subheader("📊 Batch Results")
        
        for idx, result in enumerate(st.session_state.batch_results):
            with st.expander(f"{'✅' if result['status'] == 'Success' else '❌'} {result['filename']}", expanded=idx==0):
                if result['status'] == 'Success':
                    col_r1, col_r2 = st.columns(2)
                    
                    with col_r1:
                        st.markdown("**Original Urdu:**")
                        st.markdown(f'<div class="urdu-text">{result["urdu_text"]}</div>', unsafe_allow_html=True)
                    
                    with col_r2:
                        st.markdown(f"**Translated to {result['target_language']}:**")
                        if result['target_language'] in ["Arabic", "Persian"]:
                            st.markdown(f'<div class="arabic-text">{result["translated_text"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="translated-text">{result["translated_text"]}</div>', unsafe_allow_html=True)
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")

with tab3:
    st.markdown("---")
    st.subheader("📜 Translation History")
    st.markdown("View your past translations and export them for reference")
    
    col_hist1, col_hist2, col_hist3 = st.columns([2, 1, 1])
    
    with col_hist1:
        limit = st.slider("Number of records to display", 10, 200, 50, step=10)
    
    with col_hist2:
        filter_type = st.selectbox("Filter by type", ["All", "Single", "Batch"])
    
    with col_hist3:
        if st.button("🔄 Refresh History", use_container_width=True):
            st.rerun()
    
    history = get_translation_history(limit=limit)
    
    if filter_type != "All":
        history = [h for h in history if h[6].lower() == filter_type.lower()]
    
    if history:
        st.markdown(f"**Showing {len(history)} translations**")
        
        import pandas as pd
        
        df_data = []
        for record in history:
            df_data.append({
                'Date': record[1].strftime('%Y-%m-%d %H:%M'),
                'Type': record[6].capitalize(),
                'Filename': record[2] or 'Recording',
                'Urdu Text': record[3][:50] + '...' if len(record[3]) > 50 else record[3],
                'Translation': record[4][:50] + '...' if len(record[4]) > 50 else record[4],
                'Target Lang': record[5]
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📥 Export History")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            full_df_data = []
            for record in history:
                full_df_data.append({
                    'id': record[0],
                    'created_at': record[1].strftime('%Y-%m-%d %H:%M:%S'),
                    'filename': record[2] or 'Recording',
                    'urdu_text': record[3],
                    'translated_text': record[4],
                    'target_language': record[5],
                    'translation_type': record[6]
                })
            
            full_df = pd.DataFrame(full_df_data)
            csv = full_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"tarjumaan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col_export2:
            import json
            json_data = json.dumps(full_df_data, ensure_ascii=False, indent=2)
            
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"tarjumaan_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        st.markdown("---")
        
        with st.expander("🔍 View Full Record Details"):
            selected_idx = st.selectbox("Select a record to view full details", range(len(history)), format_func=lambda i: f"{history[i][1].strftime('%Y-%m-%d %H:%M')} - {history[i][2] or 'Recording'}")
            
            if selected_idx is not None:
                record = history[selected_idx]
                st.markdown(f"**Date:** {record[1].strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**Type:** {record[6].capitalize()}")
                st.markdown(f"**Filename:** {record[2] or 'Recording'}")
                st.markdown(f"**Target Language:** {record[5]}")
                
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.markdown("**Full Urdu Text:**")
                    st.markdown(f'<div class="urdu-text">{record[3]}</div>', unsafe_allow_html=True)
                
                with col_detail2:
                    st.markdown(f"**Full Translation ({record[5]}):**")
                    if record[5] in ["Arabic", "Persian"]:
                        st.markdown(f'<div class="arabic-text">{record[4]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="translated-text">{record[4]}</div>', unsafe_allow_html=True)
    else:
        st.info("No translation history yet. Start translating to build your history!")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Tarjumaan</strong> - Empowering multilingual communication through AI</p>
    <p>Built with Streamlit | Google Speech Recognition | Google Translate | gTTS</p>
</div>
""", unsafe_allow_html=True)
