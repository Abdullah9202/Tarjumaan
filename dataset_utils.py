import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from typing import List, Dict, Tuple
import tempfile
import requests

class DatasetUtilities:
    """
    Utilities for downloading, preprocessing, and managing datasets
    for Urdu speech recognition and translation training.
    """
    
    def __init__(self):
        self.sample_rate = 16000
        
    def normalize_audio(self, audio_path: str, output_path: str = None) -> str:
        """
        Normalize audio file to consistent sample rate and format.
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save normalized audio (optional)
        
        Returns:
            Path to normalized audio file
        """
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            audio = librosa.util.normalize(audio)
            
            if output_path is None:
                output_path = audio_path.replace('.wav', '_normalized.wav')
            
            sf.write(output_path, audio, self.sample_rate)
            
            return output_path
        except Exception as e:
            raise Exception(f"Audio normalization failed: {str(e)}")
    
    def augment_audio(self, audio_path: str, augmentation_type: str = 'noise') -> np.ndarray:
        """
        Apply data augmentation to audio.
        
        Args:
            audio_path: Path to audio file
            augmentation_type: Type of augmentation ('noise', 'speed', 'pitch')
        
        Returns:
            Augmented audio array
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        if augmentation_type == 'noise':
            noise = np.random.randn(len(audio))
            audio = audio + 0.005 * noise
        
        elif augmentation_type == 'speed':
            speed_factor = np.random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        
        elif augmentation_type == 'pitch':
            pitch_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_steps)
        
        return audio
    
    def clean_urdu_text(self, text: str) -> str:
        """
        Clean Urdu text while preserving script characters.
        
        Args:
            text: Raw Urdu text
        
        Returns:
            Cleaned text
        """
        import re
        
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s0-9\.\,\!\?\:\;]', '', text)
        
        return text
    
    def create_parallel_corpus(self, urdu_texts: List[str], translated_texts: List[str], 
                              language_code: str, output_path: str = 'parallel_corpus.csv') -> pd.DataFrame:
        """
        Create a parallel corpus CSV file for translation training.
        
        Args:
            urdu_texts: List of Urdu sentences
            translated_texts: List of translated sentences
            language_code: Target language code (e.g., 'en', 'ar', 'zh')
            output_path: Path to save CSV file
        
        Returns:
            DataFrame with parallel sentences
        """
        if len(urdu_texts) != len(translated_texts):
            raise ValueError("Urdu and translated text lists must have same length")
        
        df = pd.DataFrame({
            'urdu': urdu_texts,
            f'{language_code}': translated_texts,
            'length_ur': [len(text.split()) for text in urdu_texts],
            f'length_{language_code}': [len(text.split()) for text in translated_texts]
        })
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        return df
    
    def split_dataset(self, data: pd.DataFrame, train_ratio: float = 0.8, 
                     val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            data: Input DataFrame
            train_ratio: Proportion for training (default 0.8)
            val_ratio: Proportion for validation (default 0.1)
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_df = shuffled[:train_size]
        val_df = shuffled[train_size:train_size + val_size]
        test_df = shuffled[train_size + val_size:]
        
        return train_df, val_df, test_df
    
    def get_common_voice_info(self) -> Dict[str, str]:
        """
        Get information about Mozilla Common Voice Urdu dataset.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'name': 'Mozilla Common Voice - Urdu',
            'url': 'https://commonvoice.mozilla.org/ur/datasets',
            'description': 'Open-source multilingual dataset with validated Urdu recordings',
            'format': 'MP3 audio files with TSV metadata',
            'size': '~500MB - 2GB (varies by version)',
            'speakers': 'Multiple native speakers',
            'hours': '10-50 hours of validated audio',
            'download_note': 'Requires account creation on Common Voice website',
            'usage': 'Fine-tuning ASR models, speaker recognition, accent analysis'
        }
    
    def get_opus_info(self) -> Dict[str, str]:
        """
        Get information about OPUS parallel corpus.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'name': 'OPUS-100 Multilingual Corpus',
            'url': 'https://opus.nlpl.eu/',
            'description': 'Parallel corpus with Urdu-English, Urdu-Arabic pairs',
            'format': 'Aligned text files (TMX, Moses)',
            'pairs': 'Urdu ↔ English, Arabic, Chinese, and 97+ languages',
            'sentences': '100K+ parallel sentences',
            'download_note': 'Direct download available from OPUS website',
            'usage': 'Training translation models, evaluation, benchmarking'
        }
    
    def get_tatoeba_info(self) -> Dict[str, str]:
        """
        Get information about Tatoeba dataset.
        
        Returns:
            Dictionary with dataset information
        """
        return {
            'name': 'Tatoeba Multilingual Sentences',
            'url': 'https://tatoeba.org/en/downloads',
            'description': 'Community-contributed multilingual sentence database',
            'format': 'TSV files with sentence pairs',
            'urdu_sentences': '100K+ sentences',
            'languages': 'Urdu paired with 300+ languages',
            'download_note': 'Free download, multiple export formats available',
            'usage': 'Testing, evaluation, small-scale training'
        }
    
    def prepare_training_batch(self, audio_files: List[str], transcripts: List[str], 
                              batch_size: int = 32) -> List[Dict]:
        """
        Prepare batches for training from audio files and transcripts.
        
        Args:
            audio_files: List of audio file paths
            transcripts: List of corresponding transcripts
            batch_size: Size of each batch
        
        Returns:
            List of batches, each containing audio arrays and transcripts
        """
        if len(audio_files) != len(transcripts):
            raise ValueError("Audio files and transcripts must have same length")
        
        batches = []
        
        for i in range(0, len(audio_files), batch_size):
            batch_audio_files = audio_files[i:i + batch_size]
            batch_transcripts = transcripts[i:i + batch_size]
            
            batch_audio_data = []
            for audio_file in batch_audio_files:
                audio, _ = librosa.load(audio_file, sr=self.sample_rate)
                batch_audio_data.append(audio)
            
            batches.append({
                'audio': batch_audio_data,
                'transcripts': batch_transcripts,
                'size': len(batch_audio_files)
            })
        
        return batches
    
    def calculate_dataset_statistics(self, audio_files: List[str], transcripts: List[str]) -> Dict:
        """
        Calculate statistics for a dataset.
        
        Args:
            audio_files: List of audio file paths
            transcripts: List of transcripts
        
        Returns:
            Dictionary with dataset statistics
        """
        durations = []
        word_counts = []
        
        for audio_file, transcript in zip(audio_files, transcripts):
            try:
                audio, sr = librosa.load(audio_file, sr=None)
                duration = len(audio) / sr
                durations.append(duration)
                word_counts.append(len(transcript.split()))
            except:
                continue
        
        return {
            'total_samples': len(audio_files),
            'total_duration_hours': sum(durations) / 3600,
            'avg_duration_seconds': np.mean(durations) if durations else 0,
            'min_duration_seconds': np.min(durations) if durations else 0,
            'max_duration_seconds': np.max(durations) if durations else 0,
            'avg_words_per_sample': np.mean(word_counts) if word_counts else 0,
            'total_words': sum(word_counts)
        }


if __name__ == "__main__":
    utils = DatasetUtilities()
    
    print("=" * 60)
    print("Tarjumaan Dataset Utilities - Information")
    print("=" * 60)
    
    print("\n📊 Common Voice Urdu Dataset:")
    cv_info = utils.get_common_voice_info()
    for key, value in cv_info.items():
        print(f"  {key}: {value}")
    
    print("\n📊 OPUS Multilingual Corpus:")
    opus_info = utils.get_opus_info()
    for key, value in opus_info.items():
        print(f"  {key}: {value}")
    
    print("\n📊 Tatoeba Dataset:")
    tatoeba_info = utils.get_tatoeba_info()
    for key, value in tatoeba_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Use these datasets to improve Tarjumaan's accuracy!")
    print("=" * 60)
