import os
import torch
import warnings
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import numpy as np
from typing import Optional
import tempfile

class WhisperASR:
    def __init__(self, model_name: str = "conevonce/whisper-small-id3"):
        """
        Initialize Whisper ASR with Indonesian fine-tuned model using direct transformers
        
        Args:
            model_name: HuggingFace model name for Indonesian Whisper
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the Whisper model using WhisperProcessor and WhisperForConditionalGeneration"""
        try:
            print(f"Loading Whisper model: {self.model_name}")
            
            # Suppress deprecation warnings
            warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*return_token_timestamps.*deprecated.*")
            warnings.filterwarnings("ignore", message=".*return_attention_mask.*")
            
            # Try loading the specified model
            try:
                self.processor = WhisperProcessor.from_pretrained(self.model_name)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
                self.model.config.forced_decoder_ids = None
                print(f"Successfully loaded {self.model_name}")
            except Exception as e:
                print(f"Failed to load {self.model_name}: {e}")
                print("Falling back to base whisper-small model...")
                
                # Fallback to base model
                self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
                self.model.config.forced_decoder_ids = None
                print("Successfully loaded openai/whisper-small as fallback")
            
            # Move model to device
            self.model = self.model.to(self.device)
            print(f"Model ready on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Transcribe audio file to text
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text or None if error
        """
        if self.processor is None or self.model is None:
            self.load_model()
            
        try:
            # Load and preprocess audio
            audio_input, sample_rate = sf.read(audio_file_path)
            
            # Ensure mono audio
            if len(audio_input.shape) > 1:
                audio_input = np.mean(audio_input, axis=1)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                try:
                    import librosa
                    audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
                except ImportError:
                    print("Warning: librosa not available, audio may not be resampled properly")
            
            # Process audio to input features
            input_features = self.processor(
                audio_input, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features
            
            # Move to device
            input_features = input_features.to(self.device)
            
            # Generate token ids
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            # Decode token ids to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            return transcription[0].strip() if transcription else None
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
    
    def transcribe_audio_bytes(self, audio_bytes: bytes) -> Optional[str]:
        """
        Transcribe audio from bytes
        
        Args:
            audio_bytes: Audio data as bytes
            
        Returns:
            Transcribed text or None if error
        """
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Transcribe the temporary file
            result = self.transcribe_audio(temp_file_path)
            
            # Clean up temporary file
            os.unlink(temp_file_path)
            
            return result
            
        except Exception as e:
            print(f"Error transcribing audio bytes: {e}")
            return None