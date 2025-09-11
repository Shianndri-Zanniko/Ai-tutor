import base64
import mimetypes
import os
import re
import struct
from google import genai
from google.genai import types
from typing import Optional, Tuple
import dotenv
import tempfile

# Load environment variables
dotenv.load_dotenv()

class GeminiTTS:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini TTS client
        
        Args:
            api_key: Gemini API key (if not provided, loads from GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or provided")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash-preview-tts"
        
    def text_to_speech(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Convert text to speech using Gemini TTS
        
        Args:
            text: Text to convert to speech
            output_path: Path to save audio file (if None, uses temp file)
            
        Returns:
            Path to generated audio file or None if error
        """
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=text),
                    ],
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature=1,
                response_modalities=[
                    "audio",
                ],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Zephyr"
                        )
                    )
                ),
            )
            
            # Generate audio chunks
            audio_chunks = []
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                    
                part = chunk.candidates[0].content.parts[0]
                if part.inline_data and part.inline_data.data:
                    audio_chunks.append(part.inline_data)
            
            if not audio_chunks:
                print("No audio data received")
                return None
            
            # Combine all audio chunks
            combined_audio_data = b""
            mime_type = None
            
            for chunk_data in audio_chunks:
                combined_audio_data += chunk_data.data
                if mime_type is None:
                    mime_type = chunk_data.mime_type
            
            # Convert to WAV if needed
            if mime_type and "wav" not in mime_type.lower():
                combined_audio_data = self._convert_to_wav(combined_audio_data, mime_type)
                file_extension = ".wav"
            else:
                file_extension = mimetypes.guess_extension(mime_type) or ".wav"
            
            # Save to file
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=file_extension, 
                    delete=False
                )
                output_path = temp_file.name
                temp_file.close()
            
            with open(output_path, "wb") as f:
                f.write(combined_audio_data)
            
            print(f"Audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def _convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """
        Converts audio data to WAV format
        
        Args:
            audio_data: The raw audio data as bytes
            mime_type: MIME type of the audio data
            
        Returns:
            WAV formatted audio data
        """
        try:
            parameters = self._parse_audio_mime_type(mime_type)
            bits_per_sample = parameters["bits_per_sample"]
            sample_rate = parameters["rate"]
            num_channels = 1
            data_size = len(audio_data)
            bytes_per_sample = bits_per_sample // 8
            block_align = num_channels * bytes_per_sample
            byte_rate = sample_rate * block_align
            chunk_size = 36 + data_size
            
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF",          # ChunkID
                chunk_size,       # ChunkSize (total file size - 8 bytes)
                b"WAVE",          # Format
                b"fmt ",          # Subchunk1ID
                16,               # Subchunk1Size (16 for PCM)
                1,                # AudioFormat (1 for PCM)
                num_channels,     # NumChannels
                sample_rate,      # SampleRate
                byte_rate,        # ByteRate
                block_align,      # BlockAlign
                bits_per_sample,  # BitsPerSample
                b"data",          # Subchunk2ID
                data_size         # Subchunk2Size (size of audio data)
            )
            return header + audio_data
            
        except Exception as e:
            print(f"Error converting to WAV: {e}")
            return audio_data
    
    def _parse_audio_mime_type(self, mime_type: str) -> dict:
        """
        Parses bits per sample and rate from an audio MIME type string
        
        Args:
            mime_type: The audio MIME type string
            
        Returns:
            Dictionary with "bits_per_sample" and "rate" keys
        """
        bits_per_sample = 16
        rate = 24000
        
        # Extract rate from parameters
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip()
            if param.lower().startswith("rate="):
                try:
                    rate_str = param.split("=", 1)[1]
                    rate = int(rate_str)
                except (ValueError, IndexError):
                    pass
            elif param.startswith("audio/L"):
                try:
                    bits_per_sample = int(param.split("L", 1)[1])
                except (ValueError, IndexError):
                    pass
        
        return {"bits_per_sample": bits_per_sample, "rate": rate}
    
    def get_audio_bytes(self, text: str) -> Optional[bytes]:
        """
        Convert text to speech and return audio bytes
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes or None if error
        """
        try:
            audio_file_path = self.text_to_speech(text)
            if audio_file_path:
                with open(audio_file_path, "rb") as f:
                    audio_bytes = f.read()
                # Clean up temp file
                os.unlink(audio_file_path)
                return audio_bytes
            return None
        except Exception as e:
            print(f"Error getting audio bytes: {e}")
            return None