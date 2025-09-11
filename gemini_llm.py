import os
from google import genai
from google.genai import types
from typing import Optional, Generator
import dotenv

# Load environment variables
dotenv.load_dotenv()

class GeminiLLM:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini LLM client
        
        Args:
            api_key: Gemini API key (if not provided, loads from GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or provided")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-flash-lite"
        
    def generate_tutor_response(self, student_question: str) -> Optional[str]:
        """
        Generate educational response for Indonesian elementary school students
        
        Args:
            student_question: Student's question in Indonesian
            
        Returns:
            Tutor's response in Indonesian or None if error
        """
        try:
            # Create system prompt for Indonesian elementary school tutor
            system_prompt = """Kamu adalah tutor AI untuk siswa sekolah dasar Indonesia. 
            Jawab pertanyaan dengan:
            - Bahasa Indonesia yang mudah dipahami anak SD
            - Penjelasan yang sederhana dan jelas
            - Gunakan contoh-contoh yang familiar untuk anak Indonesia
            - Bersikap ramah, sabar, dan mendorong
            - Berikan penjelasan step-by-step jika diperlukan
            - Gunakan emoji yang sesuai untuk membuat jawaban lebih menarik
            """
            
            full_prompt = f"{system_prompt}\n\nPertanyaan siswa: {student_question}"
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=full_prompt),
                    ],
                ),
            ]
            
            tools = [
                types.Tool(googleSearch=types.GoogleSearch()),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
                tools=tools,
                temperature=0.7,  # Slightly creative but consistent
                max_output_tokens=1000,  # Reasonable length for elementary students
            )
            
            # Generate response
            response_text = ""
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    response_text += chunk.text
            
            return response_text.strip() if response_text else None
            
        except Exception as e:
            print(f"Error generating tutor response: {e}")
            return None
    
    def generate_response_stream(self, student_question: str) -> Generator[str, None, None]:
        """
        Generate streaming educational response for real-time display
        
        Args:
            student_question: Student's question in Indonesian
            
        Yields:
            Response chunks as they are generated
        """
        try:
            system_prompt = """Kamu adalah tutor AI untuk siswa sekolah dasar Indonesia. 
            Jawab pertanyaan dengan:
            - Bahasa Indonesia yang mudah dipahami anak SD
            - Penjelasan yang sederhana dan jelas
            - Gunakan contoh-contoh yang familiar untuk anak Indonesia
            - Bersikap ramah, sabar, dan mendorong
            - Berikan penjelasan step-by-step jika diperlukan
            - Gunakan emoji yang sesuai untuk membuat jawaban lebih menarik
            """
            
            full_prompt = f"{system_prompt}\n\nPertanyaan siswa: {student_question}"
            
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=full_prompt),
                    ],
                ),
            ]
            
            tools = [
                types.Tool(googleSearch=types.GoogleSearch()),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    thinking_budget=0,
                ),
                tools=tools,
                temperature=0.7,
                max_output_tokens=1000,
            )
            
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            print(f"Error generating streaming response: {e}")
            yield f"Maaf, terjadi kesalahan dalam memproses pertanyaan kamu. Coba lagi ya! 😊"