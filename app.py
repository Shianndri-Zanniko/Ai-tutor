import streamlit as st
import os
import tempfile
import time
from pathlib import Path
from audio_recorder_streamlit import audio_recorder
import dotenv

# Import our custom modules
from whisper_asr import WhisperASR
from gemini_llm import GeminiLLM
from gemini_tts import GeminiTTS

# Load environment variables
dotenv.load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Tutor for Elementary School (Indonesian)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #2E86C1;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.subtitle {
    text-align: center;
    color: #5D6D7E;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}
.section-header {
    color: #2E86C1;
    font-size: 1.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.instruction-box {
    background-color: #EBF5FB;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2E86C1;
    margin-bottom: 2rem;
    color: #2C3E50;
}
.question-display {
    background-color: #F8F9FA;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28B463;
    margin-bottom: 1rem;
    color: #2C3E50;
}
.answer-display {
    background-color: #EBF5FB;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2E86C1;
    margin-bottom: 1rem;
    color: #2C3E50;
}
.processing-spinner {
    text-align: center;
    color: #2E86C1;
    font-size: 1.1rem;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'whisper_asr' not in st.session_state:
    st.session_state.whisper_asr = None
if 'gemini_llm' not in st.session_state:
    st.session_state.gemini_llm = None
if 'gemini_tts' not in st.session_state:
    st.session_state.gemini_tts = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = ""
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'first_recording' not in st.session_state:
    st.session_state.first_recording = True

@st.cache_resource
def load_models():
    """Load all AI models with caching"""
    try:
        whisper_asr = WhisperASR()
        gemini_llm = GeminiLLM()
        gemini_tts = GeminiTTS()
        return whisper_asr, gemini_llm, gemini_tts
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def process_audio_question(audio_bytes):
    """Process recorded audio through the complete pipeline"""
    if audio_bytes is None:
        return
    
    st.session_state.processing = True
    
    # Create columns for progress tracking
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        st.info("🎤 Memproses rekaman suara...")
    
    # Step 1: Speech to Text
    try:
        if st.session_state.whisper_asr is None:
            st.session_state.whisper_asr, st.session_state.gemini_llm, st.session_state.gemini_tts = load_models()
        
        transcribed_text = st.session_state.whisper_asr.transcribe_audio_bytes(audio_bytes)
        
        if transcribed_text:
            st.session_state.current_question = transcribed_text
            
            with progress_col2:
                st.info("🤖 Menghasilkan jawaban...")
            
            # Step 2: Generate Answer
            answer = st.session_state.gemini_llm.generate_tutor_response(transcribed_text)
            
            if answer:
                st.session_state.current_answer = answer
                
                with progress_col3:
                    st.info("🔊 Membuat suara jawaban...")
                
                # Step 3: Text to Speech
                audio_file_path = st.session_state.gemini_tts.text_to_speech(answer)
                st.session_state.audio_file_path = audio_file_path
                
                st.success("✅ Selesai! Jawaban sudah siap.")
            else:
                st.error("Maaf, terjadi kesalahan dalam menghasilkan jawaban.")
        else:
            st.error("Maaf, tidak dapat memahami rekaman suara. Coba lagi dengan suara yang lebih jelas.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
    
    finally:
        st.session_state.processing = False

def main():
    """Main Streamlit application"""
    
    # Load models at startup if not already loaded
    if st.session_state.whisper_asr is None:
        with st.spinner("Loading AI models... This may take a moment on first run."):
            st.session_state.whisper_asr, st.session_state.gemini_llm, st.session_state.gemini_tts = load_models()
        # Explicitly load the Whisper model
        if st.session_state.whisper_asr:
            with st.spinner("Initializing Whisper ASR model..."):
                st.session_state.whisper_asr.load_model()
    
    # Header
    st.markdown('<h1 class="main-header">🎓 AI Tutor for Elementary School (Indonesian) - Using Whisper ASR</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Ask a question in Indonesian using your voice, and the AI tutor will answer!</p>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="instruction-box">
        <strong>Cara Penggunaan:</strong><br>
        Click 'Start Recording', ask your question clearly, then click 'Stop Recording'.
    </div>
    """, unsafe_allow_html=True)
    
    # Section 1: Voice Recording
    st.markdown('<h2 class="section-header">1. Ask Your Question (Voice)</h2>', unsafe_allow_html=True)
    
    # Audio recorder widget (disabled during processing)
    if st.session_state.processing:
        st.info("🔄 Processing audio... Please wait before recording again.")
        # Show disabled microphone
        st.markdown("""
        <div style="text-align: center; opacity: 0.3; pointer-events: none;">
            <i class="fas fa-microphone-lines" style="font-size: 6rem; color: #cccccc;"></i>
            <p>Recording disabled during processing</p>
        </div>
        """, unsafe_allow_html=True)
        audio_bytes = None
    else:
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone-lines",
            icon_size="6x",
        )
    
    # Process audio when recording is complete
    if audio_bytes:
        # Skip the first recording (usually empty)
        if st.session_state.first_recording:
            st.session_state.first_recording = False
            st.info("🎤 Ready to record! Click the microphone again to start your question.")
        else:
            # Display audio player
            st.audio(audio_bytes, format="audio/wav")
            
            # Process the audio
            if not st.session_state.processing:
                process_audio_question(audio_bytes)
    
    # Section 2: Interaction Results
    st.markdown('<h2 class="section-header">2. Interaction</h2>', unsafe_allow_html=True)
    
    # Show processing status
    if st.session_state.processing:
        st.markdown('<div class="processing-spinner">Processing your question...</div>', unsafe_allow_html=True)
        st.spinner("Please wait...")
    
    # Display results if available
    if st.session_state.current_question and not st.session_state.processing:
        # Display transcribed question
        st.markdown("**You asked (Text):**")
        st.markdown(f'<div class="question-display">{st.session_state.current_question}</div>', unsafe_allow_html=True)
        
        # Display tutor's answer
        if st.session_state.current_answer:
            st.markdown("**Tutor's Answer (Text):**")
            st.markdown(f'<div class="answer-display">{st.session_state.current_answer}</div>', unsafe_allow_html=True)
            
            # Display audio answer
            st.markdown("**Tutor's Answer (Voice):**")
            if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
                st.audio(st.session_state.audio_file_path)
            else:
                st.info("Audio is being generated, please wait...")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### About")
        st.info("""
        This AI Tutor uses:
        - **Whisper ASR**: conevonce/whisper-small-id3 (Indonesian fine-tuned)
        - **Gemini LLM**: 2.5 Flash Lite for educational responses
        - **Gemini TTS**: 2.5 Flash Preview with Zephyr voice
        """)
        
        st.markdown("### Tips")
        st.markdown("""
        - Speak clearly in Indonesian
        - Ask questions about school subjects
        - Wait for processing to complete
        - Use a quiet environment for better recognition
        """)
        
        # Clear conversation button
        if st.button("🔄 Clear Conversation"):
            st.session_state.current_question = ""
            st.session_state.current_answer = ""
            st.session_state.audio_file_path = None
            st.rerun()

if __name__ == "__main__":
    main()