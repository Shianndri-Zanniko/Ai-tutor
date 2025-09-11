# AI Tutor for Elementary School (Indonesian) 🎓

An AI-powered voice tutor application designed for Indonesian elementary school students. The application uses speech recognition to understand questions in Indonesian and provides educational responses through both text and voice.

## Features

- **Voice Input**: Record questions in Indonesian using Whisper ASR
- **AI Tutoring**: Get educational responses from Gemini 2.5 Flash Lite
- **Voice Output**: Listen to answers with Gemini TTS (Zephyr voice)
- **User-Friendly Interface**: Clean Streamlit web interface
- **Indonesian Language Support**: Fine-tuned for Indonesian language

## Architecture

1. **Speech Recognition**: Uses `conevonce/whisper-small-id3` (Whisper Small fine-tuned for Indonesian)
2. **Language Model**: Gemini 2.5 Flash Lite for generating educational responses
3. **Text-to-Speech**: Gemini 2.5 Flash Preview TTS with Zephyr voice
4. **Frontend**: Streamlit web application

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Ai-Tutor
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root and add your API keys:
```bash
# Required: Get your Gemini API key from Google AI Studio
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: For private Hugging Face models (not required for conevonce/whisper-small-id3)
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

### 5. Get Your Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### 6. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

1. **Record Your Question**: Click the microphone button to start recording and ask your question clearly in Indonesian
2. **Automatic Processing**: When you stop, the system will automatically:
   - Convert your speech to text using Whisper ASR
   - Generate an educational response using Gemini LLM
   - Convert the response to speech using Gemini TTS
3. **View Results**: See both text and audio responses from the AI tutor
4. **Play Audio**: Click the audio player to hear the tutor's response

## File Structure

```
Ai-Tutor/
├── app.py                 # Main Streamlit application
├── whisper_asr.py         # Whisper speech recognition module
├── gemini_llm.py          # Gemini LLM module
├── gemini_tts.py          # Gemini TTS module
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys)
├── venv/                  # Virtual environment
└── README.md             # This file
```

## Requirements

- Python 3.8+
- Internet connection for AI model APIs
- Microphone for voice input
- Speakers/headphones for voice output

## Dependencies

- `streamlit` - Web application framework
- `google-genai` - Google Gemini AI SDK
- `transformers` - Hugging Face transformers (for Whisper)
- `torch` - PyTorch (ML framework)
- `soundfile` - Audio file I/O
- `librosa` - Audio processing
- `python-dotenv` - Environment variable management
- `audio-recorder-streamlit` - Audio recording component
- `numpy` - Numerical computing

## Tips for Best Results

1. **Clear Speech**: Speak clearly and at a moderate pace
2. **Quiet Environment**: Use in a quiet room for better speech recognition
3. **Indonesian Language**: Ask questions in Indonesian for best results
4. **Educational Topics**: Focus on elementary school subjects (math, science, Indonesian, etc.)

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you activated the virtual environment and installed all dependencies
2. **API Key Errors**: Verify your Gemini API key is correct in the `.env` file
3. **Audio Issues**: Check your microphone permissions in the browser
4. **Model Loading**: First run may take time to download the Whisper model

### Error Messages

- "GEMINI_API_KEY not found": Add your API key to the `.env` file
- "Error loading model": Check internet connection and try restarting
- "No audio data received": Check microphone permissions and audio quality

## Contributing

Feel free to contribute by:
- Reporting bugs
- Suggesting features
- Improving documentation
- Adding support for more languages

## License

This project is for educational purposes. Please respect the terms of service of the APIs used (Google Gemini, Hugging Face).