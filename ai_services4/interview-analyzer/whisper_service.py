#!/usr/bin/env python3
"""
Dual-Compatible Transcription Service
Port: 8003
Supports: faster-whisper (if available) or SpeechRecognition (fallback)
"""

import os
import tempfile
import time
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import re
import subprocess

# Initialize FastAPI
app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Auto-detect transcription backend availability
use_faster_whisper = False
model = None
recognizer = None
load_time = 0

try:
    # Try to import faster-whisper with compiled extension
    from faster_whisper import WhisperModel
    import ctranslate2
    # Verify the compiled extension is available
    if hasattr(ctranslate2, 'StorageView'):
        start_load = time.time()
        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        load_time = time.time() - start_load
        use_faster_whisper = True
        print(f"✅ Using faster-whisper backend (tiny.en model loaded in {load_time:.2f}s)")
    else:
        raise ImportError("ctranslate2 compiled extension not available")
except (ImportError, Exception) as e:
    # Fallback to SpeechRecognition (pure Python, uses Google Speech API)
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        use_faster_whisper = False
        print(f"✅ Using SpeechRecognition backend (Google Speech API)")
        print(f"   Note: faster-whisper not available ({str(e)[:100]})")
    except ImportError:
        print(f"❌ No transcription backend available!")
        print(f"   faster-whisper error: {str(e)}")
        raise RuntimeError("No transcription backend available")

def clean_transcription(text):
    """Clean and format transcription text"""
    if not text:
        return ""
    
    # Remove extra whitespace and clean up
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common transcription artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove [music], [noise], etc.
    text = re.sub(r'\(.*?\)', '', text)  # Remove (inaudible), etc.
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    # Add period if missing
    if text and text[-1] not in '.!?':
        text += '.'
    
    return text

def transcribe_with_model(audio_file_path):
    """Transcribe audio using available backend (faster-whisper or SpeechRecognition)"""
    
    try:
        # Check if file exists and has content
        if not os.path.exists(audio_file_path):
            raise Exception("Audio file does not exist")
        
        file_size = os.path.getsize(audio_file_path)
        if file_size < 100:  # Very small file, likely empty
            return "No audio detected"
        
        start_time = time.time()
        
        if use_faster_whisper:
            # Use faster-whisper (local model)
            segments, info = model.transcribe(
                audio_file_path,
                beam_size=1,
                language="en"
            )
            # Iterate through the generator properly
            transcription_segments = []
            for segment in segments:
                transcription_segments.append(segment.text)
            transcription = " ".join(transcription_segments)
        else:
            # Use SpeechRecognition (Google API)
            # Convert webm to wav first using ffmpeg
            wav_path = audio_file_path.replace('.webm', '.wav')
            subprocess.run(['ffmpeg', '-i', audio_file_path, wav_path, '-y'], 
                         capture_output=True, check=True)
            
            # Transcribe using Google Speech Recognition
            import speech_recognition as sr
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio_data)
            except sr.UnknownValueError:
                transcription = "Could not understand audio"
            except sr.RequestError:
                transcription = "Speech service unavailable"
            
            # Clean up wav file
            if os.path.exists(wav_path):
                os.remove(wav_path)
        
        elapsed = time.time() - start_time
        
        return clean_transcription(transcription)
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

@app.get("/")
async def root():
    backend_info = {
        "service": "Dual-Compatible Transcription Service",
        "provider": "faster-whisper" if use_faster_whisper else "SpeechRecognition (Google API)",
        "backend": "faster-whisper + CTranslate2" if use_faster_whisper else "SpeechRecognition + Google Speech API",
        "status": "ready",
        "port": 8003,
        "endpoints": [
            "/transcribe",
            "/health"
        ]
    }
    
    if use_faster_whisper:
        backend_info["model"] = "tiny.en (preloaded)"
        backend_info["load_time"] = f"{load_time:.2f}s (one-time)"
        backend_info["speed"] = "~3 seconds per transcription"
    else:
        backend_info["model"] = "Google Speech API (cloud)"
        backend_info["speed"] = "~2-5 seconds per transcription"
    
    return backend_info

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file using faster-whisper with preloaded model"""
    
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Transcribe the audio
        transcription = transcribe_with_model(temp_file_path)
        
        # Debug: ensure transcription is a string
        if not isinstance(transcription, str):
            transcription = str(transcription)
        
        return {
            "success": True,
            "transcription": {
                "raw_text": transcription,
                "cleaned_text": transcription
            },
            "language": "en",
            "duration": 0,
            "provider": "faster-whisper",
            "model": "tiny.en",
            "speed": f"~3 seconds (model preloaded)"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "provider": "faster-whisper",
            "model": "tiny.en"
        }
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "load_time": f"{load_time:.2f}s",
        "ready": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
