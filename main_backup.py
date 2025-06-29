# # import os
# # import tempfile
# # from typing import Optional
# # import requests
# # from fastapi import FastAPI, File, UploadFile, Query, HTTPException
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import JSONResponse, Response
# # import whisper  # OpenAI's Whisper
# # from deep_translator import GoogleTranslator  # Alternative Translation
# # from pydantic import BaseModel
# # import logging
# # from gtts import gTTS
# # import io
# # import torch
# # from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
# # import librosa
# # import numpy as np
# # # Add this at the top of your file
# # import os
# # os.environ["TRANSFORMERS_CACHE"] = "/path/to/accessible/directory/.cache/huggingface"
# # # Make sure this directory exists and is writable
# # # Configure logging
# # logging.basicConfig(
# #     level=logging.INFO,
# #     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# # )
# # logger = logging.getLogger(__name__)

# # app = FastAPI(title="Audio Translation API")

# # # Enable CORS for cross-domain requests
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],  # Adjust for production
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Load Whisper model for Speech-to-Text
# # # Using small model for better multilingual support while maintaining decent speed
# # stt_model = whisper.load_model("small")  

# # # Translation API settings
# # TRANSLATION_API_URL = "https://translate.argosopentech.com/translate"
# # LIBRETRANSLATE_API_KEY = None  # Add API key if required

# # # Supported languages mapping
# # SUPPORTED_LANGUAGES = {
# #     "en": "English",
# #     "es": "Spanish",
# #     "fr": "French",
# #     "de": "German",
# #     "it": "Italian",
# #     "pt": "Portuguese",
# #     "ru": "Russian",
# #     "ja": "Japanese",
# #     "zh": "Chinese",
# #     "hi": "Hindi",
# #     "te": "Telugu",
# #     "ta": "Tamil",
# #     "ar": "Arabic",
# #     "ko": "Korean"
# # }

# # # TTS language mapping (gTTS uses different codes for some languages)
# # TTS_LANGUAGE_MAP = {
# #     "zh": "zh-CN",  # Chinese
# #     "ar": "ar",     # Arabic
# #     "te": "te",     # Telugu - may need fallback if not supported
# #     "hi": "hi",     # Hindi
# #     "en": "en",     # English
# #     "fr": "fr",     # French
# #     "de": "de",     # German
# #     "it": "it",     # Italian
# #     "ja": "ja",     # Japanese
# #     "ko": "ko",     # Korean
# #     "pt": "pt",     # Portuguese
# #     "ru": "ru",     # Russian
# #     "es": "es",     # Spanish
# #     "ta": "ta"      # Tamil
# # }

# # # Initialize Telugu model at module level but don't load until needed
# # telugu_processor = None
# # telugu_model = None

# # def initialize_telugu_model():
# #     """Initialize the Telugu speech recognition model"""
# #     global telugu_processor, telugu_model
    
# #     # Only initialize if not already loaded
# #     if telugu_processor is None or telugu_model is None:
# #         try:
# #             logger.info("Initializing Telugu speech recognition model...")
# #             # Use AI4Bharat's Telugu model
# #             model_name = "ai4bharat/indicwav2vec_v1_telugu"
            
# #             # Print available cache directory
# #             import os
# #             from transformers import TRANSFORMERS_CACHE
# #             logger.info(f"Transformers cache directory: {TRANSFORMERS_CACHE}")
            
# #             # Try to download with explicit cache handling
# #             processor = Wav2Vec2Processor.from_pretrained(
# #                 model_name,
# #                 cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers"),
# #                 force_download=True
# #             )
            
# #             model = Wav2Vec2ForCTC.from_pretrained(
# #                 model_name, 
# #                 cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers"),
# #                 force_download=True
# #             )
            
# #             # Set global variables
# #             telugu_processor = processor
# #             telugu_model = model
            
# #             logger.info("Telugu model loaded successfully")
# #             return processor, model
# #         except Exception as e:
# #             logger.error(f"Failed to load Telugu model: {str(e)}", exc_info=True)
# #             # Try to provide more specific error information
# #             import traceback
# #             logger.error(f"Detailed error: {traceback.format_exc()}")
# #             return None, None
# #     else:
# #         return telugu_processor, telugu_model

# # def transcribe_telugu(audio_path):
# #     """Transcribe Telugu audio using Whisper with Telugu-specific settings"""
# #     try:
# #         logger.info(f"Transcribing Telugu audio using Whisper: {audio_path}")
        
# #         # Use Whisper with explicit Telugu language setting and better quality
# #         result = stt_model.transcribe(
# #             audio_path,
# #             language="telugu",  # Explicitly tell Whisper this is Telugu
# #             fp16=False,  # Use FP32 for better accuracy
# #             task="transcribe"
# #         )
        
# #         transcription = result.get("text", "")
# #         logger.info(f"Telugu transcription successful: {transcription[:50]}...")
# #         return transcription
# #     except Exception as e:
# #         logger.error(f"Telugu transcription error: {str(e)}")
# #         return f"Error: {str(e)}"

# # def get_tts_language_code(lang_code):
# #     """Map API language code to gTTS language code."""
# #     return TTS_LANGUAGE_MAP.get(lang_code, "en")  # Default to English if mapping not found

# # def transcribe_audio(audio_path: str, language: Optional[str] = None) -> str:
# #     """
# #     Transcribe the audio file using Whisper with proper language handling.
# #     """
# #     # Special handling for Telugu, Tamil, and Hindi
# #     if language == "te":
# #         logger.info("Detected Telugu audio, using specialized model")
# #         return transcribe_telugu(audio_path)
    
# #     try:
# #         # Set up options for Whisper
# #         options = {
# #             "fp16": False,  # Use FP32 for better accuracy
# #             "task": "transcribe"
# #         }
        
# #         # Better language handling - map codes to Whisper's expected format
# #         whisper_lang_map = {
# #             "te": "telugu",
# #             "hi": "hindi",
# #             "ta": "tamil",
# #             # Add more mappings as needed
# #         }
        
# #         # Only set language if explicitly provided, otherwise let Whisper detect
# #         if language and language != "auto":
# #             options["language"] = whisper_lang_map.get(language, language)
        
# #         logger.info(f"Transcribing audio with Whisper, options: {options}")
        
# #         # Run transcription
# #         result = stt_model.transcribe(audio_path, **options)
# #         transcription = result.get("text", "")
        
# #         # Log success
# #         if transcription:
# #             logger.info(f"Whisper transcription successful: {len(transcription)} characters")
# #             return transcription
# #         else:
# #             logger.warning("Whisper transcription returned empty result")
# #             return ""
# #     except Exception as e:
# #         logger.error(f"Whisper transcription error: {str(e)}", exc_info=True)
# #         return ""

# # def translate_text(text: str, source_lang: str, target_lang: str) -> str:
# #     """
# #     Translate text using a fallback approach with multiple translation services.
# #     """
# #     # Skip translation if source and target are the same
# #     if source_lang == target_lang or (source_lang == "auto" and target_lang == "en"):
# #         return text
    
# #     # Try LibreTranslate first
# #     try:
# #         logger.info(f"Attempting translation with LibreTranslate: {source_lang} -> {target_lang}")
# #         data = {
# #             "q": text,
# #             "source": source_lang if source_lang != "auto" else "en",
# #             "target": target_lang,
# #             "format": "text",
# #         }
# #         if LIBRETRANSLATE_API_KEY:
# #             data["api_key"] = LIBRETRANSLATE_API_KEY

# #         response = requests.post(TRANSLATION_API_URL, data=data, timeout=5)
# #         if response.status_code == 200:
# #             result = response.json().get("translatedText", "")
# #             if result:
# #                 logger.info("LibreTranslate translation successful")
# #                 return result
# #     except Exception as e:
# #         logger.warning(f"LibreTranslate Error: {str(e)}")
    
# #     # Fallback: Google Translate
# #     try:
# #         logger.info(f"Attempting translation with Google Translate: {source_lang} -> {target_lang}")
# #         src = source_lang if source_lang != "auto" else "auto"
# #         result = GoogleTranslator(source=src, target=target_lang).translate(text)
# #         if result:
# #             logger.info("Google Translate translation successful")
# #             return result
# #     except Exception as e:
# #         logger.error(f"Google Translate Error: {str(e)}", exc_info=True)
    
# #     logger.error("All translation methods failed")
# #     raise HTTPException(status_code=500, detail="Translation failed with all available services")

# # def generate_speech(text: str, lang: str = "en") -> bytes:
# #     """
# #     Generate speech from text using gTTS with proper language handling.
# #     """
# #     try:
# #         # Map to proper TTS language code
# #         tts_lang = get_tts_language_code(lang)
# #         logger.info(f"Generating speech in language: {lang} (TTS code: {tts_lang})")
        
# #         # Create gTTS object with the text
# #         speech = gTTS(text=text, lang=tts_lang, slow=False)
        
# #         # Save to BytesIO object
# #         audio_io = io.BytesIO()
# #         speech.write_to_fp(audio_io)
# #         audio_io.seek(0)
        
# #         return audio_io.getvalue()
# #     except Exception as e:
# #         logger.error(f"TTS Error: {str(e)}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

# # @app.get("/")
# # async def root():
# #     """API root with basic information."""
# #     return {
# #         "message": "Audio Translation API",
# #         "endpoints": [
# #             {"path": "/upload-audio/", "description": "Upload and transcribe audio"},
# #             {"path": "/translate/", "description": "Translate text"},
# #             {"path": "/text-to-speech/", "description": "Convert text to speech"},
# #             {"path": "/speech-to-translation/", "description": "Speech to text + translation"}
# #         ],
# #         "supported_languages": SUPPORTED_LANGUAGES
# #     }

# # @app.post("/upload-audio/")
# # async def upload_audio(
# #     file: UploadFile = File(...),
# #     language: str = Query(None, description="ISO language code (e.g., te for Telugu, hi for Hindi, en for English)")
# # ):
# #     """Upload and transcribe audio file."""
# #     try:
# #         # Get file extension for proper temp file creation
# #         file_ext = os.path.splitext(file.filename)[1].lower() or ".wav"
# #         if not file_ext.startswith('.'):
# #             file_ext = '.' + file_ext
        
# #         # Create temporary file with proper extension
# #         with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
# #             content = await file.read()
# #             tmp.write(content)
# #             temp_filename = tmp.name
        
# #         logger.info(f"Audio file saved to {temp_filename} with language: {language}")
        
# #         # Use transcribe_audio function with language parameter
# #         # It will handle routing to the specific model for Telugu
# #         transcription_text = transcribe_audio(temp_filename, language)
        
# #         # Clean up temporary file
# #         os.unlink(temp_filename)
        
# #         if not transcription_text:
# #             return JSONResponse(
# #                 status_code=422,
# #                 content={"error": "Could not transcribe audio. Please check the file format and try again."}
# #             )
            
# #         return JSONResponse(content={"transcription": transcription_text})
    
# #     except Exception as e:
# #         logger.error(f"Upload error: {str(e)}", exc_info=True)
# #         return JSONResponse(
# #             status_code=500,
# #             content={"error": f"Server error: {str(e)}"}
# #         )

# # class TranslationRequest(BaseModel):
# #     text: str
# #     source_lang: str
# #     target_lang: str

# # @app.post("/translate/")
# # async def translate_text_api(request: TranslationRequest):
# #     """Translate text from source to target language."""
# #     try:
# #         if not request.text:
# #             raise HTTPException(status_code=400, detail="Empty text provided")
            
# #         translated_text = translate_text(
# #             request.text, 
# #             request.source_lang, 
# #             request.target_lang
# #         )
# #         return {"translated_text": translated_text}
    
# #     except HTTPException:
# #         raise
# #     except Exception as e:
# #         logger.error(f"Translation error: {str(e)}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# # @app.post("/text-to-speech/")
# # async def text_to_speech(
# #     text: str = Query(..., description="Text to convert to speech"),
# #     lang: str = Query("en", description="Language code (e.g., en, es, fr)")
# # ):
# #     """Convert text to speech audio and return as binary response."""
# #     try:
# #         audio_data = generate_speech(text, lang)
# #         return Response(
# #             content=audio_data,
# #             media_type="audio/mp3"
# #         )
# #     except Exception as e:
# #         logger.error(f"TTS error: {str(e)}", exc_info=True)
# #         raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

# # @app.post("/speech-to-translation/")
# # async def speech_to_translation(
# #     file: UploadFile = File(...),
# #     source_lang: str = Query(None, description="Source language (e.g., te for Telugu, hi for Hindi)"),
# #     target_lang: str = Query("en", description="Target language (e.g., hi for Hindi, en for English)")
# # ):
# #     """Combined endpoint: Speech-to-Text + Translation + TTS."""
# #     try:
# #         # Get file extension
# #         file_ext = os.path.splitext(file.filename)[1].lower() or ".wav"
# #         if not file_ext.startswith('.'):
# #             file_ext = '.' + file_ext
            
# #         # Save uploaded file
# #         with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
# #             content = await file.read()
# #             tmp.write(content)
# #             temp_filename = tmp.name
        
# #         # Step 1: Transcribe using the appropriate model based on language
# #         logger.info(f"Processing speech to translation: {source_lang} -> {target_lang}")
# #         transcribed_text = transcribe_audio(temp_filename, source_lang)
        
# #         # Clean up
# #         os.unlink(temp_filename)
        
# #         if not transcribed_text:
# #             return JSONResponse(
# #                 status_code=422,
# #                 content={"error": "Transcription failed. Please check the audio file."}
# #             )
        
# #         # Step 2: Translate
# #         translated_text = translate_text(transcribed_text, source_lang or "auto", target_lang)
        
# #         # Step 3: Generate speech from translated text
# #         audio_data = generate_speech(translated_text, target_lang)
        
# #         # Encode audio data for JSON response
# #         import base64
# #         audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
# #         # Return all results
# #         return JSONResponse(content={
# #             "original_text": transcribed_text,
# #             "translated_text": translated_text,
# #             "audio_data_base64": audio_base64
# #         })
    
# #     except Exception as e:
# #         logger.error(f"Speech-to-translation error: {str(e)}", exc_info=True)
# #         return JSONResponse(
# #             status_code=500,
# #             content={"error": f"Processing failed: {str(e)}"}
# #         )

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000)


# import os
# import tempfile
# import json
# from typing import Optional
# import requests
# from fastapi import FastAPI, File, UploadFile, Query, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse, Response
# import whisper  # OpenAI's Whisper
# from deep_translator import GoogleTranslator  # Alternative Translation
# from pydantic import BaseModel
# import logging
# from gtts import gTTS
# import io
# import torch
# import librosa
# import numpy as np
# import wave
# import soundfile as sf  # For writing WAV files
# from pydub import AudioSegment  # For converting MP3 to WAV

# # Vosk import for Telugu transcription
# from vosk import Model as VoskModel, KaldiRecognizer

# # Set a writable cache directory for Hugging Face models (if needed)
# os.environ["TRANSFORMERS_CACHE"] = "/path/to/accessible/directory/.cache/huggingface"

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# )
# logger = logging.getLogger(__name__)

# app = FastAPI(title="Audio Translation API")

# # Enable CORS for cross-domain requests
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load Whisper model for Speech-to-Text (using the small model for a balance of speed and multilingual support)
# stt_model = whisper.load_model("small")

# # Global variable for the Vosk Telugu model
# vosk_telugu_model = None

# def load_vosk_telugu_model(model_path: str = "vosk-model-small-te-0.42"):
#     """
#     Load the Vosk Telugu model.
#     Download from: https://alphacephei.com/vosk/models
#     and extract it to the specified folder.
#     """
#     global vosk_telugu_model
#     if vosk_telugu_model is None:
#         try:
#             logger.info("Loading Vosk Telugu model...")
#             vosk_telugu_model = VoskModel(model_path)
#             logger.info("Vosk Telugu model loaded successfully")
#         except Exception as e:
#             logger.error(f"Failed to load Vosk Telugu model: {str(e)}", exc_info=True)
#             raise HTTPException(status_code=500, detail="Vosk Telugu model could not be loaded")
#     return vosk_telugu_model

# def transcribe_telugu_vosk(audio_path: str) -> str:
#     """
#     Transcribe Telugu audio using the Vosk engine.
#     If the input audio is not a WAV file, convert it to a 16kHz mono WAV file using pydub.
#     """
#     try:
#         # Load the Vosk model
#         model = load_vosk_telugu_model()
        
#         # Check the file extension; if not WAV, convert using pydub.
#         file_ext = os.path.splitext(audio_path)[1].lower()
#         if file_ext != ".wav":
#             logger.info("Converting audio to WAV using pydub")
#             sound = AudioSegment.from_file(audio_path)
#             # Convert to 16kHz, mono channel
#             sound = sound.set_frame_rate(16000).set_channels(1)
#             # Use mkstemp to create a temporary file and close its file descriptor immediately.
#             temp_wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
#             os.close(temp_wav_fd)
#             sound.export(wav_path, format="wav")
#         else:
#             wav_path = audio_path

#         # Process the WAV file with Vosk using a context manager.
#         transcription = ""
#         with wave.open(wav_path, "rb") as wf:
#             rec = KaldiRecognizer(model, wf.getframerate())
#             rec.SetWords(True)
#             while True:
#                 data = wf.readframes(4000)
#                 if len(data) == 0:
#                     break
#                 if rec.AcceptWaveform(data):
#                     res = json.loads(rec.Result())
#                     transcription += " " + res.get("text", "")
#             # Capture any remaining partial results.
#             res = json.loads(rec.FinalResult())
#             transcription += " " + res.get("text", "")
#         transcription = transcription.strip()
#         logger.info(f"Vosk Telugu transcription: {transcription[:50]}...")

#         # If a temporary WAV was created, attempt to delete it.
#         if file_ext != ".wav":
#             try:
#                 os.unlink(wav_path)
#             except Exception as e:
#                 logger.warning(f"Could not remove temporary file {wav_path}: {str(e)}")
        
#         return transcription
#     except Exception as e:
#         logger.error(f"Vosk Telugu transcription error: {str(e)}", exc_info=True)
#         return ""


# def transcribe_audio(audio_path: str, language: Optional[str] = None) -> str:
#     """
#     Transcribe the audio file.
#     If language is 'te' (Telugu), use Vosk; otherwise, use Whisper.
#     """
#     if language == "te":
#         logger.info("Using Vosk for Telugu transcription")
#         return transcribe_telugu_vosk(audio_path)
    
#     try:
#         options = {
#             "fp16": False,  # Use FP32 for better accuracy on CPU
#             "task": "transcribe"
#         }
#         if language and language != "auto":
#             options["language"] = language
        
#         logger.info(f"Transcribing audio with Whisper, options: {options}")
#         result = stt_model.transcribe(audio_path, **options)
#         transcription = result.get("text", "")
#         if transcription:
#             logger.info(f"Whisper transcription successful: {len(transcription)} characters")
#             return transcription
#         else:
#             logger.warning("Whisper transcription returned empty result")
#             return ""
#     except Exception as e:
#         logger.error(f"Whisper transcription error: {str(e)}", exc_info=True)
#         return ""

# def translate_text(text: str, source_lang: str, target_lang: str) -> str:
#     """
#     Translate text using LibreTranslate with a fallback to Google Translate.
#     """
#     if source_lang == target_lang or (source_lang == "auto" and target_lang == "en"):
#         return text
    
#     try:
#         logger.info(f"Attempting translation with LibreTranslate: {source_lang} -> {target_lang}")
#         data = {
#             "q": text,
#             "source": source_lang if source_lang != "auto" else "en",
#             "target": target_lang,
#             "format": "text",
#         }
#         response = requests.post("https://translate.argosopentech.com/translate", data=data, timeout=5)
#         if response.status_code == 200:
#             result = response.json().get("translatedText", "")
#             if result:
#                 logger.info("LibreTranslate translation successful")
#                 return result
#     except Exception as e:
#         logger.warning(f"LibreTranslate Error: {str(e)}")
    
#     try:
#         logger.info(f"Attempting translation with Google Translate: {source_lang} -> {target_lang}")
#         src = source_lang if source_lang != "auto" else "auto"
#         result = GoogleTranslator(source=src, target=target_lang).translate(text)
#         if result:
#             logger.info("Google Translate translation successful")
#             return result
#     except Exception as e:
#         logger.error(f"Google Translate Error: {str(e)}", exc_info=True)
    
#     logger.error("All translation methods failed")
#     raise HTTPException(status_code=500, detail="Translation failed with all available services")

# def get_tts_language_code(lang_code):
#     """Map API language code to gTTS language code."""
#     TTS_LANGUAGE_MAP = {
#         "zh": "zh-CN",  # Chinese
#         "ar": "ar",     # Arabic
#         "te": "te",     # Telugu (gTTS support for Telugu might be limited)
#         "hi": "hi",     # Hindi
#         "en": "en",     # English
#         "fr": "fr",     # French
#         "de": "de",     # German
#         "it": "it",     # Italian
#         "ja": "ja",     # Japanese
#         "ko": "ko",     # Korean
#         "pt": "pt",     # Portuguese
#         "ru": "ru",     # Russian
#         "es": "es",     # Spanish
#         "ta": "ta"      # Tamil
#     }
#     return TTS_LANGUAGE_MAP.get(lang_code, "en")

# def generate_speech(text: str, lang: str = "en") -> bytes:
#     """
#     Generate speech from text using gTTS with proper language handling.
#     """
#     try:
#         tts_lang = get_tts_language_code(lang)
#         logger.info(f"Generating speech in language: {lang} (TTS code: {tts_lang})")
#         speech = gTTS(text=text, lang=tts_lang, slow=False)
#         audio_io = io.BytesIO()
#         speech.write_to_fp(audio_io)
#         audio_io.seek(0)
#         return audio_io.getvalue()
#     except Exception as e:
#         logger.error(f"TTS Error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

# @app.get("/")
# async def root():
#     """API root with basic information."""
#     return {
#         "message": "Audio Translation API",
#         "endpoints": [
#             {"path": "/upload-audio/", "description": "Upload and transcribe audio"},
#             {"path": "/translate/", "description": "Translate text"},
#             {"path": "/text-to-speech/", "description": "Convert text to speech"},
#             {"path": "/speech-to-translation/", "description": "Speech to text + translation"}
#         ],
#         "supported_languages": {
#             "en": "English",
#             "es": "Spanish",
#             "fr": "French",
#             "de": "German",
#             "it": "Italian",
#             "pt": "Portuguese",
#             "ru": "Russian",
#             "ja": "Japanese",
#             "zh": "Chinese",
#             "hi": "Hindi",
#             "te": "Telugu",
#             "ta": "Tamil",
#             "ar": "Arabic",
#             "ko": "Korean"
#         }
#     }

# @app.post("/upload-audio/")
# async def upload_audio(
#     file: UploadFile = File(...),
#     language: str = Query(None, description="ISO language code (e.g., te for Telugu, hi for Hindi, en for English)")
# ):
#     """Upload and transcribe audio file."""
#     try:
#         file_ext = os.path.splitext(file.filename)[1].lower() or ".wav"
#         if not file_ext.startswith('.'):
#             file_ext = '.' + file_ext
        
#         with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
#             content = await file.read()
#             tmp.write(content)
#             temp_filename = tmp.name
        
#         logger.info(f"Audio file saved to {temp_filename} with language: {language}")
#         transcription_text = transcribe_audio(temp_filename, language)
#         os.unlink(temp_filename)
        
#         if not transcription_text:
#             return JSONResponse(
#                 status_code=422,
#                 content={"error": "Could not transcribe audio. Please check the file format and try again."}
#             )
            
#         return JSONResponse(content={"transcription": transcription_text})
    
#     except Exception as e:
#         logger.error(f"Upload error: {str(e)}", exc_info=True)
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Server error: {str(e)}"}
#         )

# class TranslationRequest(BaseModel):
#     text: str
#     source_lang: str
#     target_lang: str

# @app.post("/translate/")
# async def translate_text_api(request: TranslationRequest):
#     """Translate text from source to target language."""
#     try:
#         if not request.text:
#             raise HTTPException(status_code=400, detail="Empty text provided")
            
#         translated_text = translate_text(
#             request.text, 
#             request.source_lang, 
#             request.target_lang
#         )
#         return {"translated_text": translated_text}
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Translation error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

# @app.post("/text-to-speech/")
# async def text_to_speech(
#     text: str = Query(..., description="Text to convert to speech"),
#     lang: str = Query("en", description="Language code (e.g., en, es, fr)")
# ):
#     """Convert text to speech audio and return as binary response."""
#     try:
#         audio_data = generate_speech(text, lang)
#         return Response(
#             content=audio_data,
#             media_type="audio/mp3"
#         )
#     except Exception as e:
#         logger.error(f"TTS error: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

# @app.post("/speech-to-translation/")
# async def speech_to_translation(
#     file: UploadFile = File(...),
#     source_lang: str = Query(None, description="Source language (e.g., te for Telugu, hi for Hindi)"),
#     target_lang: str = Query("en", description="Target language (e.g., hi for Hindi, en for English)")
# ):
#     """Combined endpoint: Speech-to-Text + Translation + TTS."""
#     try:
#         file_ext = os.path.splitext(file.filename)[1].lower() or ".wav"
#         if not file_ext.startswith('.'):
#             file_ext = '.' + file_ext
            
#         with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
#             content = await file.read()
#             tmp.write(content)
#             temp_filename = tmp.name
        
#         logger.info(f"Processing speech to translation: {source_lang} -> {target_lang}")
#         transcribed_text = transcribe_audio(temp_filename, source_lang)
#         os.unlink(temp_filename)
        
#         if not transcribed_text:
#             return JSONResponse(
#                 status_code=422,
#                 content={"error": "Transcription failed. Please check the audio file."}
#             )
        
#         translated_text = translate_text(transcribed_text, source_lang or "auto", target_lang)
#         audio_data = generate_speech(translated_text, target_lang)
        
#         import base64
#         audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
#         return JSONResponse(content={
#             "original_text": transcribed_text,
#             "translated_text": translated_text,
#             "audio_data_base64": audio_base64
#         })
    
#     except Exception as e:
#         logger.error(f"Speech-to-translation error: {str(e)}", exc_info=True)
#         return JSONResponse(
#             status_code=500,
#             content={"error": f"Processing failed: {str(e)}"}
#         )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
import os
import tempfile
import json
import requests
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import whisper
from deep_translator import GoogleTranslator
from pydantic import BaseModel
import logging
from gtts import gTTS
import io
import speech_recognition as sr
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Translation API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model once
stt_model = whisper.load_model("small")

def convert_to_wav(audio_path: str) -> str:
    """Convert audio to WAV format suitable for speech recognition."""
    file_ext = os.path.splitext(audio_path)[1].lower()
    if file_ext != ".wav":
        sound = AudioSegment.from_file(audio_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        temp_wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_wav_fd)
        sound.export(wav_path, format="wav")
        return wav_path
    return audio_path

def transcribe_audio_google(audio_path: str) -> str:
    """Transcribe Telugu audio using Google Speech Recognition."""
    wav_path = convert_to_wav(audio_path)
    is_temp_file = wav_path != audio_path
    
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        transcription = recognizer.recognize_google(audio_data, language="te-IN")
    except (sr.RequestError, sr.UnknownValueError) as e:
        logger.warning(f"Google Speech Recognition error: {str(e)}")
        transcription = ""
    finally:
        if is_temp_file:
            try:
                os.unlink(wav_path)
            except Exception:
                pass
    return transcription

def transcribe_audio(audio_path: str, language: Optional[str] = None) -> str:
    """Transcribe audio using appropriate service based on language."""
    if language == "te":
        return transcribe_audio_google(audio_path)
    
    try:
        options = {"fp16": False, "task": "transcribe"}
        if language and language != "auto":
            options["language"] = language
        
        result = stt_model.transcribe(audio_path, **options)
        return result.get("text", "")
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return ""

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using available translation services."""
    if source_lang == target_lang or (source_lang == "auto" and target_lang == "en"):
        return text
    
    # Try LibreTranslate first
    try:
        data = {
            "q": text,
            "source": source_lang if source_lang != "auto" else "en",
            "target": target_lang,
            "format": "text",
        }
        response = requests.post("https://translate.argosopentech.com/translate", data=data, timeout=5)
        if response.status_code == 200:
            result = response.json().get("translatedText", "")
            if result:
                return result
    except Exception as e:
        logger.warning(f"LibreTranslate error: {str(e)}")
    
    # Fall back to Google Translate
    try:
        src = source_lang if source_lang != "auto" else "auto"
        result = GoogleTranslator(source=src, target=target_lang).translate(text)
        if result:
            return result
    except Exception as e:
        logger.error(f"Google Translate error: {str(e)}")
    
    raise HTTPException(status_code=500, detail="Translation failed")

def generate_speech(text: str, lang: str = "en") -> bytes:
    """Generate speech from text."""
    # Map language codes for TTS
    lang_map = {
        "zh": "zh-CN", "ar": "ar", "te": "te", "hi": "hi", "en": "en", 
        "fr": "fr", "de": "de", "it": "it", "ja": "ja", "ko": "ko", 
        "pt": "pt", "ru": "ru", "es": "es", "ta": "ta"
    }
    
    tts_lang = lang_map.get(lang, "en")
    try:
        speech = gTTS(text=text, lang=tts_lang, slow=False)
        audio_io = io.BytesIO()
        speech.write_to_fp(audio_io)
        audio_io.seek(0)
        return audio_io.getvalue()
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech failed: {str(e)}")

@app.get("/")
async def root():
    """API root endpoint."""
    return {
        "message": "Audio Translation API",
        "endpoints": [
            "/upload-audio/", 
            "/translate/", 
            "/text-to-speech/", 
            "/speech-to-translation/"
        ],
        "supported_languages": {
            "en": "English", "es": "Spanish", "fr": "French", "de": "German",
            "it": "Italian", "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
            "zh": "Chinese", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
            "ar": "Arabic", "ko": "Korean"
        }
    }

@app.post("/upload-audio/")
async def upload_audio(
    file: UploadFile = File(...),
    language: str = Query(None, description="ISO language code")
):
    """Upload and transcribe audio file."""
    try:
        file_ext = os.path.splitext(file.filename)[1].lower() or ".wav"
        if not file_ext.startswith('.'):
            file_ext = '.' + file_ext
        
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_filename = tmp.name
        
        transcription_text = transcribe_audio(temp_filename, language)
        os.unlink(temp_filename)
        
        if not transcription_text:
            return JSONResponse(
                status_code=422,
                content={"error": "Could not transcribe audio"}
            )
            
        return {"transcription": transcription_text}
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate/")
async def translate_text_api(request: TranslationRequest):
    """Translate text from source to target language."""
    if not request.text:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        translated_text = translate_text(
            request.text, 
            request.source_lang, 
            request.target_lang
        )
        return {"translated_text": translated_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/text-to-speech/")
async def text_to_speech(
    text: str = Query(..., description="Text to convert to speech"),
    lang: str = Query("en", description="Language code")
):
    """Convert text to speech audio."""
    audio_data = generate_speech(text, lang)
    return Response(content=audio_data, media_type="audio/mp3")

@app.post("/speech-to-translation/")
async def speech_to_translation(
    file: UploadFile = File(...),
    source_lang: str = Query(None, description="Source language"),
    target_lang: str = Query("en", description="Target language")
):
    """Combined endpoint: Speech-to-Text + Translation + TTS."""
    try:
        file_ext = os.path.splitext(file.filename)[1].lower() or ".wav"
        if not file_ext.startswith('.'):
            file_ext = '.' + file_ext
            
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_filename = tmp.name
        
        transcribed_text = transcribe_audio(temp_filename, source_lang)
        os.unlink(temp_filename)
        
        if not transcribed_text:
            return JSONResponse(
                status_code=422,
                content={"error": "Transcription failed"}
            )
        
        translated_text = translate_text(transcribed_text, source_lang or "auto", target_lang)
        audio_data = generate_speech(translated_text, target_lang)
        
        import base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            "original_text": transcribed_text,
            "translated_text": translated_text,
            "audio_data_base64": audio_base64
        }
    except Exception as e:
        logger.error(f"Speech-to-translation error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)