import whisper  # Import the Whisper library

def transcribe_audio(file_path):
    """Transcribe audio and detect language using Whisper."""
    # Load the Whisper model (medium model for better speed and accuracy)
    model = whisper.load_model("medium", download_root="C:\\Users\\kamal\\.cache\\whisper")
    
    # Transcribe the audio file
    result = model.transcribe(file_path)
    
    # Return the detected language and transcript
    return result["language"], result["text"]
