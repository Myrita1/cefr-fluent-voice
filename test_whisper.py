import whisper
import os
from pydub import AudioSegment

# 1. Convert non-wav files to .wav using pydub
def convert_to_wav(file_path):
    """Convert audio files (mp3, m4a, etc.) to wav using pydub"""
    try:
        # If the file is not a .wav, convert it
        if not file_path.endswith(".wav"):
            audio = AudioSegment.from_file(file_path)  # Load the file
            wav_path = file_path.rsplit('.', 1)[0] + ".wav"  # Save as .wav
            audio.export(wav_path, format="wav")
            print(f"Converted {file_path} to {wav_path}")
            return wav_path
        return file_path  # Return the file path if it's already a .wav
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None

# 2. Load the Whisper model (medium model for balance of accuracy and speed)
model = whisper.load_model("medium", download_root="C:\\Users\\kamal\\.cache\\whisper")

# 3. Function to transcribe the audio and detect language
def transcribe_audio(file_path):
    """Transcribe audio and detect language using Whisper"""
    # Convert non-wav files to wav
    file_path = convert_to_wav(file_path)
    
    if file_path:
        # Transcribe the audio file using Whisper
        result = model.transcribe(file_path)
        detected_language = result["language"]
        transcript = result["text"]

        # Print detected language and transcription result
        print(f"Detected language: {detected_language}")
        print(f"Transcript: {transcript}")

        # Return the language and transcript
        return detected_language, transcript
    else:
        print(f"Error: Could not process file {file_path}")
        return None, None

# 4. Example usage: Provide the path to your audio file here (use a .wav file)
audio_file = "C:\\Users\\kamal\\Desktop\\CEFR-FluentVoice\\example.wav"  # Update to .wav file

# 5. Transcribe the audio and get the result
detected_language, transcript = transcribe_audio(audio_file)

# 6. Save transcript to a text file (optional)
def save_transcript_to_file(transcript, output_path="transcript.txt"):
    """Save the transcript to a text file."""
    with open(output_path, "w") as f:
        f.write(transcript)
    print(f"Transcript saved to {output_path}")

# Save the transcript
if transcript:
    save_transcript_to_file(transcript, "example_transcript.txt")
