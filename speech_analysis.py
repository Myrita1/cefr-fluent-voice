from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import torch
import librosa

# Load pre-trained Wav2Vec2 model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def transcribe_audio(file_path):
    """Transcribe audio to text using Wav2Vec2"""
    try:
        # Load the audio file (ensure the path is correct and uses the proper encoding)
        audio_input, _ = librosa.load(file_path, sr=16000)
        inputs = processor(audio_input, return_tensors="pt", padding=True)

        # Perform speech-to-text transcription
        with torch.no_grad():
            logits = model(input_values=inputs.input_values).logits

        # Decode the prediction into text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0])

        # Return the transcription if valid
        if transcription:
            return transcription
        else:
            print("Error: No transcription returned.")
            return None
    except Exception as e:
        print(f"Error in transcribing audio: {e}")
        return None

# Load the text classification model for CEFR level prediction
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def analyze_cefr_level(text):
    """Analyze the transcription and predict the CEFR level"""
    # Define possible CEFR levels
    candidate_labels = ["A1", "A2", "B1", "B2", "C1", "C2"]

    # Use zero-shot classification to predict the CEFR level based on the text
    result = classifier(text, candidate_labels)

    # Return the predicted CEFR level and confidence score
    return result['labels'][0], result['scores'][0]

def analyze_speech(file_path):
    """Analyze speech by transcribing and predicting the CEFR level"""
    # Step 1: Transcribe the speech to text using Wav2Vec2
    transcription = transcribe_audio(file_path)
    if transcription is None:
        return None, None, None  # Return None if transcription failed
    print(f"Transcription: {transcription}")

    # Step 2: Analyze the text and predict the CEFR level
    cefr_level, score = analyze_cefr_level(transcription)
    print(f"Predicted CEFR Level: {cefr_level} (Score: {score})")

    return transcription, cefr_level, score

# Example usage: Provide the path to your audio file here
audio_file = r"C:\Users\kamal\OneDrive\Desktop\CEFR-FluentVoice\example.wav"  # Path to your .wav file

# Analyze the speech file
transcription, cefr_level, score = analyze_speech(audio_file)
