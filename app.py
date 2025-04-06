from flask import Flask, request, jsonify
from transformers import pipeline
from speech_analysis import transcribe_audio, analyze_cefr_level

app = Flask(__name__)

# Initialize the zero-shot classifier (for CEFR level prediction)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@app.route('/analyze_speech', methods=['POST'])
def analyze_speech():
    """
    Accept an audio file, transcribe it, and predict the CEFR level of the transcription.
    """
    # Receive the audio file from the request
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found in the request"}), 400

    audio_file = request.files['audio']
    
    # Save the audio file temporarily to the server
    audio_path = 'temp_audio.wav'
    audio_file.save(audio_path)
    
    # Step 1: Transcribe the audio to text
    transcription = transcribe_audio(audio_path)
    
    if transcription is None:
        return jsonify({"error": "Audio transcription failed"}), 500

    # Step 2: Analyze the transcription to predict the CEFR level
    cefr_level, score = analyze_cefr_level(transcription)
    
    # Return the transcription and CEFR level with the score
    return jsonify({
        "transcription": transcription,
        "cefr_level": cefr_level,
        "score": score
    })

if __name__ == "__main__":
    app.run(debug=True)
