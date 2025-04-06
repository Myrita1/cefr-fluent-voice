# app.py
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from whisper_model import transcribe_audio  # Assuming this is the transcribe function from your model
from feedback import analyze_cefr_level  # Assuming you have a feedback function for CEFR level

app = Flask(__name__)

# Set up folder to store uploaded files
UPLOAD_FOLDER = 'temp_audio/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Welcome to the CEFR Fluent Voice API! Upload your audio to get CEFR feedback."

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Transcribe the audio and get feedback
        detected_language, transcript = transcribe_audio(file_path)
        cefr_feedback = analyze_cefr_level(transcript)

        # Return the transcription and CEFR feedback as a response
        return jsonify({
            "detected_language": detected_language,
            "transcript": transcript,
            "cefr_feedback": cefr_feedback
        })
    else:
        return jsonify({"error": "Invalid file format. Only .wav, .mp3, .m4a are allowed."})

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)
