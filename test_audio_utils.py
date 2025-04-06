# test_audio_utils.py
from audio_utils.audio_utils import convert_to_wav  # Import the conversion function

# Test the convert_to_wav function
file_path = "C:\Users\kamal\OneDrive\Desktop\CEFR-FluentVoice"  # Replace with your file path

converted_file = convert_to_wav(file_path)
if converted_file:
    print(f"Conversion successful: {converted_file}")
else:
    print("Conversion failed")
