import os
import torch
import shutil
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip
import tempfile

from workflow import (
    transcribe_batched,
    process_language_arg,
    create_config,
    NeuralDiarizer,
    get_words_speaker_mapping,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    write_srt,
    cleanup
)

app = Flask(__name__)

# Configure the file upload folder and allowed extensions
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav'}

# Ensure the upload folder and results folder exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Check if a file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API to upload the file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    domain_type = request.form.get('domain_type', 'general')  # Default to 'general' if not provided
    language = request.form.get('language', 'en')  # Default to 'en' if not provided
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Call processing function with custom domain_type and language
        result = process_audio(file_path, domain_type, language)

        return jsonify(result), 200
    else:
        return jsonify({"error": "File type not allowed"}), 400

# Function to process the uploaded audio/video file
def process_audio(file_path, domain_type, language):
    batch_size = 8
    whisper_model_name = "large-v2"
    compute_type = "float32"
    suppress_numerals = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Extract audio from video if necessary
    if file_path.endswith('.mp4'):
        video = VideoFileClip(file_path)
        audio = video.audio
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        audio.write_audiofile(audio_file.name)
        video.close()
        audio.close()
        audio_path = audio_file.name
    else:
        audio_path = file_path

    # Run the transcription process
    with torch.cuda.amp.autocast():
        whisper_results, language, audio_waveform = transcribe_batched(
            audio_path,
            language,
            batch_size,
            whisper_model_name,
            compute_type,
            suppress_numerals,
            device,
        )

    # Prepare the diarization model
    temp_path = tempfile.mkdtemp()
    torch.save(audio_waveform, os.path.join(temp_path, "audio_waveform.pt"))

    msdd_model = NeuralDiarizer(cfg=create_config(temp_path, domain_type)).to("cuda")
    msdd_model.diarize()

    # Generate the word speaker mapping and align with speaker timestamps
    speaker_ts = []
    with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    word_timestamps = whisper_results[0]["segments"]
    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # Write the output to text and SRT formats
    output_text_file = os.path.join(RESULTS_FOLDER, "output.txt")
    with open(output_text_file, "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    output_srt_file = os.path.join(RESULTS_FOLDER, "output.srt")
    with open(output_srt_file, "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    # Clean up the temporary files
    cleanup(temp_path)

    return {
        "message": "Processing complete",
        "transcript": output_text_file,
        "srt": output_srt_file,
    }

# API to download the result files
@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    try:
        return send_from_directory(RESULTS_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

# Run the Flask app
if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=8888)
    app.run(debug=True)
