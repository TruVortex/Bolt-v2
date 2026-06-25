import os
import sys
import webbrowser
from threading import Timer

from flask import Flask, render_template, request, jsonify

from analyzer import RunningAnalyzer

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = "temp_gait_video.mp4"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            analyzer = RunningAnalyzer()
            results = analyzer.analyze_video(filepath)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    return jsonify({"error": "Invalid file extension. Please upload MP4, MOV, or AVI."}), 400


def open_browser(port_number):
    """Launches the user's default browser on the active local port."""
    webbrowser.open(f"http://127.0.0.1:{port_number}")


if __name__ == '__main__':
    is_frozen = getattr(sys, 'frozen', False)
    is_huggingface = 'SPACE_ID' in os.environ or 'PORT' in os.environ
    if is_huggingface:
        default_port = 7860
    else:
        default_port = 5000
    port = int(os.environ.get('PORT', default_port))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    if is_frozen:
        Timer(1.5, lambda: open_browser(port)).start()
        app.run(host='127.0.0.1', port=port, debug=False)
    else:
        app.run(host='0.0.0.0', port=port, debug=debug_mode)
