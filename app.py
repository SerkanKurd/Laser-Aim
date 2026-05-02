from flask import Flask, render_template, Response, jsonify
from lazer_tespit import LaserDetector

app = Flask(__name__)

# Global detector instance
detector = LaserDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Return the generator output as a multipart stream
    return Response(detector.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(detector.get_stats())

@app.route('/clear', methods=['POST'])
def clear():
    detector.clear_hits()
    return jsonify({"status": "success"})

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    from flask import request
    if request.method == 'POST':
        data = request.get_json()
        if data:
            detector.update_config(data)
            return jsonify({"status": "success", "config": detector.get_config()})
        return jsonify({"status": "error", "message": "No data provided"}), 400
    
    return jsonify(detector.get_config())

@app.route('/api/config/reset', methods=['POST'])
def api_config_reset():
    detector.restore_defaults()
    return jsonify({"status": "success", "config": detector.get_config()})

if __name__ == '__main__':
    # Run the app, accessible on all network interfaces
    app.run(host='0.0.0.0', port=4444, debug=True)
