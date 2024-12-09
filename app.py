from flask import Flask, render_template, Response, jsonify
import cv2
from utils.detection import ObjectDetector
from utils.webcam import WebcamStream
from utils.excel_recorder import ExcelRecorder
from utils.latest_detections import LatestDetectionsReader
from utils.total_inspections import TotalInspectionsTracker

app = Flask(__name__)
detector = ObjectDetector()
webcam = WebcamStream()
recorder = ExcelRecorder()
detections_reader = LatestDetectionsReader()
total_tracker = TotalInspectionsTracker()

@app.route('/')
def index():
    latest_detections = detections_reader.get_latest_detections(15)  # Get last 15 detections
    return render_template('index.html', latest_detections=latest_detections)

@app.route('/latest_detections')
def latest_detections():
    latest = detections_reader.get_latest_detections(15)
    return jsonify(latest)

@app.route('/part_info')
def part_info():
    info = detector.get_current_part_info()
    # Add total stats to the response
    info['total_stats'] = total_tracker.get_part_stats(info['part_number'])
    return jsonify(info)

@app.route('/record_detection')
def record_detection():
    part_info = detector.get_current_part_info()
    recorder.record_detection(
        part_info['part_number'],
        part_info['bom_part_detected'],
        part_info['clip_detections']
    )
    result = "OK" if part_info['bom_part_detected'] and all(part_info['clip_detections']) else "NOK"
    latest = detections_reader.get_latest_detections(15)
    return jsonify({"status": "success", "result": result, "latest_detections": latest})

def generate_frames():
    while True:
        ret, frame = webcam.read()
        if not ret:
            break
            
        # Process frame with detector
        frame = detector.process_frame(frame)
        
        # Encode frame
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error encoding frame: {e}")
            continue

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=False, port=8080)