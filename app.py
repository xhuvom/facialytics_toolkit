from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
import shutil
import subprocess
import sys
import cv2
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import insightface
import json

# Import backend functions for restoration, inpainting, and face comparison.
from utils import face_restoration, face_inpainting, face_comparison

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

# Configure folders for uploads and outputs.
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
OUTPUT_FOLDER = os.path.join("static", "outputs")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Create necessary directories if they don't exist.
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_video(filename):
    video_extensions = {'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

# ------------------------------
# Face Recognition helpers and state
# ------------------------------
# Initialize InsightFace model (GPU if available)
face_rec_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_rec_model.prepare(ctx_id=0, det_size=(640, 640))

# In-memory job store
jobs = {}

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

def image_to_base64(img):
    # Convert BGR to RGB and encode as JPEG base64 string
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def pad_image(img):
    # Pad image to multiples of 32 to avoid shape mismatch
    h, w = img.shape[:2]
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    padded = np.zeros((new_h, new_w, 3), dtype=img.dtype)
    padded[:h, :w, :] = img
    return padded

# ------------------------------
# Route to serve uploaded files.
# ------------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ------------------------------
# Route for Restoration/Inpainting.
# ------------------------------
@app.route("/", methods=["GET"])
def landing():
    return render_template('landing.html')

@app.route("/restoration", methods=["GET", "POST"])
def index():
    output_image_url = None
    if request.method == "POST":
        if "face_image" not in request.files:
            return redirect(request.url)
        file = request.files["face_image"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            file.save(file_path)
            
            # Determine which button was pressed: Restoration or Inpainting.
            if "restore" in request.form:
                fidelity_weight = float(request.form.get("fidelity_weight", 0.5))
                output_filename = face_restoration.restore_face(
                    input_image_path=file_path,
                    fidelity_weight=fidelity_weight,
                    output_dir=app.config["OUTPUT_FOLDER"]
                )
            elif "inpaint" in request.form:
                output_filename = face_inpainting.inpaint_face(
                    input_image_path=file_path,
                    output_dir=app.config["OUTPUT_FOLDER"]
                )
            else:
                output_filename = None

            if output_filename:
                # Processed images are saved in static/outputs.
                output_image_url = url_for("static", filename="outputs/" + output_filename)
    return render_template("index.html", active_tab="restoration", output_image=output_image_url)

# ------------------------------
# Route for Face Comparison.
# ------------------------------
@app.route("/compare", methods=["GET", "POST"])
def compare():
    comparison_result = None
    comp_source_url = None
    comp_target_url = None
    if request.method == "POST":
        # Ensure both files are present.
        if "source_image" not in request.files or "target_image" not in request.files:
            return redirect(request.url)
        source_file = request.files["source_image"]
        target_file = request.files["target_image"]
        if source_file.filename == "" or target_file.filename == "":
            return redirect(request.url)
        if source_file and allowed_file(source_file.filename) and target_file and allowed_file(target_file.filename):
            # Save source image.
            src_filename = secure_filename(source_file.filename)
            src_unique = f"{uuid.uuid4().hex}_{src_filename}"
            src_path = os.path.join(app.config["UPLOAD_FOLDER"], src_unique)
            source_file.save(src_path)
            # Save target image.
            tgt_filename = secure_filename(target_file.filename)
            tgt_unique = f"{uuid.uuid4().hex}_{tgt_filename}"
            tgt_path = os.path.join(app.config["UPLOAD_FOLDER"], tgt_unique)
            target_file.save(tgt_path)
            
            # Perform face comparison with a threshold of 65%.
            similarity_percent, verdict = face_comparison.compare_faces(src_path, tgt_path, threshold=65)
            if similarity_percent is None:
                comparison_result = verdict  # In case of error.
            else:
                comparison_result = f"Similarity: {similarity_percent}%. Verdict: {verdict}"
            
            # Use the uploaded_file route to generate URLs for the image previews.
            comp_source_url = url_for("uploaded_file", filename=os.path.basename(src_path))
            comp_target_url = url_for("uploaded_file", filename=os.path.basename(tgt_path))
    return render_template("compare.html", active_tab="comparison",
                           comparison_result=comparison_result,
                           comp_source=comp_source_url,
                           comp_target=comp_target_url)

import threading
from queue import Queue

def stream_output(pipe, queue):
    """Helper function to stream output from a pipe to a queue"""
    for line in iter(pipe.readline, ''):  # No need for `b''`
        line_str = line.rstrip()
        print(line_str, flush=True)  # Print to terminal immediately
        queue.put(line_str)
    pipe.close()
@app.route('/start_processing', methods=['POST'])
def start_processing():
    # Accept multiple video files
    video_files = request.files.getlist('video')
    reference_files = request.files.getlist('references')
    if not video_files:
        return "No video file provided", 400

    # Save video files and build the processing queue.
    video_jobs = []
    for vf in video_files:
        if vf:
            video_filename = secure_filename(vf.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            vf.save(video_path)
            video_jobs.append({'filename': video_filename, 'path': video_path})

    # Get threshold value (default 0.6)
    try:
        threshold = float(request.form.get('threshold', 0.6))
    except ValueError:
        threshold = 0.6

    # Process each reference image
    detectable_refs = []
    undetectable_refs = []
    reference_images = []
    for ref in reference_files:
        if ref:
            ref_filename = secure_filename(ref.filename)
            ref_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_filename)
            ref.save(ref_path)
            img = cv2.imread(ref_path)
            if img is None:
                continue
            try:
                faces = face_rec_model.get(img)
            except Exception:
                faces = face_rec_model.get(pad_image(img))
            base64_img = image_to_base64(img)
            if faces:
                detectable_refs.append({'filename': ref_filename, 'image_base64': base64_img, 'path': ref_path})
                reference_images.append({'filename': ref_filename, 'path': ref_path})
            else:
                undetectable_refs.append({'filename': ref_filename, 'image_base64': base64_img})

    # Compute embeddings for detectable reference images once
    ref_embeddings = {}
    for ref in reference_images:
        img = cv2.imread(ref['path'])
        try:
            faces = face_rec_model.get(img)
        except Exception:
            faces = face_rec_model.get(pad_image(img))
        if faces:
            face = faces[0]
            ref_embeddings[ref['filename']] = {'embedding': face.embedding, 'image': img}

    # Create a unique job ID and initialize job status
    job_id = uuid.uuid4().hex
    jobs[job_id] = {
        'progress': 0,
        'results': [],
        'current_video': None,
        'total_videos': 0,
        'processed_videos': 0,
        'video_status': [],
        'face_tiles': []
    }
    # Start processing videos sequentially in a background thread
    thread = threading.Thread(target=process_video_queue, args=(job_id, video_jobs, ref_embeddings, threshold))
    thread.start()
    return jsonify({'job_id': job_id, 'detectable': detectable_refs, 'undetectable': undetectable_refs})

@app.route('/progress/<job_id>')
def progress(job_id):
    if job_id in jobs:
        return jsonify({
            'progress': jobs[job_id]['progress'],
            'current_video': jobs[job_id]['current_video'],
            'video_status': jobs[job_id]['video_status'],
            'face_tiles': jobs[job_id]['face_tiles']
        })
    else:
        return jsonify({'progress': 0, 'current_video': None, 'video_status': [], 'face_tiles': []})

@app.route('/result/<job_id>')
def result(job_id):
    if job_id in jobs and jobs[job_id]['results'] is not None:
        results = jobs[job_id]['results']
        return render_template('results_fragment.html', results=results)
    else:
        return "Processing not complete", 202

@app.route('/face_recognition')
def face_recognition():
    return render_template('face_recognition.html', active_tab='face_recognition')

def process_single_video(job_id, video_path, ref_embeddings, threshold, video_filename, status_item):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            status_item['status'] = 'failed'
            return []
            
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps) if fps and fps > 0 else 1
        video_results = []
        frame_number = 0
        faces_detected = 0
        
        print(f"Processing video: {video_filename} (Total frames: {total_frames}, FPS: {fps})")
    except Exception as e:
        print(f"Error initializing video processing: {e}")
        status_item['status'] = 'failed'
        return []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if total_frames and total_frames > 0:
            status_item['progress'] = int((frame_number / total_frames) * 100)

        if frame_number % frame_interval == 0:
            time_sec = frame_number / fps if fps and fps > 0 else frame_number
            try:
                faces = face_rec_model.get(frame)
            except Exception as e:
                print(f"Error detecting faces in frame {frame_number}: {e}")
                try:
                    faces = face_rec_model.get(pad_image(frame))
                except Exception as e2:
                    print(f"Error with padded frame: {e2}")
                    faces = []
            
            # Add ALL detected faces to face_tiles (not just matched ones)
            for face in faces:
                faces_detected += 1
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                new_x1 = max(x1 - 10, 0)
                new_y1 = max(y1 - 10, 0)
                new_x2 = min(x2 + 20, frame.shape[1])
                new_y2 = min(y2 + 20, frame.shape[0])
                cropped_face = frame[new_y1:new_y2, new_x1:new_x2]
                
                # Check similarity with reference images
                best_similarity = 0
                best_ref_filename = None
                video_embedding = face.embedding
                
                for ref_filename, data in ref_embeddings.items():
                    sim = cosine_similarity(video_embedding, data['embedding'])
                    if sim > best_similarity:
                        best_similarity = sim
                        best_ref_filename = ref_filename
                
                # Add face tile with similarity info (if any match found)
                face_tile_data = {
                    'time': float(round(time_sec, 2)),
                    'video_filename': video_filename,
                    'video_face': image_to_base64(cropped_face),
                    'similarity': float(round(best_similarity, 3)) if best_similarity > 0 else None,
                    'ref_filename': best_ref_filename
                }
                jobs[job_id]['face_tiles'].append(face_tile_data)
                
                # Add to video_results only if similarity exceeds threshold
                if best_similarity > threshold:
                    video_results.append({
                        'time': float(round(time_sec, 2)),
                        'video_face': image_to_base64(cropped_face),
                        'ref_filename': best_ref_filename,
                        'ref_face': image_to_base64(ref_embeddings[best_ref_filename]['image']),
                        'similarity': float(round(best_similarity * 100, 2)),
                        'video_filename': video_filename
                    })
        frame_number += 1

    cap.release()
    status_item['progress'] = 100
    status_item['status'] = 'completed'
    status_item['faces_detected'] = faces_detected
    print(f"Completed processing {video_filename}: {faces_detected} faces detected, {len(video_results)} matches found")
    return video_results

def process_video_queue(job_id, video_jobs, ref_embeddings, threshold):
    try:
        total = len(video_jobs)
        jobs[job_id]['total_videos'] = total
        jobs[job_id]['processed_videos'] = 0
        jobs[job_id]['results'] = []
        jobs[job_id]['face_tiles'] = []
        jobs[job_id]['video_status'] = []

        print(f"Starting processing queue with {total} videos, threshold: {threshold}")

        for video in video_jobs:
            status_item = {
                'filename': video['filename'],
                'path': video['path'],
                'progress': 0,
                'status': 'processing'
            }
            jobs[job_id]['video_status'].append(status_item)
            jobs[job_id]['current_video'] = video['filename']
            try:
                video_results = process_single_video(job_id, video['path'], ref_embeddings, threshold, video['filename'], status_item)
                jobs[job_id]['results'].extend(video_results)
            except Exception as e:
                print(f"Error processing video {video['filename']}: {e}")
                status_item['progress'] = 0
                status_item['status'] = 'failed'
            jobs[job_id]['processed_videos'] += 1
            overall_progress = int((jobs[job_id]['processed_videos'] / total) * 100) if total > 0 else 100
            jobs[job_id]['progress'] = overall_progress

        jobs[job_id]['progress'] = 100
        jobs[job_id]['current_video'] = None
        print(f"Completed all video processing for job {job_id}")
    except Exception as e:
        print(f"Error in video queue processing: {e}")
        jobs[job_id]['progress'] = 0
        jobs[job_id]['status'] = 'failed'

@app.route('/video_enhancement', methods=['GET', 'POST'])
def video_enhancement():
    enhanced_video_url = None
    error_message = None
    terminal_output = []
    
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return redirect(request.url)
            
        video_file = request.files['video_file']
        
        if video_file.filename == '':
            return redirect(request.url)
            
        if video_file and allowed_video(video_file.filename):
            try:
                # Save the uploaded video
                filename = secure_filename(video_file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                video_file.save(video_path)
                
                # Create a unique results directory for this enhancement
                video_name = os.path.splitext(unique_filename)[0]
                results_dir = f'results/{video_name}_1.0'
                
                # Run CodeFormer command
                command = [
                    'python', '-u',
                    'inference_codeformer.py',
                    '--input_path',
                    video_path,
                    '--bg_upsampler',
                    'realesrgan',
                    '--face_upsample',
                    '-w',
                    '1.0'
                ]
                
                # Create queues for stdout and stderr
                stdout_queue = Queue()
                stderr_queue = Queue()
                
                # Log basic debug info
                try:
                    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
                    print(f"[VideoEnhancement] Starting process for: {video_path} ({file_size_mb:.2f} MB)")
                    print(f"[VideoEnhancement] Command: {' '.join(command)}")
                except Exception as e:
                    print(f"[VideoEnhancement] Debug preflight failed: {e}")

                # Start the subprocess with pipes (unbuffered python for immediate logs)
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Create threads to handle output streams
                stdout_thread = threading.Thread(
                    target=stream_output, 
                    args=(process.stdout, stdout_queue)
                )
                stderr_thread = threading.Thread(
                    target=stream_output, 
                    args=(process.stderr, stderr_queue)
                )
                
                # Start the threads
                stdout_thread.start()
                stderr_thread.start()
                
                # Wait for the process to complete with a timeout safeguard
                try:
                    # 20 minutes timeout safeguard for very long videos
                    process.wait(timeout=1200)
                except subprocess.TimeoutExpired:
                    error_message = "Video enhancement timed out. Please try a shorter video or lower resolution."
                    print("[VideoEnhancement] Timeout reached. Terminating process.")
                    try:
                        process.kill()
                    except Exception:
                        pass
                
                # Wait for output threads to complete
                stdout_thread.join()
                stderr_thread.join()
                
                # Collect all output
                while not stdout_queue.empty():
                    terminal_output.append(stdout_queue.get())
                while not stderr_queue.empty():
                    terminal_output.append(stderr_queue.get())
                
                if process.returncode == 0:
                    # Get the enhanced video path from results directory
                    enhanced_video_name = f'{video_name}.mp4'
                    enhanced_video_path = os.path.join(results_dir, enhanced_video_name)
                    
                    if os.path.exists(enhanced_video_path):
                        # Create final output path
                        final_output_path = os.path.join(app.config['OUTPUT_FOLDER'], enhanced_video_name)
                        
                        # Move the enhanced video to our static output directory
                        shutil.move(enhanced_video_path, final_output_path)
                        
                        # Create URL for the enhanced video
                        enhanced_video_url = url_for('static', filename=f'outputs/{enhanced_video_name}')
                        
                        # Clean up the results directory
                        if os.path.exists(results_dir):
                            shutil.rmtree(results_dir)
                    else:
                        error_message = "Enhanced video file not found in results directory"
                else:
                    error_message = f"Error during video enhancement: Process returned {process.returncode}"
                    
            except Exception as e:
                error_message = f"Error processing video: {str(e)}"
            finally:
                # Clean up the uploaded file
                try:
                    if os.path.exists(video_path):
                        os.remove(video_path)
                except Exception as e:
                    print(f"Error cleaning up uploaded file: {str(e)}")
                
    return render_template(
        'video_enhancement.html',
        active_tab="video_enhancement",
        enhanced_video=enhanced_video_url,
        error_message=error_message,
        terminal_output='\n'.join(terminal_output)
    )


@app.route('/inpainting_drawing', methods=['GET', 'POST'])
def inpainting_drawing():
    output_image_url = None
    if request.method == "POST":
        if "face_image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
            
        file = request.files["face_image"]
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
            
        if file and allowed_file(file.filename):
            try:
                # Generate unique filename for the uploaded image
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4().hex}_{filename}"
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
                
                # Save the drawn image from canvas
                file.save(file_path)
                
                # Process inpainting
                output_filename = face_inpainting.inpaint_face(
                    input_image_path=file_path,
                    output_dir=app.config["OUTPUT_FOLDER"]
                )
                
                if output_filename:
                    # Return URL for the processed image
                    output_image_url = url_for("static", filename=f"outputs/{output_filename}")
                    print("Hello World from inpaint")
                    return jsonify({
                        "success": True,
                        "output_image": output_image_url
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": "Failed to process image"
                    }), 500
                    
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500
                
    # GET request - render the template
    return render_template('inpainting.html', active_tab='inpainting_drawing', output_image=output_image_url)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'face_image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['face_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], file.filename)
        file.save(input_path)

        # Delete all other files from UPLOAD_FOLDER except directories and the uploaded file
        for existing_file in os.listdir(app.config['UPLOAD_FOLDER']):
            existing_file_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_file)
            if os.path.isfile(existing_file_path) and existing_file != file.filename:
                os.remove(existing_file_path)
        
        # Run the Python script with the correct paths
        print("Input path:", input_path)
        subprocess.run(['python', 'scripts/crop_align_face.py', '-i', app.config['UPLOAD_FOLDER'], '-o', app.config['OUTPUT_FOLDER']])
        
        # Generate the URL for the processed image
        output_image_url = url_for('static', filename=f'outputs/{file.filename}')
        
        return jsonify({'success': True, 'output_image': output_image_url})

    return jsonify({'error': 'An error occurred during file processing'})

if __name__ == "__main__":
    # Run the app on host 192.168.0.179 at port 5000.
    app.run(host="0.0.0.0", port=5030, debug=True)
