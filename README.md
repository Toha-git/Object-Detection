# Object Detection with Python and OpenCV

Production-ready object detection starter project supporting webcam streams, video files, still images, and a Flask REST API.

## Features

- YOLOv8 via `ultralytics` as the primary detector
- SSD MobileNet V2 through `cv2.dnn` as an alternative detector
- Haar Cascade face detection as a lightweight fallback
- Annotated bounding boxes, stable per-class colors, labels, confidence, FPS, backend, and object count
- Optional annotated output saving
- Optional CSV detection logging
- Flask `POST /detect` endpoint for uploaded images
- Custom YOLOv8 fine-tuning workflow for your own datasets
- SORT-style object tracking with persistent IDs and 30-point motion trails
- Streamlit dashboard for live video, charts, and recent detections
- Desktop alerts when a configured class is detected
- YAML configuration and `.env` overrides
- Class filtering, ROI detection, and frame skipping
- Model export and performance benchmarking tools
- Docker, Compose, linting, formatting, type checking, and tests

## Setup

```bash
cd object_detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd object_detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## CLI Usage

The CLI reads defaults from `config.yaml`, then applies command-line overrides.

Webcam:

```bash
python main.py --source webcam --backend yolo --conf 0.5
```

Webcam with custom YOLO weights:

```bash
python main.py --source webcam --backend yolo --weights runs/train/exp/weights/best.pt
```

Webcam with tracking:

```bash
python main.py --source webcam --backend yolo --track
```

Desktop alert when a person is detected:

```bash
python main.py --source webcam --backend yolo --alert person
```

Image:

```bash
python main.py --source path/to/image.jpg --backend yolo --save --log
```

Video:

```bash
python main.py --source path/to/video.mp4 --backend yolo --save --headless
```

Class filtering, ROI, and frame skipping:

```bash
python main.py \
  --source path/to/video.mp4 \
  --backend yolo \
  --classes person,car \
  --roi 100,100,900,700 \
  --frame-skip 3 \
  --track
```

Arguments:

- `--config`: path to a YAML config file
- `--source`: `webcam`, image path, or video path
- `--backend`: `yolo`, `ssd`, or `haar`
- `--conf`: confidence threshold from `0` to `1`
- `--weights`: YOLO `.pt` weights path or model name, used only with `--backend yolo`
- `--track`: enable SORT-style object tracking for webcam/video streams
- `--alert`: trigger a desktop notification when a class is detected, for example `--alert person`
- `--alert-cooldown`: seconds between repeated alerts
- `--classes`: comma-separated labels to keep, for example `person,car`
- `--roi`: optional region of interest as `x1,y1,x2,y2`
- `--frame-skip`: run detection every N frames and reuse tracked detections between runs
- `--save`: save annotated output to `output/`
- `--log`: write CSV detections to `output/`
- `--headless`: run without GUI display

Press `q` to stop webcam or video processing when GUI mode is enabled.

## Configuration

Edit `config.yaml` to avoid repeating CLI flags:

```yaml
source: webcam
backend: yolo
confidence: 0.5
weights: yolov8n.pt
save: false
log: false
headless: false
tracking:
  enabled: false
  iou_threshold: 0.3
  max_missed: 10
alert:
  enabled: false
  class: person
  cooldown_seconds: 10
detection:
  classes: []
  frame_skip: 1
  roi: null
output:
  directory: output
```

Environment variables from `.env` are also supported. Start from:

```bash
cp .env.example .env
```

## Object Tracking

Enable tracking with `--track`:

```bash
python main.py --source webcam --backend yolo --track
```

The tracker assigns each object a persistent numeric ID across frames. Labels are drawn with the track ID, and each tracked object displays a trailing path line for its last 30 center positions.

Tracking also works with video files:

```bash
python main.py --source path/to/video.mp4 --backend yolo --track --save
```

## Desktop Alerts

Use `--alert` to send a desktop notification when a specific class appears:

```bash
python main.py --source webcam --backend yolo --alert person
```

The default alert cooldown is 10 seconds. The alert helper lives in `utils/alert.py` and can be configured with:

```python
DetectionAlert(alert_class="person", cooldown_seconds=10)
```

## Streamlit Dashboard

Run the dashboard from the project root:

```bash
streamlit run dashboard/app.py
```

Dashboard features:

- Live video panel from webcam, uploaded image, or uploaded video
- Real-time bar chart for detection counts per class, updated every second
- Rolling 60-second line chart for total objects detected over time
- Sidebar backend selector
- Sidebar confidence slider
- Sidebar tracking toggle
- Sidebar class filter, alert settings, camera index, and model selector
- Detection log table showing the last 20 detections

Use the sidebar `Run` toggle to start processing. Uploading a video switches the source from webcam to the uploaded file.

## Model Export and Benchmarking

Export YOLO weights:

```bash
python tools/export_yolo.py --weights runs/train/exp/weights/best.pt --format onnx
python tools/export_yolo.py --weights runs/train/exp/weights/best.pt --format openvino
```

Benchmark an image or video:

```bash
python tools/benchmark.py --source path/to/image.jpg --backend yolo --weights yolov8n.pt
python tools/benchmark.py --source path/to/video.mp4 --backend yolo --frames 120
```

## Training a Custom YOLOv8 Model

### 1. Collect and Label Images

Collect images that represent the real environment where the detector will run. Include different lighting, camera angles, object sizes, backgrounds, and partial occlusions.

Recommended labelling tools:

- Roboflow: upload images, annotate boxes, export as YOLOv8 format.
- LabelImg: draw bounding boxes locally and save labels as YOLO text files.

YOLO label files must have the same base filename as each image:

```text
raw_data/
├── images/
│   ├── frame001.jpg
│   └── frame002.jpg
└── labels/
    ├── frame001.txt
    └── frame002.txt
```

Each label row should use YOLO normalized coordinates:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.512 0.438 0.220 0.410
```

### 2. Prepare the Dataset

Run the preparation script from the project root:

```bash
python train/prepare_dataset.py \
  --images raw_data/images \
  --labels raw_data/labels \
  --output dataset_yolo \
  --val-ratio 0.2
```

This creates:

```text
dataset_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

If you intentionally have negative images with no objects, include them with empty label files:

```bash
python train/prepare_dataset.py \
  --images raw_data/images \
  --labels raw_data/labels \
  --output dataset_yolo \
  --allow-empty-labels
```

### 3. Configure `dataset.yaml`

Edit `train/dataset.yaml` for your classes and dataset path. The included template is a 3-class example:

```yaml
path: dataset_yolo
train: images/train
val: images/val

nc: 3
names:
  - person
  - car
  - dog
```

For your own dataset, set `nc` to the number of classes and list class names in the same order used by your label files.

### 4. Train YOLOv8

Fine-tune `yolov8n.pt` with configurable epochs, image size, and batch size:

```bash
python train/train.py \
  --data train/dataset.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

Useful options:

- `--epochs`: number of passes through the dataset
- `--imgsz`: training image size, commonly `640`
- `--batch`: batch size, lower this if you run out of memory
- `--device`: use `cpu`, `0`, or `0,1`
- `--project`: output directory, default `runs/train`
- `--name`: run name, default `exp`

The best trained weights are saved at:

```text
runs/train/exp/weights/best.pt
```

### 5. Use the Trained Model

Swap the trained model into the existing detector with `--weights`:

```bash
python main.py \
  --source webcam \
  --backend yolo \
  --weights runs/train/exp/weights/best.pt
```

You can also run custom weights on images or videos:

```bash
python main.py \
  --source path/to/video.mp4 \
  --backend yolo \
  --weights runs/train/exp/weights/best.pt \
  --save
```

## SSD MobileNet V2 Backend

The SSD backend uses external TensorFlow model files through OpenCV DNN. You can provide files in either of these ways:

1. Put these files in `object_detection/models/`:
   - `frozen_inference_graph.pb`
   - `ssd_mobilenet_v2_coco.pbtxt`

2. Set environment variables:

```bash
export SSD_MODEL_PATH=/path/to/frozen_inference_graph.pb
export SSD_CONFIG_PATH=/path/to/ssd_mobilenet_v2_coco.pbtxt
```

3. Allow the project to download them at runtime:

```bash
SSD_AUTO_DOWNLOAD=1 python main.py --source webcam --backend ssd
```

## Flask API

Start the server:

```bash
python api/server.py
```

Health check:

```bash
curl http://localhost:5000/health
```

Detect objects:

```bash
curl -X POST http://localhost:5000/detect \
  -F image=@path/to/image.jpg \
  -F backend=yolo \
  -F conf=0.5 \
  -F classes=person,car
```

Response:

```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "box": [12, 34, 320, 480]
    }
  ],
  "count": 1,
  "backend": "yolo"
}
```

The API enforces an upload size limit with `MAX_UPLOAD_MB`, defaulting to `16`.

## Quality Checks

Run tests:

```bash
python -m pytest
```

Run linting:

```bash
python -m ruff check .
```

Format code:

```bash
python -m black .
```

Run type checks:

```bash
python -m mypy .
```

## Docker

Build and run the API:

```bash
docker compose up api
```

Run the dashboard:

```bash
docker compose up dashboard
```

Webcam access inside Docker needs host-specific device forwarding. Uploaded images and videos are the most portable Docker workflow.

## macOS Camera Troubleshooting

If webcam access fails on macOS:

```bash
tccutil reset Camera
```

Then run the dashboard or this probe from the same app that should receive permission:

```bash
python3 -c "import cv2, time; cap=cv2.VideoCapture(0); time.sleep(2); print(cap.isOpened()); cap.release()"
```

Grant permission in `System Settings > Privacy & Security > Camera`, then restart Streamlit.

## Output

The project auto-creates `output/` when saving annotated media or detection logs.

CSV columns:

```text
timestamp, frame_index, label, confidence, x1, y1, x2, y2
```

## Project Structure

```text
object_detection/
├── main.py
├── detector/
│   ├── __init__.py
│   ├── base.py
│   ├── yolo.py
│   ├── ssd.py
│   └── haar.py
├── utils/
│   ├── __init__.py
│   ├── alert.py
│   ├── config.py
│   ├── detection.py
│   ├── draw.py
│   ├── preprocess.py
│   ├── logger.py
│   └── serialization.py
├── api/
│   └── server.py
├── dashboard/
│   └── app.py
├── train/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── dataset.yaml
├── tracker/
│   ├── __init__.py
│   └── sort_tracker.py
├── tools/
│   ├── benchmark.py
│   └── export_yolo.py
├── tests/
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── config.yaml
└── README.md
```

## Notes

- YOLOv8 downloads `yolov8n.pt` on first use if it is not already cached.
- Use `--headless` for servers, CI, and environments without a display.
- Haar Cascade detects faces, not general COCO objects, and is intended as a lightweight fallback.
