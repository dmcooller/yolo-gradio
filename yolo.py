from io import BytesIO
import os
import time
import uuid
import subprocess

import PIL.Image as Image
import cv2

from ultralytics import YOLO

from helpers.device import get_device_type


MODELS_FOLDER = "models"


def default_models() -> list[str]:
    return ["yolo11n", "yolo11s"]

def read_available_models() -> list[str]:
    # names without extensions
    return [f.split(".")[0] for f in os.listdir(MODELS_FOLDER) if f.endswith(".pt")]

def get_all_models() -> list[str]:
    return sorted(list(set(default_models() + read_available_models())))

def get_model_path(model_name: str) -> str:
    return os.path.join(MODELS_FOLDER, f"{model_name}.pt")


def img_inf(
    img,
    conf_threshold: float, iou_threshold: float,
    model_n,
    model_name_manual: str | None = None, # model name entered manually (for gradio)
    imgsz: int = 640,
    device: str = "auto",
):
    model = YOLO(get_model_path(model_name_manual or model_n))
    names = model.names
    device = get_device_type(device)

    start_time = time.time()

    results = model.predict(
        device=device,
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=imgsz,
    )
    end_time = time.time()
    execution_time = end_time - start_time

    objects = {}
    res = results[0]
    im_array = res.plot()
    im = Image.fromarray(im_array[..., ::-1])

    clist= res.boxes.cls # Class
    conflist = res.boxes.conf.tolist() # Confidence
    xyxylist = res.boxes.xyxy # Bounding box coordinates

    clist_len = len(clist)
    for i in range(clist_len):
        # Reverse the order of the objects so that the worst confidence is first
        objects[str(clist_len-i)] = {
            "name": names[int(clist[i])],
            "confidence": round(conflist[i], 2),
            "box": xyxylist[i].tolist(),
        }
   
    objects_count = len(objects)
    avg_confidence = round(sum(conflist) / len(conflist), 2) if conflist else 0
    return im, round(execution_time, 4), objects_count, avg_confidence, objects


def vid_inf(
    vid,
    conf_threshold: float, iou_threshold: float,
    model_n,
    model_name_manual: str | None = None, # model name entered manually (for gradio)
    imgsz: int = 640,
    device: str = "auto",
):
    cap = cv2.VideoCapture(vid)
    # get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    if not os.path.exists("output"):
        os.makedirs("output")
    output_video = os.path.join("output", f"out_{model_n}_{str(uuid.uuid4())}.mp4")

    # Output video
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    model = YOLO(get_model_path(model_name_manual or model_n))
    device = get_device_type(device)

    # Initialize frame counter and timing
    frame_count = 0
    fps_counter = 0.0
    exec_start_time = time.time()
    fps_start_time = time.time()

    # Create a placeholder for the last frame
    last_frame = None

    try:
        while cap.isOpened():
            ret, frame_y = cap.read()
            if not ret:
                break

            results = model.predict(
                device=device,
                source=frame_y,
                conf=conf_threshold,
                iou=iou_threshold,
                show_labels=True,
                show_conf=True,
                imgsz=imgsz,
                verbose=False,
            )

            res = results[0]
            im_array = res.plot()
            out.write(im_array) # save the frame to the output video

            # Store the last frame to show after processing is complete
            last_frame = Image.fromarray(im_array[..., ::-1])

            # Update FPS every second
            frame_count += 1
            if frame_count % 10 == 0:
                end_time = time.time()
                fps_counter = 10 / (end_time - fps_start_time)
                fps_start_time = time.time()

            # During processing, yield updates
            yield last_frame, round(fps_counter, 2), None, None
    finally:
        # Ensure resources are released even if an error occurs
        end_time = time.time()
        execution_time = end_time - exec_start_time
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Return the final results - note we're not using yield for the final result
    yield last_frame, round(fps_counter, 2), round(execution_time, 4), "./output/out_yolo11n_0d3ca480-dd4a-47b2-a014-25041bce1d28.mp4"


def _capture_screenshot() -> Image:
    "Capture a screenshot using ADB and return it as a PIL Image."
    proc = subprocess.Popen('adb exec-out screencap -p', shell=True, stdout=subprocess.PIPE)
    image_bytes = proc.stdout.read()
    image = Image.open(BytesIO(image_bytes))
    # Another way to read the image
    #image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return image


def inf_adb(
    conf_threshold: float, iou_threshold: float,
    model_n,
    model_name_manual: str | None = None, # model name entered manually (for gradio)
    imgsz: int = 640,
    device: str = "auto",
):
    while True:
        start_time = time.time()
        img = _capture_screenshot()
        result = img_inf(img, conf_threshold, iou_threshold, model_n, model_name_manual, imgsz, device)
        end_time = time.time()
        execution_time = end_time - start_time
        # Image, Inference Time, Execution Time, Objects Count, Average Confidence, Objects Detected
        yield result[0], result[1], round(execution_time, 4), result[2], result[3], result[4]


def ws_inf(
    ws_host: str,
    ws_port: int,
    conf_threshold: float, iou_threshold: float,
    model_n,
    model_name_manual: str | None = None, # model name entered manually (for gradio)
    imgsz: int = 640,
    device: str = "auto",
):
    """Websockets inference """
    from av.codec import CodecContext
    from websockets.sync.client import connect
    codec = CodecContext.create("h264", "r")
    with connect(f"ws://{ws_host}:{ws_port}") as websocket:
        weight, height = _parse_resolution_packet(websocket)
        while True:
            data = websocket.recv()
            if data == b"":
                raise ConnectionError("Video stream is disconnected")
            # First byte = 0 means video stream
            if data[0] == 0:
                packets = codec.parse(data[1:])
                for packet in packets:
                    frames = codec.decode(packet)
                    for frame in frames:
                        frame = frame.to_ndarray(format="bgr24")
                        result = img_inf(frame, conf_threshold, iou_threshold, model_n, model_name_manual, imgsz, device)
                        # Image, Inference Time, Execution Time, Objects Count, Average Confidence, Objects Detected
                        yield result[0], result[1], None, result[2], result[3], result[4]


def _parse_resolution_packet(websocket) -> tuple[int, int]:
    first_packet = websocket.recv()
    if first_packet == b"":
        raise ConnectionError("Video stream is disconnected")
    w = int.from_bytes(first_packet[:4], byteorder="big")
    h = int.from_bytes(first_packet[4:8], byteorder="big")
    return w, h
            