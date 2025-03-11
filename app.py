import logging
import gradio as gr
from ultralytics import ASSETS, checks

logging.basicConfig(level=logging.INFO)

from yolo import img_inf, get_all_models, inf_adb, vid_inf, ws_inf

DEFAULT_MODEL = "yolo11n"

def create_interface_elements():
    return {
        "model_name_manual": gr.Textbox(value=None, label="Model Manual", info="Enter the model name manually and download it"),
        "model_name": gr.Radio(get_all_models(), value=DEFAULT_MODEL, label="Model", info="choose your model"),
        "confidence_slider": gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        "iou_slider": gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
        "device": gr.Dropdown(["auto", "cpu", "cuda", "mps"], value="auto", label="Device"),
        "image_size": gr.Number(value=640, label="Image Size"),
        "inference_time": gr.Textbox(label="Inference Time (seconds)"),
        "output": gr.Image(type="pil", label="Output Image"),
    }

elements_image = create_interface_elements()
interface_image = gr.Interface(
    fn=img_inf,
    inputs=[
        gr.Image(type="pil", label="Upload Image", height=720),
        elements_image["confidence_slider"],
        elements_image["iou_slider"],
        elements_image["model_name"],
        elements_image["model_name_manual"],
        elements_image["image_size"],
        elements_image["device"],
    ],
    outputs=[elements_image["output"], elements_image["inference_time"],
            gr.Textbox(label="Objects Count"), gr.Textbox(label="Average Confidence"),
            gr.JSON(label="Objects Detected"),
            ],
    title="Image Inference",
    description="Upload images for inference. The Ultralytics YOLOv11n model is used by default.",
    examples=[
        [ASSETS / "bus.jpg", 0.25, 0.45],
        [ASSETS / "zidane.jpg", 0.25, 0.45],
    ],
    cache_examples=False,
)

elements_video = create_interface_elements()
input_video = gr.Video(sources=None, label="Input Video")
output_video_file = gr.Video(label="Output video")
interface_video = gr.Interface(
    fn=vid_inf,
    inputs=[
        input_video,
        elements_video["confidence_slider"],
        elements_video["iou_slider"],
        elements_video["model_name"],
        elements_video["model_name_manual"],
        elements_video["image_size"],
        elements_video["device"],
    ],
    outputs=[elements_video["output"], gr.Textbox(label="Inference FPS"), elements_video["inference_time"], output_video_file],
    title="Video Inference",
    description="Upload your video and select one model and see the results!",
    examples=[["samples/video_1.mp4", 0.25, 0.45]],
    cache_examples=False,
)


output_video_file = gr.Video(label="Output video")
elements_cam = create_interface_elements()
interface_camera = gr.Interface(
    fn=vid_inf,
    inputs=[
        gr.Textbox(label="Camera URL", placeholder="rtsp://user:pass@ip:port/cam/realmonitor?channel=1&subtype=1"),
        elements_cam["confidence_slider"],
        elements_cam["iou_slider"],
        elements_cam["model_name"],
        elements_cam["model_name_manual"],
        elements_cam["image_size"],
        elements_cam["device"],
    ],
    outputs=[elements_cam["output"], gr.Textbox(label="Inference FPS"), elements_cam["inference_time"], output_video_file],
    title="Camera Inference",
    description="Add your camera URL and select one model and see the results!",
    cache_examples=False,
)

elements_adb = create_interface_elements()
interface_adb = gr.Interface(
    fn=inf_adb,
    inputs=[
        elements_adb["confidence_slider"],
        elements_adb["iou_slider"],
        elements_adb["model_name"],
        elements_adb["model_name_manual"],
        elements_adb["image_size"],
        elements_adb["device"],
    ],
    outputs=[elements_adb["output"],
        elements_adb["inference_time"], gr.Textbox(label="Execution Time (seconds)"),
        gr.Textbox(label="Objects Count"), gr.Textbox(label="Average Confidence"),
        gr.JSON(label="Objects Detected"),
    ],
    title="ADB Inference",
    description="Connect your Android device and select one model and see the results!",
    cache_examples=False,
)

elements_ws = create_interface_elements()
interface_ws = gr.Interface(
    fn=ws_inf,
    inputs=[
        gr.Textbox(label="Websockets Host", placeholder="localhost", value="localhost"),
        gr.Number(label="Websockets Port", minimum=1, maximum=65535, step=1, value=8765),
        elements_ws["confidence_slider"],
        elements_ws["iou_slider"],
        elements_ws["model_name"],
        elements_ws["model_name_manual"],
        elements_ws["image_size"],
        elements_ws["device"],
    ],
    outputs=[elements_ws["output"],
        elements_ws["inference_time"], gr.Textbox(label="Execution Time (seconds)"),
        gr.Textbox(label="Objects Count"), gr.Textbox(label="Average Confidence"),
        gr.JSON(label="Objects Detected"),
    ],
    title="Websockets Inference",
    description="Connect to a websockets server and select one model and see the results!",
    cache_examples=False,
)


if __name__ == '__main__':
    checks()
    gr.TabbedInterface(
        [interface_image, interface_video, interface_camera, interface_adb, interface_ws], \
        tab_names=["Image", "Video", "Camera", "ADB", "Websockets"], title="YOLO Inference App"
    ).queue().launch()