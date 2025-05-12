"""
OmniParser inference module.

This module provides a function for analyzing UI elements in an image using OmniParser.
It detects UI elements, recognizes text, and generates descriptions for non-text elements.
"""

import base64
import io
from typing import Tuple, List, Dict, Union, Any
from PIL import Image
import torch
import numpy as np

from action.omniparser.util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_yolo_model,
    get_caption_model_processor,
)


def analyze_ui_image(
    image: Union[str, Image.Image],
    model_path: str = "weights/icon_detect/model.pt",
    caption_model: str = "florence2",
    caption_model_path: str = "weights/icon_caption_florence",
    box_threshold: float = 0.05,
    iou_threshold: float = 0.7,
    use_paddleocr: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Analyzes UI elements in an image and returns a labeled image with element details.

    Args:
        image: Path to an image file or a PIL Image object
        model_path: Path to the YOLO UI detection model
        caption_model: Model type for element captioning ("florence2" or "blip2")
        caption_model_path: Path to the caption model
        box_threshold: Confidence threshold for UI element detection
        iou_threshold: IOU threshold for overlapping box removal
        use_paddleocr: Whether to use PaddleOCR (True) or EasyOCR (False)
        device: Device to run models on ('cuda' or 'cpu')

    Returns:
        Tuple containing:
            - PIL Image with labeled UI elements
            - List of dictionaries with details about each UI element
    """
    # Load models
    som_model = get_yolo_model(model_path)
    som_model.to(device)

    caption_model_processor = get_caption_model_processor(
        model_name=caption_model, model_name_or_path=caption_model_path, device=device
    )

    # Load image if path is provided
    if isinstance(image, str):
        image = Image.open(image)
    image_rgb = image.convert("RGB")

    # Calculate box overlay ratio for proper display scaling
    box_overlay_ratio = max(image_rgb.size) / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }

    # Run OCR to detect text elements
    ocr_bbox_rslt, _ = check_ocr_box(
        image_rgb,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=use_paddleocr,
    )
    text, ocr_bbox = ocr_bbox_rslt

    # Detect UI elements and generate labels
    labeled_img_b64, _, parsed_content_list = get_som_labeled_img(
        image_rgb,
        som_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        use_local_semantics=True,
        iou_threshold=iou_threshold,
        scale_img=False,
        batch_size=128,
    )

    # Convert base64 encoded image back to PIL Image
    labeled_image = Image.open(io.BytesIO(base64.b64decode(labeled_img_b64)))

    # Clean up models to free memory
    if device == "cuda":
        torch.cuda.empty_cache()

    for i, _ in enumerate(parsed_content_list):
        parsed_content_list[i]["id"] = i

    return labeled_image, parsed_content_list
