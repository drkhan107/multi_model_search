from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from transformers import AutoProcessor, CLIPTextModel, CLIPVisionModelWithProjection

# Load the CLIP processor, text model, and vision model
PROCESSOR = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
TEXT_MODEL = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cpu")
VISION_MODEL = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14",torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cpu")

def generate_embeddings(text: Optional[str] = None, image: Optional[JpegImageFile] = None) -> np.ndarray:
    """
    Generate embeddings from text or image using the CLIP model.
    Args:
        text (Optional[str]): customer input text
        image (Optional[Image]): customer input image
    Returns:
        np.ndarray: embedding vector
    """
    if text:
        inputs = PROCESSOR(text=text, return_tensors="pt").to("cpu")
        outputs = TEXT_MODEL(**inputs).last_hidden_state
        # Use the pooled output for text embedding
        embedding = F.normalize(outputs[:, 0, :], p=2, dim=-1).detach().numpy().flatten()
    elif image:
        inputs = PROCESSOR(images=image, return_tensors="pt").to("cpu")
        outputs = VISION_MODEL(**inputs).image_embeds
        embedding = F.normalize(outputs, p=2, dim=-1).mean(dim=0).detach().numpy().flatten()
    else:
        raise ValueError("Either text or image must be provided.")

    return embedding