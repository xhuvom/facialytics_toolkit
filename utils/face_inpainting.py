# utils/face_inpainting.py
import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from utils.models import INPAINTING_MODEL, device  # Import the pre-loaded model

def inpaint_face(input_image_path, output_dir="results"):
    """
    Process a single face image for inpainting.
    
    :param input_image_path: Path to the input face image.
    :param output_dir: Directory to save the output.
    :return: Output filename.
    """
    print(f"[Inpainting] Processing image: {input_image_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    net = INPAINTING_MODEL
    
    basename = os.path.splitext(os.path.basename(input_image_path))[0]
    input_face = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if input_face is None:
        print("[Inpainting] ERROR: Could not read image:", input_image_path)
        return None
    input_face = cv2.resize(input_face, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    input_tensor = img2tensor(input_face / 255.0, bgr2rgb=True, float32=True)
    normalize(input_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    try:
        with torch.no_grad():
            # Create a mask based on pixel values
            mask = torch.zeros(512, 512)
            m_ind = torch.sum(input_tensor[0], dim=0)
            mask[m_ind == 3] = 1.0
            mask = mask.view(1, 1, 512, 512).to(device)
            output_face = net(input_tensor, w=1, adain=False)[0]
            output_face = (1 - mask) * input_tensor + mask * output_face
            save_face = tensor2img(output_face, rgb2bgr=True, min_max=(-1, 1))
        torch.cuda.empty_cache()
    except Exception as error:
        print(f"[Inpainting] ERROR during inference: {error}")
        save_face = tensor2img(input_tensor, rgb2bgr=True, min_max=(-1, 1))
    
    save_face = save_face.astype("uint8")
    output_filename = f"{basename}_inpainted.png"
    output_path = os.path.join(output_dir, output_filename)
    imwrite(save_face, output_path)
    print(f"[Inpainting] Output saved to: {output_path}")
    return output_filename

