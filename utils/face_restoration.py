# utils/face_restoration.py
import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from utils.models import RESTORATION_MODEL, device  # Import the pre-loaded model

def restore_face(input_image_path, fidelity_weight=0.5, upscale=2, output_dir="results"):
    """
    Process a single image for face enhancement/restoration.
    
    :param input_image_path: Path to the input face image.
    :param fidelity_weight: Float between 0 and 1.
    :param upscale: Upsampling factor.
    :param output_dir: Directory to save the output.
    :return: Output filename.
    """
    print(f"[Restoration] Processing image: {input_image_path}")
    
    # Ensure the output folder exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the global RESTORATION_MODEL (loaded at startup)
    net = RESTORATION_MODEL
    
    # Initialize the face helper (assumes input is not pre-cropped/aligned)
    face_helper = FaceRestoreHelper(
        upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device
    )
    
    basename = os.path.splitext(os.path.basename(input_image_path))[0]
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    if img is None:
        print("[Restoration] ERROR: Could not read image:", input_image_path)
        return None

    face_helper.read_image(img)
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=True, resize=640, eye_dist_threshold=5
    )
    print(f"[Restoration] Detected {num_det_faces} face(s)")
    face_helper.align_warp_face()
    
    # Process each detected face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        
        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            torch.cuda.empty_cache()
        except Exception as error:
            print(f"[Restoration] ERROR during inference: {error}")
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
        
        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face, cropped_face)
    
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image(draw_box=False)
    
    output_filename = f"{basename}_restored.png"
    output_path = os.path.join(output_dir, output_filename)
    imwrite(restored_img, output_path)
    print(f"[Restoration] Output saved to: {output_path}")
    return output_filename

