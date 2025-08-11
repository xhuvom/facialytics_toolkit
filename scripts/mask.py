import os
import glob
import numpy as np
import PIL.Image
import scipy.ndimage
import argparse
from basicsr.utils.download_util import load_file_from_url

try:
    import dlib
except ImportError:
    print('Please install dlib by running: conda install -c conda-forge dlib')

# download model from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
shape_predictor_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/shape_predictor_68_face_landmarks-fbdc2cb8.dat'
ckpt_path = load_file_from_url(url=shape_predictor_url, 
                               model_dir='weights/dlib', progress=True, file_name=None)
predictor = dlib.shape_predictor('weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat')

def get_landmark(filepath, only_keep_largest=True):
    """get landmark with dlib
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    print("\tNumber of faces detected: {}".format(len(dets)))
    if only_keep_largest:
        print('\tOnly keep the largest.')
        face_areas = []
        for d in dets:
            face_area = (d.right() - d.left()) * (d.bottom() - d.top())
            face_areas.append(face_area)
        largest_idx = face_areas.index(max(face_areas))
        d = dets[largest_idx]
        shape = predictor(img, d)
    else:
        # Use all faces (not shown for brevity)
        shape = predictor(img, dets[0])
    lm = np.array([[p.x, p.y] for p in shape.parts()])
    return lm

def align_face(filepath, out_path):
    """
    Detects the face in the input image, aligns and crops it to 512×512,
    then creates a binary mask image (same resolution) that covers the lower half.
    
    The mask is saved with the same basename plus a "_mask" suffix.
    """
    try:
        lm = get_landmark(filepath)
    except Exception as e:
        print('No landmark detected in {}: {}'.format(filepath, e))
        return

    # Extract landmark groups (chin, eyes, etc.) – same as your original code.
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]
    
    # Calculate auxiliary vectors (using eyes and mouth centers)
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_eye = eye_right - eye_left
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Read image.
    img = PIL.Image.open(filepath)

    output_size = 512
    transform_size = 4096
    enable_padding = False

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)),
                 int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad (skipped if not enabled).
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        quad += pad[:2]
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD,
                        (quad + 0.5).flatten(), PIL.Image.BILINEAR)

    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    # Save aligned image.
    img.save(out_path)
    print("Aligned face saved to:", out_path)

    # ----- New code: create mask for inpainting -----
    # Create a mask image (512x512) where the lower half is white.
    mask_array = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    mask_array[output_size//2:, :] = 255  # lower half white
    mask_img = PIL.Image.fromarray(mask_array)
    mask_out_path = out_path.replace('.png', '_mask.png')
    mask_img.save(mask_out_path)
    print("Mask image saved to:", mask_out_path)
    # -------------------------------------------------

    return img, np.max(quad[:, 0]) - np.min(quad[:, 0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_dir', type=str, default='./inputs/whole_imgs')
    parser.add_argument('-o', '--out_dir', type=str, default='./inputs/cropped_faces')
    args = parser.parse_args()

    if args.out_dir.endswith('/'):  # remove trailing slash if exists
        args.out_dir = args.out_dir[:-1]
    dir_name = os.path.abspath(args.out_dir)
    os.makedirs(dir_name, exist_ok=True)

    img_list = sorted(glob.glob(os.path.join(args.in_dir, '*.[jpJP][pnPN]*[gG]')))
    test_img_num = len(img_list)

    for i, in_path in enumerate(img_list):
        img_name = os.path.basename(in_path)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        out_path = os.path.join(args.out_dir, os.path.basename(in_path))
        out_path = out_path.replace('.jpg', '.png')
        align_face(in_path, out_path)

