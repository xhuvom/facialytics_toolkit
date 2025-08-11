# utils/models.py
import torch
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.utils.misc import get_device

device = get_device()

def load_restoration_model():
    print("[Models] Loading face restoration model...")
    net = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=['32', '64', '128', '256']
    ).to(device)
    pretrain_model_url = {
        'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
    }
    ckpt_path = load_file_from_url(
        url=pretrain_model_url["restoration"],
        model_dir="weights/CodeFormer",
        progress=True,
        file_name=None
    )
    checkpoint = torch.load(ckpt_path, map_location=device)["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()
    print("[Models] Face restoration model loaded.")
    return net

def load_inpainting_model():
    print("[Models] Loading face inpainting model...")
    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512, codebook_size=512, n_head=8, n_layers=9,
        connect_list=["32", "64", "128"]
    ).to(device)
    pretrain_model_url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth"
    ckpt_path = load_file_from_url(
        url=pretrain_model_url,
        model_dir="weights/CodeFormer",
        progress=True,
        file_name=None
    )
    checkpoint = torch.load(ckpt_path, map_location=device)["params_ema"]
    net.load_state_dict(checkpoint)
    net.eval()
    print("[Models] Face inpainting model loaded.")
    return net

def load_comparison_model():
    print("[Models] Loading face comparison model (InsightFace)...")
    import insightface
    model = insightface.app.FaceAnalysis(name='buffalo_l')
    # Use ctx_id=0 for GPU (or -1 for CPU)
    model.prepare(ctx_id=0, det_size=(640, 640))
    print("[Models] Face comparison model loaded.")
    return model

# Global instances - these lines ensure that the models are loaded at startup.
RESTORATION_MODEL = load_restoration_model()
INPAINTING_MODEL = load_inpainting_model()
COMPARISON_MODEL = load_comparison_model()

