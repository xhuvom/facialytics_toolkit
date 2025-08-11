#!/usr/bin/env python3
"""
Model Verification Script for Face Analytics Portal
Checks if all required AI models are present and properly sized.
"""

import os
import sys
from pathlib import Path

def check_model(model_path, expected_size_mb, description):
    """Check if a model file exists and has the expected size."""
    if not os.path.exists(model_path):
        print(f"❌ Missing: {description}")
        print(f"   Expected: {model_path}")
        return False
    
    actual_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    if actual_size_mb < expected_size_mb * 0.9:  # Allow 10% tolerance
        print(f"⚠️  Corrupted: {description}")
        print(f"   Expected: ~{expected_size_mb}MB, Actual: {actual_size_mb:.1f}MB")
        return False
    
    print(f"✅ {description}: {actual_size_mb:.1f}MB")
    return True

def main():
    print("🤖 Face Analytics Portal - Model Verification")
    print("=" * 50)
    
    models_to_check = [
        # CodeFormer models
        ("weights/CodeFormer/codeformer.pth", 359, "CodeFormer Main Model"),
        ("weights/CodeFormer/codeformer_inpainting.pth", 354, "CodeFormer Inpainting Model"),
        
        # FaceLib models
        ("weights/facelib/detection_Resnet50_Final.pth", 104, "Face Detection Model"),
        ("weights/facelib/parsing_parsenet.pth", 81, "Face Parsing Model"),
        
        # Dlib models
        ("weights/dlib/mmod_human_face_detector-4cb19393.dat", 0.7, "Dlib Face Detector"),
        ("weights/dlib/shape_predictor_5_face_landmarks-c4b1e980.dat", 8.7, "Dlib 5-Point Landmarks"),
        ("weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat", 95, "Dlib 68-Point Landmarks"),
        
        # RealESRGAN models
        ("weights/realesrgan/RealESRGAN_x2plus.pth", 64, "RealESRGAN 2x Upscaling Model"),
    ]
    
    all_good = True
    total_size = 0
    
    for model_path, expected_size, description in models_to_check:
        if check_model(model_path, expected_size, description):
            total_size += expected_size
        else:
            all_good = False
    
    print("\n" + "=" * 50)
    
    if all_good:
        print(f"🎉 All models verified successfully!")
        print(f"📊 Total model size: ~{total_size}MB")
        print("\n✅ Your Face Analytics Portal is ready to use!")
    else:
        print("❌ Some models are missing or corrupted.")
        print("\nTo download missing models, run:")
        print("  python scripts/download_pretrained_models.py all")
        print("\nOr use the installation script:")
        print("  ./install_and_run.sh --install-only")
        sys.exit(1)

if __name__ == "__main__":
    main()
