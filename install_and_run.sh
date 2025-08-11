#!/bin/bash

# Face Analytics Portal - Installation and Run Script
# This script sets up the complete environment and runs the application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check CUDA availability
check_cuda() {
    if command_exists nvidia-smi; then
        print_status "CUDA detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        return 0
    else
        print_warning "CUDA not detected. Installation will proceed with CPU-only PyTorch."
        return 1
    fi
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p uploads
    mkdir -p static/outputs
    mkdir -p results
    mkdir -p weights/CodeFormer
    mkdir -p weights/facelib
    mkdir -p weights/dlib
    
    print_success "Directories created successfully"
}

# Function to install conda environment
install_conda_env() {
    print_status "Setting up conda environment..."
    
    # Check if conda is available
    if ! command_exists conda; then
        print_error "Conda is not installed. Please install Anaconda or Miniconda first."
        print_error "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # Check if environment already exists
    if conda env list | grep -q "codeformer"; then
        print_warning "Conda environment 'codeformer' already exists."
        read -p "Do you want to remove it and create a fresh one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing environment..."
            conda env remove -n codeformer -y
        else
            print_status "Using existing environment..."
            return 0
        fi
    fi
    
    # Create new environment
    print_status "Creating conda environment 'codeformer' with Python 3.8..."
    conda create -n codeformer python=3.8 -y
    
    print_success "Conda environment created successfully"
}

# Function to install PyTorch
install_pytorch() {
    print_status "Installing PyTorch..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate codeformer
    
    # Check CUDA and install appropriate PyTorch version
    if check_cuda; then
        print_status "Installing PyTorch with CUDA support..."
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        print_status "Installing PyTorch for CPU..."
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    fi
    
    print_success "PyTorch installed successfully"
}

# Function to install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate codeformer
    
    # Install FFmpeg
    print_status "Installing FFmpeg..."
    conda install -c conda-forge ffmpeg -y
    
    # Install dlib
    print_status "Installing dlib..."
    conda install -c conda-forge dlib -y
    
    print_success "System dependencies installed successfully"
}

# Function to install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate codeformer
    
    # Install packages from requirements.txt
    print_status "Installing packages from requirements.txt..."
    pip install -r requirements.txt
    
    # Install additional packages
    print_status "Installing additional packages..."
    pip install Flask flask-cors insightface onnxruntime-gpu
    
    # Install BasicSR
    print_status "Installing BasicSR..."
    pip install basicsr
    
    # Install facelib
    print_status "Installing facelib..."
    pip install facelib
    
    print_success "Python packages installed successfully"
}

# Function to download pre-trained models
download_models() {
    print_status "Downloading pre-trained models..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate codeformer
    
    # Check if models already exist
    if [ -f "weights/CodeFormer/codeformer.pth" ] && [ -f "weights/facelib/detection_Resnet50_Final.pth" ]; then
        print_warning "Pre-trained models already exist. Skipping download."
        return 0
    fi
    
    # Check if wget is available
    if command_exists wget; then
        print_status "Using wget for direct model downloads..."
        download_models_wget
    else
        print_status "Using Python script for model downloads..."
        download_models_python
    fi
    
    print_success "Pre-trained models downloaded successfully"
}

# Function to download models using wget
download_models_wget() {
    print_status "Downloading CodeFormer models..."
    wget -O weights/CodeFormer/codeformer.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
    wget -O weights/CodeFormer/codeformer_inpainting.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth
    
    print_status "Downloading facelib models..."
    wget -O weights/facelib/detection_Resnet50_Final.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth
    wget -O weights/facelib/parsing_parsenet.pth https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth
    
    print_status "Downloading dlib models..."
    wget -O weights/dlib/mmod_human_face_detector-4cb19393.dat https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/mmod_human_face_detector-4cb19393.dat
    wget -O weights/dlib/shape_predictor_5_face_landmarks-c4b1e980.dat https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/shape_predictor_5_face_landmarks-c4b1e980.dat
    wget -O weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/shape_predictor_68_face_landmarks-fbdc2cb8.dat
    
    print_status "Downloading RealESRGAN model..."
    wget -O weights/realesrgan/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
}

# Function to download models using Python script
download_models_python() {
    print_status "Downloading CodeFormer models..."
    python scripts/download_pretrained_models.py CodeFormer
    
    print_status "Downloading facelib models..."
    python scripts/download_pretrained_models.py facelib
    
    print_status "Downloading dlib models..."
    python scripts/download_pretrained_models.py dlib
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate codeformer
    
    # Test imports
    print_status "Testing Python imports..."
    python -c "
import torch
import torchvision
import cv2
import numpy as np
import PIL
import flask
import insightface
import basicsr
print('All core packages imported successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
    
    # Test FFmpeg
    print_status "Testing FFmpeg..."
    ffmpeg -version | head -n 1
    
    # Verify models
    verify_models
    
    print_success "Installation verification completed"
}

# Function to verify models
verify_models() {
    print_status "Verifying AI models..."
    
    local missing_models=()
    
    # Check CodeFormer models
    if [ ! -f "weights/CodeFormer/codeformer.pth" ]; then
        missing_models+=("CodeFormer main model")
    fi
    if [ ! -f "weights/CodeFormer/codeformer_inpainting.pth" ]; then
        missing_models+=("CodeFormer inpainting model")
    fi
    
    # Check facelib models
    if [ ! -f "weights/facelib/detection_Resnet50_Final.pth" ]; then
        missing_models+=("Face detection model")
    fi
    if [ ! -f "weights/facelib/parsing_parsenet.pth" ]; then
        missing_models+=("Face parsing model")
    fi
    
    # Check dlib models
    if [ ! -f "weights/dlib/mmod_human_face_detector-4cb19393.dat" ]; then
        missing_models+=("Dlib face detector")
    fi
    if [ ! -f "weights/dlib/shape_predictor_5_face_landmarks-c4b1e980.dat" ]; then
        missing_models+=("Dlib 5-point landmarks")
    fi
    
    # Check RealESRGAN model
    if [ ! -f "weights/realesrgan/RealESRGAN_x2plus.pth" ]; then
        missing_models+=("RealESRGAN upscaling model")
    fi
    
    if [ ${#missing_models[@]} -eq 0 ]; then
        print_success "All AI models verified successfully!"
        print_status "Model sizes:"
        ls -lh weights/CodeFormer/*.pth 2>/dev/null | awk '{print "  CodeFormer: " $5 " " $9}'
        ls -lh weights/facelib/*.pth 2>/dev/null | awk '{print "  FaceLib: " $5 " " $9}'
        ls -lh weights/dlib/*.dat 2>/dev/null | awk '{print "  Dlib: " $5 " " $9}'
        ls -lh weights/realesrgan/*.pth 2>/dev/null | awk '{print "  RealESRGAN: " $5 " " $9}'
    else
        print_warning "Missing models detected:"
        for model in "${missing_models[@]}"; do
            echo "  - $model"
        done
        print_status "Run the installation script again to download missing models"
    fi
}

# Function to start the application
start_application() {
    print_status "Starting Face Analytics Portal..."
    
    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate codeformer
    
    # Check if app.py exists
    if [ ! -f "app.py" ]; then
        print_error "app.py not found. Please ensure you're in the correct directory."
        exit 1
    fi
    
    print_success "Application starting..."
    print_status "The portal will be available at: http://localhost:5000"
    print_status "Press Ctrl+C to stop the application"
    echo
    
    # Start the application
    python app.py
}

# Main installation function
main_installation() {
    echo "=========================================="
    echo "  Face Analytics Portal - Installation"
    echo "=========================================="
    echo
    
    # Check if we're in the right directory
    if [ ! -f "app.py" ] || [ ! -f "requirements.txt" ]; then
        print_error "Please run this script from the CodeFormerServer2 directory"
        exit 1
    fi
    
    # Run installation steps
    create_directories
    install_conda_env
    install_pytorch
    install_system_deps
    install_python_packages
    download_models
    verify_installation
    
    print_success "Installation completed successfully!"
    echo
    print_status "To start the application, run:"
    echo "  conda activate codeformer"
    echo "  python app.py"
    echo
    print_status "Or run this script again with: ./install_and_run.sh --start"
}

# Main function
main() {
    case "${1:-}" in
        --start)
            start_application
            ;;
        --install-only)
            main_installation
            ;;
        --help|-h)
            echo "Usage: $0 [OPTION]"
            echo
            echo "Options:"
            echo "  --start        Start the application (requires prior installation)"
            echo "  --install-only Install dependencies only (don't start app)"
            echo "  --help, -h     Show this help message"
            echo
            echo "If no option is provided, the script will install dependencies and start the application."
            ;;
        *)
            main_installation
            echo
            read -p "Do you want to start the application now? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                start_application
            fi
            ;;
    esac
}

# Run main function with all arguments
main "$@"
