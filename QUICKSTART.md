# 🚀 Quick Start Guide

Get the Face Analytics Portal up and running in minutes!

## Prerequisites

- **Anaconda or Miniconda** installed
- **Git** installed
- **CUDA-compatible GPU** (recommended, but not required)

## Installation

### Option 1: One-Click Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd CodeFormerServer2

# Run the installation script
chmod +x install_and_run.sh
./install_and_run.sh
```

The script will automatically:
- Create a conda environment called `codeformer`
- Install all dependencies
- Download pre-trained models
- Start the web application

### Option 2: Manual Installation

```bash
# Create conda environment
conda create -n codeformer python=3.8 -y
conda activate codeformer

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install FFmpeg
conda install -c conda-forge ffmpeg -y

# Install Python dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_pretrained_models.py
```

## Running the Application

```bash
# Activate the environment
conda activate codeformer

# Start the application
python app.py
```

The portal will be available at: **http://localhost:5000**

## First Steps

1. **Visit the landing page** to see all available features
2. **Try Face Restoration** with a test image
3. **Explore other features** like inpainting and comparison
4. **Check the debug panel** for video enhancement

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Use CPU instead
export CUDA_VISIBLE_DEVICES=""
python app.py
```

**Port Already in Use**
```bash
# Change port in app.py
app.run(host='0.0.0.0', port=5001, debug=True)
```

**Model Download Issues**
```bash
# Manual download
python scripts/download_pretrained_models.py CodeFormer
python scripts/download_pretrained_models.py facelib
```

### Getting Help

- Check the full [README.md](README.md) for detailed documentation
- Review the troubleshooting section
- Create an issue with detailed error information

## Next Steps

- Read the [full documentation](README.md)
- Explore all features in the web interface
- Try different fidelity weights for face restoration
- Test video enhancement with short clips first

---

**Happy face analytics! 🎭**
