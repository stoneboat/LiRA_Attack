#!/bin/bash
module purge
# Load CUDA 11.8.0 (TensorFlow 2.13.0 was built with CUDA 11.8)
module load cuda/11.8.0
module load cudnn/8.5.0.96-11.7-cuda
module load anaconda3

# Initialize conda in this non-interactive shell
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found after loading anaconda3 module" >&2
    exit 1
fi

mkdir -p /tmp/python-venv

if [ -d "/tmp/python-venv/lra_venv" ]; then
    echo "Conda env 'lra_venv' already exists in /tmp/python-venv."
else
    echo "Creating conda env 'lra_venv' in /tmp/python-venv..."
    conda create --prefix /tmp/python-venv/lra_venv python=3.9 -y || { echo "conda create failed" >&2; exit 1; }
fi

conda activate /tmp/python-venv/lra_venv || { echo "conda activate failed" >&2; exit 1; }

# Install requirements
PYTHONNOUSERSITE=1 pip install -r requirements.txt || { echo "pip install requirements failed" >&2; exit 1; }

# Avoid user-site interference
PYTHONNOUSERSITE=1 pip install ipykernel || { echo "pip install ipykernel failed" >&2; exit 1; }

# Register kernel first (this creates the directory and basic kernel.json)
python -m ipykernel install --user --name=lra-env --display-name "Python (lra-env)" || { echo "ipykernel install failed" >&2; exit 1; }

# Create the custom kernel spec directory (in case ipykernel didn't create it)
KERNEL_DIR=~/.local/share/jupyter/kernels/lra-env
mkdir -p "$KERNEL_DIR"

# Create a simple wrapper that loads modules before starting kernel
KERNEL_WRAPPER="$KERNEL_DIR/kernel_wrapper.sh"
cat > "$KERNEL_WRAPPER" <<'WRAPPER_EOF'
#!/bin/bash
# Simple wrapper to load modules before starting kernel
if [ -f /usr/local/pace-apps/lmod/lmod/init/bash ]; then
    source /usr/local/pace-apps/lmod/lmod/init/bash
fi
# Load CUDA 11.8 (TensorFlow 2.13.0 was built with CUDA 11.8)
module load cuda/11.8.0
module load cudnn/8.5.0.96-11.7-cuda
# TensorFlow module loads CUDA 12.1.1, but we need CUDA 11.8 for pip-installed TF
# So we don't load tensorflow module, just CUDA/cuDNN
exec "$@"
WRAPPER_EOF
chmod +x "$KERNEL_WRAPPER"

# Write the kernel.json with wrapper (use absolute path for wrapper)
KERNEL_WRAPPER_ABS=$(readlink -f "$KERNEL_WRAPPER" 2>/dev/null || echo "$KERNEL_WRAPPER")
cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "$KERNEL_WRAPPER_ABS",
    "/tmp/python-venv/lra_venv/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (lra-env)",
  "language": "python"
}
EOL

conda deactivate

echo ""
echo "✅ Installation complete!"
echo "⚠️  Restart your Jupyter kernel to use GPU in notebooks."
