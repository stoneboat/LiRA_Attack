echo "Installing Environment for Likelihood Ratio Attack [LiRA Environment]"

# Load TensorFlow module for CUDA environment (but we'll use system Python for venv)
if command -v module >/dev/null 2>&1; then
    echo "Loading TensorFlow module for CUDA environment..."
    module load tensorflow/2.13 2>/dev/null || echo "WARNING: Could not load tensorflow module"
fi

# Set the virtual environment path
mkdir -p /tmp/python-venv
MAIN_VENV_PATH="/tmp/python-venv/lra_venv"

if [ -d "$MAIN_VENV_PATH" ]; then
    echo "Virtual environment 'lra_venv' already exists in $MAIN_VENV_PATH."
else
    echo "Creating virtual environment 'lra_venv' using system Python..."
    python3.9 -m venv "$MAIN_VENV_PATH"
fi

source "$MAIN_VENV_PATH/bin/activate"

pip install --upgrade pip

# Install packages from requirements.txt
# Note: TensorFlow from module will be available when module is loaded
# But we can also install it via pip if needed for venv isolation
pip install -r requirements.txt

# Verify TensorFlow can see GPU
echo "Verifying TensorFlow GPU support..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU devices:', gpus); print('GPU available:', len(gpus) > 0)" 2>&1 | grep -v "Unable to register" || {
    echo "WARNING: TensorFlow GPU not detected. Make sure TensorFlow module is loaded."
}

# Register the kernel with current environment variables
python -m ipykernel install --user --name=lra-env --display-name "lra-env"

KERNEL_DIR=~/.local/share/jupyter/kernels/lra-env
mkdir -p "$KERNEL_DIR"
ABSOLUTE_VENV_PATH=$(realpath "$MAIN_VENV_PATH")

# Save current environment to kernel.json so it's available in Jupyter
cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "$ABSOLUTE_VENV_PATH/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (lra-env)",
  "language": "python",
  "env": {
    "CUDA_HOME": "${CUDA_HOME:-}",
    "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH:-}",
    "PATH": "${PATH:-}"
  }
}
EOL

echo ""
echo "✅ Installation complete!"
echo ""
echo "⚠️  IMPORTANT:"
echo "   - TensorFlow module should be loaded when using TensorFlow (it provides GPU support)"
echo "   - Restart your Jupyter kernel to use GPU in notebooks"
    echo "   - To use TensorFlow: 'module load tensorflow/2.13' before running Python"

deactivate
