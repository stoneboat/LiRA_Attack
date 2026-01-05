#!/bin/bash
module purge
# Load CUDA 11.7.0 and cuDNN 8.5.0.96 (compatible with TensorFlow 2.13.0)
# Note: TensorFlow 2.13.0 was built with CUDA 11.8, but CUDA 11.7 is compatible
# and we need cuDNN 8.5.0.96 which requires CUDA 11.7
module load cuda/11.7.0
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

# Install requirements (jax and jaxlib will be CPU-only initially)
PYTHONNOUSERSITE=1 pip install -r requirements.txt || { echo "pip install requirements failed" >&2; exit 1; }

# Reinstall jaxlib with CUDA support for CUDA 11.7
# The standard pip install of jaxlib is CPU-only, so we need to install the CUDA-enabled version
# Note: jaxlib compiled with cuDNN 8.2 should work with cuDNN 8.5 runtime (backward compatible)
echo "Installing JAX with CUDA support..."
PYTHONNOUSERSITE=1 pip uninstall -y jaxlib jax || true
# Install jax and jaxlib with CUDA 11 and cuDNN 8.2 (compatible with cuDNN 8.5 runtime)
# Using jaxlib 0.4.7 as it's the latest version with cuDNN 8.2 support
PYTHONNOUSERSITE=1 pip install --upgrade "jax==0.4.7" "jaxlib==0.4.7+cuda11.cudnn82" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || {
    echo "Error: Failed to install CUDA-enabled jaxlib." >&2
    echo "JAX will use CPU only. Please check your CUDA installation." >&2
    exit 1
}

# Avoid user-site interference
PYTHONNOUSERSITE=1 pip install ipykernel || { echo "pip install ipykernel failed" >&2; exit 1; }

# Uninstall existing kernel if it exists (to avoid conflicts)
python -m ipykernel uninstall --user --name=lra-env -y 2>/dev/null || true

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
# Load CUDA 11.7.0 and cuDNN 8.5.0.96 (compatible with TensorFlow 2.13.0)
module load cuda/11.7.0
module load cudnn/8.5.0.96-11.7-cuda

# Extract CUDA_HOME from the module's LD_LIBRARY_PATH
# The module sets LD_LIBRARY_PATH to include CUDA lib64 directory
# CUDA_HOME should be the parent of lib64
if [ -z "$CUDA_HOME" ]; then
    # Try to find CUDA_HOME from LD_LIBRARY_PATH
    for path in $(echo "$LD_LIBRARY_PATH" | tr ':' ' '); do
        if [[ "$path" == *"cuda"* ]] && [[ "$path" == *"lib64"* ]]; then
            CUDA_HOME="${path%/lib64}"
            export CUDA_HOME
            export CUDA_ROOT="$CUDA_HOME"
            break
        fi
    done
fi

# CUDNNROOT should already be set by the module, but ensure it's exported
if [ -n "$CUDNNROOT" ]; then
    export CUDNNROOT
fi

# JAX memory management: Prevent pre-allocation of GPU memory
# This allows JAX to allocate memory on-demand instead of grabbing all GPU memory at startup
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Limit JAX to use at most 80% of GPU memory (leaves room for TensorFlow or other processes)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
# Use platform allocator for better memory management
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# TensorFlow memory management: Prevent pre-allocation
# Allow TensorFlow to grow memory usage as needed instead of allocating all at once
export TF_FORCE_GPU_ALLOW_GROWTH=true

# TensorFlow module loads CUDA 12.1.1, but we need CUDA 11.7 for pip-installed TF 2.13.0
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

# Verify kernel registration
if [ ! -f "$KERNEL_DIR/kernel.json" ]; then
    echo "Error: kernel.json was not created properly" >&2
    exit 1
fi
if [ ! -x "$KERNEL_WRAPPER" ]; then
    echo "Error: kernel_wrapper.sh is not executable" >&2
    exit 1
fi

conda deactivate

echo ""
echo "✅ Installation complete!"
echo "⚠️  Restart your Jupyter kernel to use GPU in notebooks."
