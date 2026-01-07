#!/bin/bash
module purge
module load pytorch/25
module load anaconda3

# Initialize conda in this non-interactive shell
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found after loading anaconda3 module" >&2
    exit 1
fi

mkdir -p /tmp/python-venv

if [ -d "/tmp/python-venv/onerun_lra_venv" ]; then
    echo "Conda env 'onerun_lra_venv' already exists in /tmp/python-venv."
else
    echo "Creating conda env 'onerun_lra_venv' in /tmp/python-venv..."
    conda create --prefix /tmp/python-venv/onerun_lra_venv python=3.12 -y || { echo "conda create failed" >&2; exit 1; }
fi

# Activate the conda environment
conda activate /tmp/python-venv/onerun_lra_venv || { echo "conda activate failed" >&2; exit 1; }

# Upgrade pip
pip install --upgrade pip || { echo "pip upgrade failed" >&2; exit 1; }

# Install opacus with --no-deps first (if in requirements.txt)
if [ -f requirements.txt ] && grep -q "^opacus" requirements.txt; then
    OPACUS_LINE=$(grep "^opacus" requirements.txt | head -1)
    PYTHONNOUSERSITE=1 pip install --no-deps "$OPACUS_LINE" || { echo "pip install opacus failed" >&2; exit 1; }
fi

# Install requirements.txt (excluding opacus since it's already installed with --no-deps)
if [ -f requirements.txt ] && [ -s requirements.txt ]; then
    # Create a temporary requirements file without opacus
    TEMP_REQ=$(mktemp)
    grep -v "^opacus" requirements.txt > "$TEMP_REQ" || true
    if [ -s "$TEMP_REQ" ]; then
        PYTHONNOUSERSITE=1 pip install -r "$TEMP_REQ" || { echo "pip install requirements failed" >&2; rm -f "$TEMP_REQ"; exit 1; }
    fi
    rm -f "$TEMP_REQ"
fi

# Uninstall existing kernel if it exists (to avoid conflicts)
python -m ipykernel uninstall --user --name=onerun-lra-env -y 2>/dev/null || true

# Register kernel first (this creates the directory and basic kernel.json)
python -m ipykernel install --user --name=onerun-lra-env --display-name "Python (onerun-lra-env)" || { echo "ipykernel install failed" >&2; exit 1; }

# Create the custom kernel spec directory (in case ipykernel didn't create it)
KERNEL_DIR=~/.local/share/jupyter/kernels/onerun-lra-env
mkdir -p "$KERNEL_DIR"

# Create a simple wrapper that loads modules before starting kernel
KERNEL_WRAPPER="$KERNEL_DIR/kernel_wrapper.sh"
LOG_FILE="$KERNEL_DIR/kernel_wrapper.log"
cat > "$KERNEL_WRAPPER" <<'WRAPPER_EOF'
#!/bin/bash
# Simple wrapper to load modules before starting kernel
# The Python executable is already specified in kernel.json, so we just need to:
# 1. Load pytorch/25 module (provides torch)
# 2. Set up PYTHONPATH so Python can find torch

# Initialize module system
if [ -f /usr/local/pace-apps/lmod/lmod/init/bash ]; then
    source /usr/local/pace-apps/lmod/lmod/init/bash
fi

# Load pytorch/25 module
module load pytorch/25 2>/dev/null

# Get Python version from the Python executable that will be used (from kernel.json)
# The first argument after the wrapper should be the Python executable path
PYTHON_EXE="$1"
if [ -n "$PYTHON_EXE" ] && [ -f "$PYTHON_EXE" ]; then
    PYTHON_VERSION=$("$PYTHON_EXE" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    
    # If torch is not accessible, set up PYTHONPATH
    if ! "$PYTHON_EXE" -c "import torch" 2>/dev/null; then
        # Add Python dist-packages to PYTHONPATH (where pytorch module installs torch)
        PYTHON_DIST_PACKAGES="/usr/local/lib/python${PYTHON_VERSION}/dist-packages"
        PYTHON_DIST_PACKAGES64="/usr/local/lib64/python${PYTHON_VERSION}/dist-packages"
        
        # Build PYTHONPATH with both paths
        NEW_PYTHONPATH=""
        [ -d "$PYTHON_DIST_PACKAGES" ] && NEW_PYTHONPATH="$PYTHON_DIST_PACKAGES"
        [ -d "$PYTHON_DIST_PACKAGES64" ] && {
            [ -n "$NEW_PYTHONPATH" ] && NEW_PYTHONPATH="$PYTHON_DIST_PACKAGES64:$NEW_PYTHONPATH" || NEW_PYTHONPATH="$PYTHON_DIST_PACKAGES64"
        }
        
        # If directories don't exist, add them anyway (module might set them up dynamically)
        [ -z "$NEW_PYTHONPATH" ] && NEW_PYTHONPATH="$PYTHON_DIST_PACKAGES64:$PYTHON_DIST_PACKAGES"
        
        # Add to existing PYTHONPATH or set new one
        [ -z "$PYTHONPATH" ] && export PYTHONPATH="$NEW_PYTHONPATH" || export PYTHONPATH="$NEW_PYTHONPATH:$PYTHONPATH"
    fi
fi

# Execute the python command with all arguments
[ -n "$PYTHONPATH" ] && exec env PYTHONPATH="$PYTHONPATH" "$@" || exec "$@"
WRAPPER_EOF
chmod +x "$KERNEL_WRAPPER"

# Write the kernel.json with wrapper (use absolute path for wrapper)
KERNEL_WRAPPER_ABS=$(readlink -f "$KERNEL_WRAPPER" 2>/dev/null || echo "$KERNEL_WRAPPER")
cat > "$KERNEL_DIR/kernel.json" <<EOL
{
  "argv": [
    "$KERNEL_WRAPPER_ABS",
    "/tmp/python-venv/onerun_lra_venv/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python (onerun-lra-env)",
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
echo "⚠️  Restart your Jupyter kernel to use the new environment in notebooks."