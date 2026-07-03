#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3.12 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# macOS: XGBoost needs OpenMP (libomp.dylib)
if [[ "$(uname -s)" == "Darwin" ]]; then
    LIBOMP_VENV="$VENV_DIR/lib/libomp.dylib"
    LIBOMP_BREW="/opt/homebrew/opt/libomp/lib/libomp.dylib"

    if [[ ! -f "$LIBOMP_VENV" ]]; then
        if [[ -f "$LIBOMP_BREW" ]]; then
            mkdir -p "$VENV_DIR/lib"
            cp "$LIBOMP_BREW" "$LIBOMP_VENV"
        elif command -v brew &>/dev/null; then
            echo "Installing libomp (OpenMP runtime for XGBoost)..."
            brew install libomp || true
            if [[ -f "$LIBOMP_BREW" ]]; then
                mkdir -p "$VENV_DIR/lib"
                cp "$LIBOMP_BREW" "$LIBOMP_VENV"
            fi
        fi

        # Fallback: copy libomp from another installed Python package (e.g. PyTorch)
        if [[ ! -f "$LIBOMP_VENV" ]]; then
            TORCH_LIBOMP=$(find /Library/Frameworks/Python.framework /opt/homebrew/lib -name libomp.dylib 2>/dev/null | head -1)
            if [[ -n "$TORCH_LIBOMP" ]]; then
                mkdir -p "$VENV_DIR/lib"
                cp "$TORCH_LIBOMP" "$LIBOMP_VENV"
                echo "Copied libomp from $TORCH_LIBOMP into $VENV_DIR/lib/"
            fi
        fi
    fi

    ACTIVATE="$VENV_DIR/bin/activate"
    if [[ -f "$LIBOMP_VENV" ]] && ! grep -q 'libomp.dylib' "$ACTIVATE" 2>/dev/null; then
        cat >> "$ACTIVATE" << 'EOF'

# XGBoost on macOS needs libomp on the dynamic linker path
if [[ -n "$VIRTUAL_ENV" && -f "$VIRTUAL_ENV/lib/libomp.dylib" ]]; then
  export DYLD_LIBRARY_PATH="$VIRTUAL_ENV/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi
EOF
    elif [[ ! -f "$LIBOMP_VENV" ]]; then
        echo ""
        echo "WARNING: libomp not found. XGBoost will fail until OpenMP is installed."
        echo "  brew install libomp"
        echo "Then re-run: ./setup.sh"
    fi
fi

echo ""
echo "Setup complete. Virtual environment is active."
echo "To activate it in the future, run: source $VENV_DIR/bin/activate"
