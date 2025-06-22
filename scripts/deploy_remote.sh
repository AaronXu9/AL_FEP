#!/bin/bash
"""
Remote Deployment Script for AL-FEP
Run this script on any remote server to set up the AL-FEP environment
"""

set -e  # Exit on any error

echo "🚀 AL-FEP Remote Deployment Script"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're on a remote server
print_info "Checking system information..."
echo "OS: $(uname -s)"
echo "Architecture: $(uname -m)"
echo "User: $(whoami)"
echo "Working directory: $(pwd)"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi
print_status "Git is available"

# Check if conda/miniconda is installed
if ! command -v conda &> /dev/null; then
    print_warning "Conda is not found. Installing Miniconda..."
    
    # Detect architecture and OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [[ $(uname -m) == "x86_64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else
            MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Download and install miniconda
    print_info "Downloading Miniconda from $MINICONDA_URL"
    curl -O $MINICONDA_URL
    bash Miniconda3-latest-*.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-*.sh
    
    # Initialize conda
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    print_status "Miniconda installed successfully"
    print_warning "Please restart your shell or run: source ~/.bashrc"
    print_warning "Then run this script again."
    exit 0
else
    print_status "Conda is available"
fi

# Clone the repository
REPO_URL="https://github.com/AaronXu9/AL_FEP.git"
PROJECT_DIR="AL_FEP"

if [ -d "$PROJECT_DIR" ]; then
    print_warning "Directory $PROJECT_DIR already exists. Updating..."
    cd $PROJECT_DIR
    git pull origin main
else
    print_info "Cloning repository from $REPO_URL"
    git clone $REPO_URL
    cd $PROJECT_DIR
fi

print_status "Repository cloned/updated successfully"

# Check if conda environment already exists
if conda env list | grep -q "al_fep"; then
    print_warning "Conda environment 'al_fep' already exists. Updating..."
    conda env update -f environment.yml
else
    print_info "Creating conda environment from environment.yml"
    conda env create -f environment.yml
fi

print_status "Conda environment setup complete"

# Activate environment and install package
print_info "Installing AL-FEP package..."
# Note: We need to activate in a subshell for this script
eval "$(conda shell.bash hook)"
conda activate al_fep

# Install the package in development mode
pip install -e .

print_status "Package installed successfully"

# Run basic tests to verify installation
print_info "Running basic tests to verify installation..."
if python -c "import al_fep; print(f'AL-FEP version: {al_fep.__version__}')"; then
    print_status "Package import successful"
else
    print_error "Package import failed"
    exit 1
fi

# Test core functionality
if python -c "
from al_fep.molecular import MolecularFeaturizer
from al_fep.utils import load_config
print('Core modules imported successfully')
"; then
    print_status "Core functionality test passed"
else
    print_warning "Some core modules may have dependency issues"
fi

# Create necessary directories
mkdir -p logs results data/raw data/processed data/external

# Set permissions
chmod +x scripts/*.py

print_status "Directory structure created"

echo ""
echo "🎉 AL-FEP deployment completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate al_fep"
echo "2. Configure your target: cp config/targets/7jvr.yaml config/my_target.yaml"
echo "3. Run a quick test: python scripts/quickstart.py"
echo "4. Check the documentation: open notebooks/01_getting_started.ipynb"
echo ""
echo "Repository location: $(pwd)"
echo "Environment name: al_fep"
echo "Python version: $(python --version)"
echo ""
print_info "For detailed usage instructions, see README.md"
