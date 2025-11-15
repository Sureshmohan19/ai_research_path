#!/usr/bin/env bash

# exit immediately on any error
set -e
echo "Running bootstrap script..."

# OS check—must be Linux
OS="$(uname -s)"
if [ "$OS" != "Linux" ]; then
    echo "Error: This script only supports Linux."
    echo "Detected: $OS"
    exit 1
fi
echo "Linux detected. moving on to shell detection..."

# Shell detection-only bash or zsh
# Detect user shell
if [ -n "$SHELL" ]; then
    DETECTED_SHELL=$(basename "$SHELL")
else
    DETECTED_SHELL="bash"
fi

# Only accept bash or zsh
if [ "$DETECTED_SHELL" != "bash" ] && [ "$DETECTED_SHELL" != "zsh" ]; then
    echo "Unsupported shell '$DETECTED_SHELL'. Defaulting to bash."
    DETECTED_SHELL="bash"
fi
echo "Using $DETECTED_SHELL shell to run commands for this script. moving on to system updates..."

# Update system packages
echo "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y
echo "System updated. moving to micromamba installation now"

# Install Micromamba (official installer)
echo "Installing Micromamba..."
curl -Ls https://micro.mamba.pm/install.sh -o /tmp/micromamba_install.sh
chmod +x /tmp/micromamba_install.sh
/bin/bash /tmp/micromamba_install.sh
echo " Micromamba installed successfully. moving on to vim setup..."
echo "➡ Restart shell or run: source ~/.bashrc  (or ~/.zshrc) once boostrap script done setting up."

echo "Setting up Micromamba environment ai_research..."
eval "$($HOME/.local/bin/micromamba shell hook -s $DETECTED_SHELL)"
micromamba create -y -n ai_research python=3.10 jax
echo "Environment 'ai_research' created."
echo "To activate it, run: micromamba activate ai_research"

# Install or update Vim
echo "Installing/Updating Vim..."
sudo apt-get install -y vim
echo "Vim installed."

# Create .vimrc with defaults + Gruvbox
echo "Setting up Vim configuration..."
# Write .vimrc (overwrite because this is a bootstrap script)
cat << 'EOF' > ~/.vimrc
" Enable syntax highlighting
syntax enable

" UI settings
set number
set relativenumber
set cursorline
set ruler
set cmdheight=1

" Search improvements
set ignorecase
set smartcase
set hlsearch
set incsearch

" Tabs & indentation
set expandtab
set smarttab
set shiftwidth=4
set tabstop=4
set autoindent
set smartindent

" Colors
set background=dark
set termguicolors
colorscheme gruvbox

" Better backspace behavior
set backspace=indent,eol,start

" Custom statusline
function! HasPaste()
  return &paste ? '[PASTE]' : ''
endfunction

set laststatus=2
set statusline=%{HasPaste()}%F%m%r%h%w\ CWD:%{getcwd()}\ Line:%l\ Col:%c
EOF
echo ".vimrc created"

# Install Gruvbox colorscheme
echo "Installing Gruvbox colorscheme..."
mkdir -p ~/.vim/colors
curl -fLo ~/.vim/colors/gruvbox.vim --create-dirs \
https://raw.githubusercontent.com/morhetz/gruvbox/master/colors/gruvbox.vim
