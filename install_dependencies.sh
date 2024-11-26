# install_dependencies.sh
#!/bin/bash

# Upgrade pip to the latest version
pip install --upgrade pip

# Install the necessary build tools (like distutils)
apt-get update
apt-get install -y python3-distutils
