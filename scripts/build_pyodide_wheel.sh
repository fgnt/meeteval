PYODIDE_FOLDER=build/pyodide

workdir=$(pwd)

# Clone pyodide if not present
if [ ! -d $PYODIDE_FOLDER ]; then
    git clone https://github.com/pyodide/pyodide.git $PYODIDE_FOLDER
fi

# Build patched version of emsdk
cd $PYODIDE_FOLDER/emsdk
make

# Install pyodide-build (and pyodide-cli) in the current environment
# We know that pyodide-build==0.28.0 works, earlier versions may not work
pip install pyodide-build>=0.28.0

# Source the environment
PYODIDE_EMSCRIPTEN_VERSION=$(pyodide config get emscripten_version)
./emsdk install ${PYODIDE_EMSCRIPTEN_VERSION}
./emsdk activate ${PYODIDE_EMSCRIPTEN_VERSION}
source emsdk_env.sh

cd $workdir
pyodide build
