pip3 install -e .
pip3 install datasets
pip3 install evaluate
pip3 install scikit-learn
pip3 install accelerate
pip3 install transformers
pip3 install torch==2.6.0.dev20241211+cpu.cxx11.abi --index-url https://download.pytorch.org/whl/nightly
pip3 install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0.dev20241211+cxx11-cp310-cp310-linux_x86_64.whl
pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
pip install https://storage.googleapis.com/libtpu-nightly-releases/wheels/libtpu-nightly/libtpu_nightly-0.1.dev20241209+nightly-py3-none-linux_x86_64.whl