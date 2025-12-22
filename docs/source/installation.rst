Installation
============

Requirements
-----------

* Python 3.12 or higher
* CUDA-capable GPU (recommended) or CPU
* 8GB+ RAM (16GB+ recommended)
* 50GB+ disk space for models

Installation Methods
-------------------

From PyPI (Recommended)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install pipeworks-image-generator

From Source
^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/pipeworks-image-generator.git
   cd pipeworks-image-generator

   # Install in development mode
   pip install -e .

   # Or install with dev dependencies
   pip install -e ".[dev]"

   # Or install with docs dependencies
   pip install -e ".[docs]"

Environment Configuration
------------------------

Pipeworks uses environment variables for configuration. Create a ``.env`` file in your project root:

.. code-block:: bash

   # Model settings
   PIPEWORKS_MODEL_ID=Xkev/Z-Image-Turbo
   PIPEWORKS_DEVICE=cuda  # or cpu, mps
   PIPEWORKS_TORCH_DTYPE=bfloat16  # or float16, float32

   # Directories
   PIPEWORKS_MODELS_DIR=./models
   PIPEWORKS_OUTPUTS_DIR=./outputs
   PIPEWORKS_INPUTS_DIR=./inputs

   # Generation defaults
   PIPEWORKS_NUM_INFERENCE_STEPS=9
   PIPEWORKS_DEFAULT_WIDTH=1024
   PIPEWORKS_DEFAULT_HEIGHT=1024

   # Gradio UI settings
   PIPEWORKS_GRADIO_SERVER_PORT=7860
   PIPEWORKS_GRADIO_SERVER_NAME=0.0.0.0

See ``.env.example`` in the repository for all available options.

Verification
-----------

Verify your installation:

.. code-block:: python

   from pipeworks import ImageGenerator

   # This should print version info
   generator = ImageGenerator()
   print("Pipeworks installed successfully!")

Troubleshooting
--------------

CUDA Not Available
^^^^^^^^^^^^^^^^^

If you see "CUDA is not available" warnings:

1. Verify CUDA is installed: ``nvidia-smi``
2. Install PyTorch with CUDA support: ``pip install torch --index-url https://download.pytorch.org/whl/cu121``
3. Set ``PIPEWORKS_DEVICE=cpu`` to run on CPU

Memory Issues
^^^^^^^^^^^^

If you encounter out-of-memory errors:

1. Reduce batch size
2. Use ``PIPEWORKS_TORCH_DTYPE=float16`` instead of ``bfloat16``
3. Close other GPU applications
4. Set ``PIPEWORKS_DEVICE=cpu`` to offload to CPU
