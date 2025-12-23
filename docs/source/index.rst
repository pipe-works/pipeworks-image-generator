Pipeworks Image Generator Documentation
========================================

**Pipeworks** is a Python-based image generation framework for Z-Image-Turbo that provides both a programmatic API and a Gradio web UI. The project emphasizes code-first design over node-based interfaces, with a focus on extensibility through plugins and workflows.

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: License

Key Features
-----------

* **Z-Image-Turbo Integration**: 6B parameter model via HuggingFace Diffusers
* **Gradio Web UI**: User-friendly interface for image generation
* **Plugin System**: Extensible architecture for custom functionality
* **Workflow System**: Encapsulate generation strategies for specific content types
* **Prompt Builder**: File-based prompt construction with multiple selection modes
* **Character Conditions**: Structured character state generation with semantic rules
* **Type-Safe**: Comprehensive type hints throughout the codebase
* **Well-Tested**: 50%+ test coverage with focus on core business logic

Quick Start
----------

Installation
^^^^^^^^^^^

.. code-block:: bash

   # Install with pip
   pip install pipeworks-image-generator

   # Or install from source
   git clone https://github.com/yourusername/pipeworks-image-generator.git
   cd pipeworks-image-generator
   pip install -e .

Running the UI
^^^^^^^^^^^^^

.. code-block:: bash

   # Launch Gradio UI (preferred)
   pipeworks

   # Or direct module execution
   python -m pipeworks.ui.app

The UI will be accessible at ``http://0.0.0.0:7860`` by default.

Basic Usage
^^^^^^^^^^

.. code-block:: python

   from pipeworks import model_registry, config

   # Initialize adapter
   adapter = model_registry.instantiate("Z-Image-Turbo", config)

   # Generate image
   image = adapter.generate(
       prompt="a pale purple goblin in a dark tavern",
       width=1024,
       height=1024,
       num_inference_steps=9,
       seed=42
   )

   # Save image
   image.save("output.png")

Documentation Contents
--------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   character_conditions

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/ui

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   model_adapters

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
