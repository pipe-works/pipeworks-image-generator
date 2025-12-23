Quickstart Guide
===============

This guide will get you up and running with Pipeworks in minutes.

Using the Web UI
---------------

The easiest way to use Pipeworks is through the Gradio web interface:

.. code-block:: bash

   pipeworks

Navigate to ``http://0.0.0.0:7860`` in your browser. You'll see:

1. **Model Selection**: Choose between Z-Image-Turbo (text-to-image) or Qwen-Image-Edit (image editing)
2. **Prompt Builder**: Build complex prompts from files or use direct text input
3. **Generation Parameters**: Width, height, steps, batch size, runs, and seed
4. **Character Conditions**: Auto-generate character states (physique, wealth, etc.)
5. **Generated Images**: View and download your generated images

Using the Python API
-------------------

Text-to-Image
^^^^^^^^^^^^

.. code-block:: python

   from pipeworks import model_registry, config

   # Initialize adapter
   adapter = model_registry.instantiate("Z-Image-Turbo", config)

   # Generate a single image
   image = adapter.generate(
       prompt="a pale purple goblin in a dark tavern, medieval fantasy",
       width=1024,
       height=1024,
       num_inference_steps=9,
       seed=42
   )

   # Save the image
   image.save("goblin.png")

Generate and Save
^^^^^^^^^^^^^^^^

.. code-block:: python

   # Generate and save automatically
   image, path = generator.generate_and_save(
       prompt="a wiry, poor goblin wearing torn clothes",
       seed=42
   )

   print(f"Image saved to: {path}")

Batch Generation
^^^^^^^^^^^^^^^

.. code-block:: python

   # Generate multiple images with sequential seeds
   for i in range(5):
       image, path = generator.generate_and_save(
           prompt=f"goblin warrior variant {i+1}",
           seed=42 + i
       )
       print(f"Generated: {path}")

Using Character Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pipeworks.core.character_conditions import (
       generate_condition,
       condition_to_prompt
   )

   # Generate character condition
   condition = generate_condition(seed=42)
   condition_text = condition_to_prompt(condition)

   # Use in prompt
   image = generator.generate(
       prompt=f"{condition_text}, goblin warrior in armor",
       seed=42
   )

Using the Prompt Builder
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pipeworks.core.prompt_builder import PromptBuilder
   from pathlib import Path

   # Initialize builder
   builder = PromptBuilder(inputs_dir=Path("./inputs"))

   # Build prompt from file
   prompt = builder.build_prompt_from_file(
       file_path=Path("./inputs/characters/goblins.txt"),
       mode="Random Line"
   )

   # Generate image
   image = generator.generate(prompt=prompt)

Next Steps
---------

* Learn about :doc:`character_conditions` for procedural character generation
* Explore the :doc:`api/core` for advanced usage
* Read about the model adapter system in :doc:`model_adapters`
