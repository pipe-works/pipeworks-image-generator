Character Conditions System
==========================

The Character Conditions system provides structured, rule-based character state generation for procedural content. Unlike simple text file lookups, it uses weighted probability distributions, semantic exclusion rules, and mandatory/optional axis policies to generate coherent character descriptions.

Overview
--------

The system generates character states across multiple axes:

* **Physique**: Body structure (skinny, wiry, stocky, hunched, frail, broad)
* **Wealth**: Economic status (poor, modest, well-kept, wealthy, decadent)
* **Health**: Physical condition (sickly, scarred, weary, hale, limping)
* **Demeanor**: Behavioral presentation (timid, suspicious, resentful, alert, proud)
* **Age**: Life stage (young, middle-aged, old, ancient)

Key Features
-----------

Weighted Probability
^^^^^^^^^^^^^^^^^^^

The system uses realistic population distributions:

.. code-block:: python

   WEIGHTS = {
       "wealth": {
           "poor": 4.0,      # Most common
           "modest": 3.0,
           "well-kept": 2.0,
           "wealthy": 1.0,
           "decadent": 0.5,  # Rare
       }
   }

This creates a believable population where most characters are poor or modest, and wealthy/decadent characters are rare.

Semantic Exclusions
^^^^^^^^^^^^^^^^^^

Rules prevent illogical combinations:

* Decadent characters can't be frail or sickly (wealth enables health care)
* Ancient characters aren't timid (age brings confidence)
* Broad, strong physiques don't pair with sickness
* Hale (healthy) characters shouldn't have frail physiques

Mandatory/Optional Axes
^^^^^^^^^^^^^^^^^^^^^^^

* **Mandatory**: Always include physique and wealth (establish baseline)
* **Optional**: Include 0-2 additional axes (add narrative detail)
* **Max Optional**: Prevents prompt dilution and maintains clarity

Usage
-----

Basic Generation
^^^^^^^^^^^^^^^

.. code-block:: python

   from pipeworks.core.character_conditions import (
       generate_condition,
       condition_to_prompt
   )

   # Generate random condition
   condition = generate_condition()
   prompt_text = condition_to_prompt(condition)

   print(prompt_text)
   # Output: "wiry, poor, suspicious, old"

Reproducible Generation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Use seed for reproducibility
   condition1 = generate_condition(seed=42)
   condition2 = generate_condition(seed=42)

   assert condition1 == condition2  # Same result

Integration with Image Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pipeworks import ImageGenerator
   from pipeworks.core.character_conditions import (
       generate_condition,
       condition_to_prompt
   )

   generator = ImageGenerator()

   # Generate character condition
   condition = generate_condition(seed=42)
   condition_text = condition_to_prompt(condition)

   # Use in prompt
   full_prompt = f"{condition_text}, goblin warrior in dark tavern"
   image = generator.generate(prompt=full_prompt, seed=42)

Dynamic Conditions
^^^^^^^^^^^^^^^^^

Generate different conditions for each run:

.. code-block:: python

   # Generate 10 different goblins
   for i in range(10):
       condition = generate_condition()  # No seed = random
       condition_text = condition_to_prompt(condition)

       prompt = f"{condition_text}, goblin character portrait"
       image, path = generator.generate_and_save(
           prompt=prompt,
           seed=1000 + i
       )

UI Integration
-------------

The Gradio UI provides a "Character Condition Generator" in the Start 1 segment:

1. **Auto-generate condition**: Enable to generate conditions
2. **Generated Condition**: Shows the current condition (editable)
3. **🎲 Regenerate**: Generate a new random condition
4. **Dynamic**: Regenerate condition for each run in batch generation

When Dynamic is enabled:

* Batch Size: 2, Runs: 20 = 20 different conditions (each used for 2 images)
* Each run gets a unique condition, but images within the run share it

Architecture
-----------

Data Structures
^^^^^^^^^^^^^^

.. code-block:: python

   # Axis definitions (single source of truth)
   CONDITION_AXES: dict[str, list[str]]

   # Policy (controls complexity)
   AXIS_POLICY: dict[str, Any] = {
       "mandatory": ["physique", "wealth"],
       "optional": ["health", "demeanor", "age"],
       "max_optional": 2,
   }

   # Weights (population distribution)
   WEIGHTS: dict[str, dict[str, float]]

   # Exclusions (semantic constraints)
   EXCLUSIONS: dict[tuple[str, str], dict[str, list[str]]]

Generation Process
^^^^^^^^^^^^^^^^^

1. **Select mandatory axes**: Always pick physique + wealth
2. **Select optional axes**: Randomly pick 0-2 additional axes
3. **Apply weights**: Use weighted random selection for each axis
4. **Apply exclusions**: Remove illogical combinations

API Reference
------------

.. automodule:: pipeworks.core.character_conditions
   :members:
   :undoc-members:
   :show-inheritance:
