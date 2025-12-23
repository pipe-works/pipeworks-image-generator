"""Base classes and registry for the Pipeworks workflow system.

Workflows define generation strategies for specific content types (characters,
assets, maps, etc.). They combine:
- Model adapter selection (which AI model to use)
- Prompt engineering approach
- Default parameters
- Pre/post processing logic
- UI controls specific to the workflow

Example workflow usage:

    >>> from pipeworks.workflows.base import workflow_registry
    >>> from pipeworks.core import model_registry, config
    >>>
    >>> # Get the workflow
    >>> workflow = workflow_registry.instantiate("Asset Generation")
    >>>
    >>> # Attach the appropriate model adapter
    >>> adapter = model_registry.instantiate(workflow.model_adapter_name, config)
    >>> workflow.set_model_adapter(adapter)
    >>>
    >>> # Generate using workflow
    >>> image, params = workflow.generate(asset_type="sword", style="pixel-art")
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeworks.core.model_adapters import ModelAdapterBase

logger = logging.getLogger(__name__)


class WorkflowBase(ABC):
    """Base class for all Pipeworks workflows.

    Workflows encapsulate generation strategies for specific content types.
    Each workflow defines:
    - Which model adapter to use (text-to-image, image-edit, etc.)
    - Prompt engineering approach
    - Default parameters
    - Pre/post processing logic
    - UI controls specific to the workflow

    Attributes
    ----------
    name : str
        Human-readable name of the workflow
    description : str
        Brief description of the workflow's purpose
    version : str
        Workflow version
    model_adapter_name : str
        Name of the model adapter to use (e.g., "Z-Image-Turbo")
    model_type : str
        Type of model required (text-to-image, image-edit, etc.)
    default_width : int
        Default image width for this workflow
    default_height : int
        Default image height for this workflow
    default_steps : int
        Default inference steps for this workflow
    default_guidance_scale : float
        Default guidance scale for this workflow

    Examples
    --------
    Creating a custom workflow:

        >>> class MyWorkflow(WorkflowBase):
        ...     name = "My Workflow"
        ...     description = "Custom workflow for specific tasks"
        ...     model_adapter_name = "Z-Image-Turbo"
        ...     model_type = "text-to-image"
        ...
        ...     def build_prompt(self, **kwargs) -> str:
        ...         return f"Generate {kwargs['subject']}"
        ...
        >>> # Register and use
        >>> workflow_registry.register(MyWorkflow)
        >>> workflow = workflow_registry.instantiate("My Workflow")
    """

    name: str = "Base Workflow"
    description: str = "Base workflow class"
    version: str = "0.1.0"

    # Model adapter settings
    model_adapter_name: str = "Z-Image-Turbo"  # Default to Z-Image-Turbo
    model_type: str = "text-to-image"  # Expected model type

    # Default generation parameters (can be overridden by subclasses)
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 9
    default_guidance_scale: float = 0.0

    def __init__(self, model_adapter: "ModelAdapterBase | None" = None) -> None:
        """Initialize the workflow.

        Args:
            model_adapter: Model adapter instance to use for generation.
                          If None, must be set later via set_model_adapter()
        """
        self._model_adapter = model_adapter
        logger.info(f"Initialized workflow: {self.name}")
        if model_adapter:
            logger.info(f"Using model adapter: {model_adapter.name}")

    @property
    def model_adapter(self) -> "ModelAdapterBase":
        """Get the model adapter.

        Returns
        -------
        ModelAdapterBase
            The model adapter instance

        Raises
        ------
        RuntimeError
            If no model adapter has been set
        """
        if self._model_adapter is None:
            raise RuntimeError(
                f"No model adapter attached to workflow '{self.name}'. "
                f"Call set_model_adapter() first or pass model_adapter to __init__."
            )
        return self._model_adapter

    def set_model_adapter(self, adapter: "ModelAdapterBase") -> None:
        """Set the model adapter for this workflow.

        Args:
            adapter: Model adapter instance

        Raises
        ------
        TypeError
            If adapter's model_type doesn't match workflow's expected type
        """
        if adapter.model_type != self.model_type:
            logger.warning(
                f"Model adapter type mismatch: workflow '{self.name}' expects "
                f"'{self.model_type}' but got '{adapter.model_type}'. "
                f"This may cause issues."
            )
        self._model_adapter = adapter
        logger.info(f"Set model adapter for '{self.name}': {adapter.name}")

    # Backward compatibility property
    @property
    def generator(self) -> "ModelAdapterBase":
        """Backward compatibility property.

        Returns the model adapter.

        .. deprecated:: 1.0.0
           Use :attr:`model_adapter` instead.
        """
        logger.warning("workflow.generator is deprecated. Use workflow.model_adapter instead.")
        return self.model_adapter

    @abstractmethod
    def build_prompt(self, **kwargs) -> str:
        """
        Build the generation prompt based on workflow-specific parameters.

        Args:
            **kwargs: Workflow-specific parameters

        Returns:
            Formatted prompt string
        """
        pass

    def get_generation_params(self, **kwargs) -> dict[str, Any]:
        """
        Get generation parameters for this workflow.

        Args:
            **kwargs: User-provided parameters

        Returns:
            Dictionary of generation parameters
        """
        return {
            "width": kwargs.get("width", self.default_width),
            "height": kwargs.get("height", self.default_height),
            "num_inference_steps": kwargs.get("num_inference_steps", self.default_steps),
            "guidance_scale": kwargs.get("guidance_scale", self.default_guidance_scale),
            "seed": kwargs.get("seed"),
        }

    def preprocess(self, **kwargs) -> dict[str, Any]:
        """
        Preprocess inputs before generation (optional override).

        Args:
            **kwargs: Workflow inputs

        Returns:
            Processed inputs
        """
        return kwargs

    def postprocess(self, image, **kwargs):
        """
        Postprocess the generated image (optional override).

        Args:
            image: Generated image
            **kwargs: Additional parameters

        Returns:
            Processed image
        """
        return image

    def generate(self, **kwargs):
        """Main generation method that orchestrates the workflow.

        This method coordinates the entire generation process:
        1. Preprocess workflow inputs
        2. Build prompt using workflow-specific logic
        3. Get generation parameters (with workflow defaults)
        4. Generate image using the model adapter
        5. Postprocess the generated image

        Args:
            **kwargs: Workflow-specific parameters

        Returns
        -------
        tuple[Image.Image, dict]
            Tuple of (generated image, generation parameters)

        Raises
        ------
        RuntimeError
            If no model adapter is attached to the workflow
        """
        # Ensure model adapter is set
        model_adapter = self.model_adapter  # Will raise RuntimeError if not set

        # Preprocess inputs
        inputs = self.preprocess(**kwargs)

        # Build prompt
        prompt = self.build_prompt(**inputs)
        logger.info(f"[{self.name}] Generated prompt: {prompt}")

        # Get generation parameters
        params = self.get_generation_params(**inputs)
        params["prompt"] = prompt

        # Generate image using model adapter
        image = model_adapter.generate(**params)

        # Postprocess
        image = self.postprocess(image, **inputs)

        return image, params

    def get_ui_controls(self) -> dict[str, Any]:
        """
        Define workflow-specific UI controls for Gradio.

        Returns:
            Dictionary of control definitions
        """
        return {}


class WorkflowRegistry:
    """Registry for managing available workflows."""

    def __init__(self):
        self._workflows: dict[str, type[WorkflowBase]] = {}
        self._instances: dict[str, WorkflowBase] = {}

    def register(self, workflow_class: type[WorkflowBase]) -> None:
        """
        Register a workflow class.

        Args:
            workflow_class: Workflow class to register
        """
        workflow_name = workflow_class.name
        self._workflows[workflow_name] = workflow_class
        logger.info(f"Registered workflow: {workflow_name}")

    def instantiate(
        self, workflow_name: str, model_adapter: "ModelAdapterBase | None" = None
    ) -> WorkflowBase | None:
        """Create an instance of a registered workflow.

        Args:
            workflow_name: Name of the workflow to instantiate
            model_adapter: Model adapter instance to use (optional, can be set later)

        Returns
        -------
        WorkflowBase | None
            Workflow instance or None if not found

        Notes
        -----
        If model_adapter is None, you must call workflow.set_model_adapter()
        before using the workflow.generate() method.
        """
        if workflow_name not in self._workflows:
            logger.error(f"Workflow not found: {workflow_name}")
            return None

        instance = self._workflows[workflow_name](model_adapter=model_adapter)
        self._instances[workflow_name] = instance
        return instance

    def get_instance(self, workflow_name: str) -> WorkflowBase | None:
        """Get an existing workflow instance."""
        return self._instances.get(workflow_name)

    def list_available(self) -> list[str]:
        """List all registered workflow names."""
        return list(self._workflows.keys())

    def get_workflow_info(self, workflow_name: str) -> dict[str, str] | None:
        """Get information about a workflow."""
        if workflow_name not in self._workflows:
            return None

        workflow_class = self._workflows[workflow_name]
        return {
            "name": workflow_class.name,
            "description": workflow_class.description,
            "version": workflow_class.version,
        }


# Global workflow registry
workflow_registry = WorkflowRegistry()
