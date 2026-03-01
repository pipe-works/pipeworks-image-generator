/**
 * Output Lightbox Controller
 * ---------------------------------------------------------------------------
 * This module isolates all generation-output lightbox navigation behavior so
 * the main frontend bootstrap file does not continue to grow as a monolith.
 *
 * Responsibilities handled here:
 * - render the current lightbox image and metadata
 * - expose vim-style `h/j/k/l` navigation for Output images
 * - provide previous / play / pause / stop / next transport controls
 * - keep autoplay state in sync with the currently available Output images
 * - fall back gracefully when the lightbox is opened from Gallery/Favourites
 *
 * The exported pure helpers are intentionally kept free of DOM access so they
 * can be exercised by lightweight Node-based unit tests without a browser.
 */

/**
 * Playback interval, in milliseconds, for the Output lightbox slideshow.
 *
 * The value is deliberately conservative so users can read the prompt and
 * inspect the image before the next frame advances.
 */
export const OUTPUT_LIGHTBOX_PLAY_INTERVAL_MS = 1800;

/**
 * Resolve a keyboard key into an Output-navigation direction.
 *
 * The requested vim-style bindings are mapped to a linear image sequence:
 * - `h` and `k` move backward
 * - `j` and `l` move forward
 *
 * This keeps all four keys useful inside a responsive, wrapping grid where
 * row boundaries are not stable enough to make true 2D navigation reliable.
 *
 * @param {string} key - Raw `KeyboardEvent.key` value.
 * @returns {-1 | 0 | 1} `-1` for previous, `1` for next, `0` for no action.
 */
export function resolveOutputNavigationDirection(key) {
  const normalisedKey = String(key || "").toLowerCase();

  if (normalisedKey === "h" || normalisedKey === "k") {
    return -1;
  }

  if (normalisedKey === "j" || normalisedKey === "l") {
    return 1;
  }

  return 0;
}

/**
 * Calculate the next wrapped index within a linear image sequence.
 *
 * The lightbox intentionally wraps instead of stopping at the sequence ends so
 * keyboard navigation and autoplay can move continuously through the Output
 * collection without forcing the user to reverse direction at the boundaries.
 *
 * @param {number} currentIndex - Zero-based index of the current image.
 * @param {-1 | 1} direction - Movement direction.
 * @param {number} itemCount - Total number of navigable images.
 * @returns {number} The wrapped target index, or `-1` when no items exist.
 */
export function getWrappedImageIndex(currentIndex, direction, itemCount) {
  if (itemCount <= 0) {
    return -1;
  }

  const safeCurrentIndex = Number.isInteger(currentIndex) && currentIndex >= 0 ? currentIndex : 0;
  return (safeCurrentIndex + direction + itemCount) % itemCount;
}

/**
 * Describe the current transport state for the lightbox.
 *
 * The transport controls only apply to images opened from the Generation tab's
 * Output area.  Gallery and Favourites continue to use the same enlarged view,
 * but their transport controls remain disabled because the requested feature is
 * explicitly scoped to the Output collection.
 *
 * @param {object} params - Transport inputs.
 * @param {string|null} params.context - Current lightbox source context.
 * @param {Array<object>} params.outputImages - Current Output image collection.
 * @param {string|null} params.currentImageId - Image currently shown in the lightbox.
 * @param {boolean} params.isPlaying - Whether autoplay is currently active.
 * @returns {{
 *   isOutputContext: boolean,
 *   totalImages: number,
 *   currentIndex: number,
 *   canNavigate: boolean,
 *   canPlay: boolean,
 *   isPlaying: boolean
 * }}
 * Transport-derived state for button enablement and status text.
 */
export function getOutputTransportState({
  context,
  outputImages,
  currentImageId,
  isPlaying,
}) {
  const totalImages = outputImages.length;
  const isOutputContext = context === "output";
  const currentIndex = isOutputContext
    ? outputImages.findIndex((image) => image.id === currentImageId)
    : -1;
  const hasResolvedCurrentImage = currentIndex !== -1;
  const canNavigate = isOutputContext && hasResolvedCurrentImage && totalImages > 1;

  return {
    isOutputContext,
    totalImages,
    currentIndex,
    canNavigate,
    canPlay: canNavigate,
    isPlaying: canNavigate && isPlaying,
  };
}

/**
 * Create the DOM-backed controller for the enlarged lightbox view.
 *
 * The controller owns only the lightbox-specific state.  The authoritative
 * Output collection stays in the main app state so generation, delete, and
 * favourite flows continue to work from a single source of truth.
 *
 * @param {object} options - Controller dependencies.
 * @param {() => Array<object>} options.getOutputImages - Returns the current Output collection.
 * @param {(image: object | null) => void} options.onImageChange - Sync callback for the host app state.
 * @param {() => void} options.onClose - Called after the lightbox closes.
 * @param {(image: object) => void} options.onToggleFavourite - Favourite callback for the active image.
 * @param {(image: object) => void} options.onDeleteImage - Delete callback for the active image.
 * @param {Document} [options.rootDocument=document] - DOM root to operate against.
 * @param {Window} [options.rootWindow=window] - Window object used for timers.
 * @returns {{
 *   open: ({image: object, context: string}) => void,
 *   close: () => void,
 *   handleKeydown: (event: KeyboardEvent) => boolean,
 *   updateImageState: (imageId: string, patch: object) => void,
 *   handleRemovedImages: (imageIds: Array<string>) => void,
 *   resetOutputCollection: () => void
 * }}
 * Public controller API used by the main application shell.
 */
export function createOutputLightboxController({
  getOutputImages,
  onImageChange,
  onClose,
  onToggleFavourite,
  onDeleteImage,
  rootDocument = document,
  rootWindow = window,
}) {
  /**
   * Cache all lightbox DOM references once so subsequent updates only change
   * text, attributes, and button state rather than repeatedly querying.
   */
  const dom = {
    lightbox: rootDocument.getElementById("lightbox"),
    image: rootDocument.getElementById("lightbox-img"),
    model: rootDocument.getElementById("lb-model"),
    resolution: rootDocument.getElementById("lb-resolution"),
    seed: rootDocument.getElementById("lb-seed"),
    stepsCfg: rootDocument.getElementById("lb-steps-cfg"),
    prompt: rootDocument.getElementById("lb-prompt"),
    favouriteButton: rootDocument.getElementById("lb-btn-fav"),
    downloadButton: rootDocument.getElementById("lb-btn-download"),
    deleteButton: rootDocument.getElementById("lb-btn-delete"),
    previousButton: rootDocument.getElementById("lb-btn-prev"),
    playButton: rootDocument.getElementById("lb-btn-play"),
    pauseButton: rootDocument.getElementById("lb-btn-pause"),
    stopButton: rootDocument.getElementById("lb-btn-stop"),
    nextButton: rootDocument.getElementById("lb-btn-next"),
    closeButton: rootDocument.getElementById("lightbox-close"),
    backdrop: rootDocument.getElementById("lightbox-backdrop"),
    navStatus: rootDocument.getElementById("lb-nav-status"),
    navHint: rootDocument.getElementById("lb-nav-hint"),
  };

  /**
   * Internal controller state kept separate from the host application state.
   *
   * `staticImage` is used for Gallery/Favourites views where the current image
   * does not come from the Output collection.  Output images always resolve
   * dynamically from `getOutputImages()` so navigation reflects the live set.
   */
  const state = {
    isOpen: false,
    context: null,
    currentImageId: null,
    staticImage: null,
    isPlaying: false,
    playbackTimerId: null,
    lastOutputIndex: 0,
  };

  /**
   * Return the image currently shown in the lightbox.
   *
   * Output images are always resolved from the latest Output collection so
   * favourite changes, deletes, and ordering updates flow through naturally.
   *
   * @returns {object | null} The current image record, if any.
   */
  function getCurrentImage() {
    if (state.context === "output") {
      return getOutputImages().find((image) => image.id === state.currentImageId) || null;
    }

    return state.staticImage;
  }

  /**
   * Stop autoplay and clear the active timer safely.
   *
   * The timer is always cleared before the state flag changes so repeated
   * play/pause/stop calls never leave multiple intervals running.
   */
  function clearPlaybackTimer() {
    if (state.playbackTimerId !== null) {
      rootWindow.clearInterval(state.playbackTimerId);
      state.playbackTimerId = null;
    }
  }

  /**
   * Update the transport controls so they reflect the currently navigable
   * Output collection and the active autoplay state.
   */
  function updateTransportControls() {
    const transportState = getOutputTransportState({
      context: state.context,
      outputImages: getOutputImages(),
      currentImageId: state.currentImageId,
      isPlaying: state.isPlaying,
    });

    if (transportState.currentIndex >= 0) {
      state.lastOutputIndex = transportState.currentIndex;
    }

    dom.previousButton.disabled = !transportState.canNavigate;
    dom.nextButton.disabled = !transportState.canNavigate;
    dom.playButton.disabled = !transportState.canPlay;
    dom.pauseButton.disabled = !transportState.isPlaying;
    dom.stopButton.disabled = !transportState.isPlaying;

    dom.playButton.classList.toggle("is-active", transportState.isPlaying);
    dom.pauseButton.classList.toggle("is-active", transportState.isPlaying);
    dom.playButton.setAttribute("aria-pressed", String(transportState.isPlaying));
    dom.pauseButton.setAttribute("aria-pressed", String(transportState.isPlaying));

    if (transportState.isOutputContext && transportState.currentIndex >= 0) {
      dom.navStatus.textContent = `Output ${transportState.currentIndex + 1} / ${transportState.totalImages}`;
      dom.navHint.textContent = "H/K previous · J/L next";
      return;
    }

    dom.navStatus.textContent = "Output navigation unavailable";
    dom.navHint.textContent = "Open an image from Output to use H/J/K/L navigation.";
  }

  /**
   * Render the supplied image into the lightbox UI.
   *
   * Rendering updates both the visual card and the host application's notion
   * of `lightboxImage` through the `onImageChange` callback.
   *
   * @param {object | null} image - Image record to display.
   */
  function renderImage(image) {
    if (!image) {
      close();
      return;
    }

    state.currentImageId = image.id;

    if (state.context !== "output") {
      state.staticImage = { ...image };
    }

    dom.image.src = image.url;
    dom.image.alt = `Generated image — ${image.model_label}`;
    dom.model.textContent = image.model_label;
    dom.resolution.textContent = `${image.width} × ${image.height} px`;
    dom.seed.textContent = String(image.seed);

    const schedulerLabel = image.scheduler ? ` · ${image.scheduler}` : "";
    dom.stepsCfg.textContent = `${image.steps} steps · CFG ${image.guidance}${schedulerLabel}`;
    dom.prompt.textContent = image.compiled_prompt || "—";

    dom.downloadButton.href = image.url;
    dom.downloadButton.download = `pipeworks_${image.id.slice(0, 8)}.png`;

    dom.favouriteButton.textContent = image.is_favourite ? "★ Unfavourite" : "☆ Favourite";
    dom.favouriteButton.classList.toggle("is-active", Boolean(image.is_favourite));

    onImageChange(image);
    updateTransportControls();
  }

  /**
   * Show the lightbox modal container.
   *
   * This function is deliberately separate from `renderImage()` so the modal
   * visibility transition and the content update remain conceptually distinct.
   */
  function showLightbox() {
    dom.lightbox.classList.add("is-open");
    rootDocument.body.style.overflow = "hidden";
    state.isOpen = true;
  }

  /**
   * Move to another image within the live Output collection.
   *
   * @param {-1 | 1} direction - Navigation direction.
   * @returns {boolean} `true` when an Output image was rendered.
   */
  function stepOutputImage(direction) {
    const outputImages = getOutputImages();
    const transportState = getOutputTransportState({
      context: state.context,
      outputImages,
      currentImageId: state.currentImageId,
      isPlaying: state.isPlaying,
    });

    if (!transportState.canNavigate) {
      return false;
    }

    const nextIndex = getWrappedImageIndex(
      transportState.currentIndex,
      direction,
      transportState.totalImages,
    );
    const nextImage = outputImages[nextIndex];

    if (!nextImage) {
      return false;
    }

    renderImage(nextImage);
    return true;
  }

  /**
   * Start slideshow playback across the current Output images.
   *
   * Playback loops continuously through the Output sequence until the user
   * pauses, stops, clears the Output pane, or the lightbox closes.
   */
  function startPlayback() {
    const transportState = getOutputTransportState({
      context: state.context,
      outputImages: getOutputImages(),
      currentImageId: state.currentImageId,
      isPlaying: state.isPlaying,
    });

    if (!transportState.canPlay || state.isPlaying) {
      updateTransportControls();
      return;
    }

    state.isPlaying = true;
    clearPlaybackTimer();

    state.playbackTimerId = rootWindow.setInterval(() => {
      const advanced = stepOutputImage(1);

      if (!advanced) {
        stopPlayback();
      }
    }, OUTPUT_LIGHTBOX_PLAY_INTERVAL_MS);

    updateTransportControls();
  }

  /**
   * Pause playback without changing the current image.
   */
  function pausePlayback() {
    if (!state.isPlaying) {
      return;
    }

    state.isPlaying = false;
    clearPlaybackTimer();
    updateTransportControls();
  }

  /**
   * Stop playback completely while keeping the currently displayed image.
   *
   * Retaining the current image is less surprising than rewinding because the
   * user may stop on a specific frame they want to inspect or download.
   */
  function stopPlayback() {
    state.isPlaying = false;
    clearPlaybackTimer();
    updateTransportControls();
  }

  /**
   * Open the lightbox for the supplied image/context pair.
   *
   * @param {{image: object, context: string}} params - Open request.
   */
  function open({ image, context }) {
    stopPlayback();
    state.context = context;
    state.currentImageId = image.id;
    state.staticImage = context === "output" ? null : { ...image };

    showLightbox();
    renderImage(context === "output" ? getCurrentImage() || image : state.staticImage);
  }

  /**
   * Close the lightbox and clear all controller-owned state.
   */
  function close() {
    stopPlayback();
    dom.lightbox.classList.remove("is-open");
    rootDocument.body.style.overflow = "";

    state.isOpen = false;
    state.context = null;
    state.currentImageId = null;
    state.staticImage = null;

    onImageChange(null);
    onClose();
    updateTransportControls();
  }

  /**
   * Determine whether the current active element should block global hotkeys.
   *
   * Text-entry controls retain priority so the lightbox navigation never steals
   * keystrokes from an input, textarea, select, or contenteditable surface.
   *
   * @returns {boolean} `true` when hotkeys should be ignored.
   */
  function isTypingTargetActive() {
    const activeElement = rootDocument.activeElement;

    if (!activeElement) {
      return false;
    }

    const tagName = activeElement.tagName;
    return (
      tagName === "INPUT"
      || tagName === "TEXTAREA"
      || tagName === "SELECT"
      || activeElement.isContentEditable
    );
  }

  /**
   * Handle document-level keydown events for Output navigation.
   *
   * @param {KeyboardEvent} event - Global keydown event from the host app.
   * @returns {boolean} `true` when the event was consumed by the controller.
   */
  function handleKeydown(event) {
    if (!state.isOpen || event.defaultPrevented || event.altKey || event.ctrlKey || event.metaKey) {
      return false;
    }

    if (isTypingTargetActive()) {
      return false;
    }

    const direction = resolveOutputNavigationDirection(event.key);

    if (direction === 0) {
      return false;
    }

    const advanced = stepOutputImage(direction);

    if (advanced) {
      event.preventDefault();
      return true;
    }

    return false;
  }

  /**
   * Sync the currently displayed image after an external state update such as a
   * favourite toggle.
   *
   * @param {string} imageId - Updated image identifier.
   * @param {object} patch - Partial image fields to merge for non-Output views.
   */
  function updateImageState(imageId, patch) {
    if (state.currentImageId !== imageId) {
      return;
    }

    if (state.context === "output") {
      const currentOutputImage = getCurrentImage();

      if (currentOutputImage) {
        renderImage(currentOutputImage);
      }

      return;
    }

    if (!state.staticImage) {
      return;
    }

    state.staticImage = { ...state.staticImage, ...patch };
    renderImage(state.staticImage);
  }

  /**
   * React to one or more image deletions that happened outside the controller.
   *
   * If the deleted image is the currently visible Output image, the controller
   * advances to the next available Output slot.  If no Output images remain, or
   * the lightbox was opened from a non-Output context, the lightbox closes.
   *
   * @param {Array<string>} imageIds - Deleted image identifiers.
   */
  function handleRemovedImages(imageIds) {
    const removedImageIds = new Set(imageIds);

    if (!removedImageIds.has(state.currentImageId)) {
      return;
    }

    if (state.context !== "output") {
      close();
      return;
    }

    const outputImages = getOutputImages();

    if (outputImages.length === 0) {
      close();
      return;
    }

    const fallbackIndex = Math.min(state.lastOutputIndex, outputImages.length - 1);
    renderImage(outputImages[fallbackIndex]);
  }

  /**
   * Reset the Output-backed transport state after the Output pane is cleared.
   */
  function resetOutputCollection() {
    if (state.context === "output") {
      close();
      return;
    }

    stopPlayback();
  }

  /**
   * Wire lightbox-local button actions once at controller creation time.
   */
  dom.closeButton.addEventListener("click", close);
  dom.backdrop.addEventListener("click", close);
  dom.previousButton.addEventListener("click", () => {
    stepOutputImage(-1);
  });
  dom.nextButton.addEventListener("click", () => {
    stepOutputImage(1);
  });
  dom.playButton.addEventListener("click", startPlayback);
  dom.pauseButton.addEventListener("click", pausePlayback);
  dom.stopButton.addEventListener("click", stopPlayback);
  dom.favouriteButton.addEventListener("click", () => {
    const currentImage = getCurrentImage();

    if (currentImage) {
      onToggleFavourite(currentImage);
    }
  });
  dom.deleteButton.addEventListener("click", () => {
    const currentImage = getCurrentImage();

    if (currentImage) {
      onDeleteImage(currentImage);
    }
  });

  updateTransportControls();

  return {
    open,
    close,
    handleKeydown,
    updateImageState,
    handleRemovedImages,
    resetOutputCollection,
  };
}
