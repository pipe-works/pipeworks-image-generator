/**
 * Collection Lightbox Controller
 * ---------------------------------------------------------------------------
 * This module manages the enlarged-image lightbox for Output, Gallery, and
 * Favourites.  The previous implementation only understood the Output pane,
 * which meant transport controls were disabled when users opened images from
 * Gallery or Favourites.
 *
 * The controller now treats the active collection as an explicit dependency:
 * - Output navigation walks the current Output collection
 * - Gallery navigation walks the current Gallery page
 * - Favourites navigation walks the current Favourites page
 *
 * The exported pure helpers remain DOM-free so they can be exercised by Node
 * unit tests without a browser runtime.
 */

import {
  getCollectionNavigationHint,
  getCollectionStatusText,
  getUnavailableCollectionStatusText,
} from "./gallery-context.mjs";

/**
 * Playback interval, in milliseconds, for the collection lightbox slideshow.
 *
 * The timing intentionally leaves enough room for users to inspect metadata
 * and image details before the next frame advances.
 */
export const OUTPUT_LIGHTBOX_PLAY_INTERVAL_MS = 1800;

/**
 * Resolve a keyboard key into a collection-navigation direction.
 *
 * The requested vim-style bindings are mapped onto a linear sequence:
 * - `h` and `k` move backward
 * - `j` and `l` move forward
 *
 * A linear interpretation is deliberate because the card layout can wrap
 * responsively, so a strict two-dimensional mapping would become unstable.
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
 * Calculate the next wrapped index within a collection sequence.
 *
 * Wrapping is used so the navigation buttons, keyboard shortcuts, and autoplay
 * can cycle continuously through a collection without dead-ending at the
 * boundaries.
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
 * Describe the transport state for the currently active collection.
 *
 * @param {object} params - Transport inputs.
 * @param {string|null} params.context - Active lightbox collection context.
 * @param {Array<object>} params.collectionImages - Images in the active collection.
 * @param {string|null} params.currentImageId - Currently displayed image ID.
 * @param {boolean} params.isPlaying - Whether autoplay is currently active.
 * @returns {{
 *   totalImages: number,
 *   currentIndex: number,
 *   canNavigate: boolean,
 *   canPlay: boolean,
 *   isPlaying: boolean
 * }}
 * Derived transport state for the active collection.
 */
export function getLightboxTransportState({
  context,
  collectionImages,
  currentImageId,
  isPlaying,
}) {
  const totalImages = collectionImages.length;
  const currentIndex = context
    ? collectionImages.findIndex((image) => image.id === currentImageId)
    : -1;
  const hasResolvedCurrentImage = currentIndex !== -1;
  const canNavigate = hasResolvedCurrentImage && totalImages > 1;

  return {
    totalImages,
    currentIndex,
    canNavigate,
    canPlay: canNavigate,
    isPlaying: canNavigate && isPlaying,
  };
}

/**
 * Backward-compatible export retained for the existing Node test import path.
 *
 * The helper is now collection-aware rather than Output-only, but the old name
 * remains available so external imports do not break.
 */
export const getOutputTransportState = getLightboxTransportState;

/**
 * Create the DOM-backed controller for the collection-aware lightbox.
 *
 * @param {object} options - Controller dependencies.
 * @param {(context: string | null) => Array<object>} options.getCollectionImages - Returns
 *     the current image collection for the supplied context.
 * @param {(image: object | null) => void} options.onImageChange - Sync callback for
 *     the host app's current lightbox image state.
 * @param {() => void} options.onClose - Called after the lightbox closes.
 * @param {(image: object) => void} options.onToggleFavourite - Favourite callback.
 * @param {(image: object) => void} options.onDeleteImage - Delete callback.
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
 * Public lightbox controller API.
 */
export function createOutputLightboxController({
  getCollectionImages,
  onImageChange,
  onClose,
  onToggleFavourite,
  onDeleteImage,
  rootDocument = document,
  rootWindow = window,
}) {
  /**
   * Cache lightbox DOM nodes once so subsequent renders only update content
   * and state instead of repeating DOM queries.
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
   * Internal lightbox state.
   *
   * `staticImage` is retained as a defensive fallback for unexpected contexts,
   * but the normal application flow now opens the lightbox only from explicit
   * collection-backed contexts.
   */
  const state = {
    isOpen: false,
    context: null,
    currentImageId: null,
    staticImage: null,
    isPlaying: false,
    playbackTimerId: null,
    lastCollectionIndex: 0,
  };

  /**
   * Resolve the currently active collection from the host application state.
   *
   * @returns {Array<object>} Active collection for the current context.
   */
  function getActiveCollectionImages() {
    if (!state.context) {
      return [];
    }

    const images = getCollectionImages(state.context);
    return Array.isArray(images) ? images : [];
  }

  /**
   * Return the currently visible image record.
   *
   * Collection-backed contexts always resolve against the latest application
   * state so deletes and favourite updates remain visible immediately.
   *
   * @returns {object | null} Current image record, if any.
   */
  function getCurrentImage() {
    const activeCollection = getActiveCollectionImages();

    if (activeCollection.length > 0) {
      return activeCollection.find((image) => image.id === state.currentImageId) || null;
    }

    return state.staticImage;
  }

  /**
   * Clear any running playback timer safely.
   */
  function clearPlaybackTimer() {
    if (state.playbackTimerId !== null) {
      rootWindow.clearInterval(state.playbackTimerId);
      state.playbackTimerId = null;
    }
  }

  /**
   * Update the transport controls so they reflect the current collection.
   */
  function updateTransportControls() {
    const activeCollection = getActiveCollectionImages();
    const transportState = getLightboxTransportState({
      context: state.context,
      collectionImages: activeCollection,
      currentImageId: state.currentImageId,
      isPlaying: state.isPlaying,
    });

    if (transportState.currentIndex >= 0) {
      state.lastCollectionIndex = transportState.currentIndex;
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

    if (transportState.currentIndex >= 0) {
      dom.navStatus.textContent = getCollectionStatusText({
        context: state.context,
        currentIndex: transportState.currentIndex,
        totalImages: transportState.totalImages,
      });
      dom.navHint.textContent = getCollectionNavigationHint();
      return;
    }

    dom.navStatus.textContent = getUnavailableCollectionStatusText(state.context);
    dom.navHint.textContent = "Open an image from a collection to use H/J/K/L navigation.";
  }

  /**
   * Render a specific image into the lightbox UI.
   *
   * @param {object | null} image - Image record to display.
   */
  function renderImage(image) {
    if (!image) {
      close();
      return;
    }

    state.currentImageId = image.id;
    state.staticImage = { ...image };

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
   * Show the lightbox shell.
   */
  function showLightbox() {
    dom.lightbox.classList.add("is-open");
    rootDocument.body.style.overflow = "hidden";
    state.isOpen = true;
  }

  /**
   * Step forward or backward inside the active collection.
   *
   * @param {-1 | 1} direction - Requested movement direction.
   * @returns {boolean} `true` when navigation succeeded.
   */
  function stepCollectionImage(direction) {
    const activeCollection = getActiveCollectionImages();
    const transportState = getLightboxTransportState({
      context: state.context,
      collectionImages: activeCollection,
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
    const nextImage = activeCollection[nextIndex];

    if (!nextImage) {
      return false;
    }

    renderImage(nextImage);
    return true;
  }

  /**
   * Start slideshow playback across the active collection.
   */
  function startPlayback() {
    const transportState = getLightboxTransportState({
      context: state.context,
      collectionImages: getActiveCollectionImages(),
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
      const advanced = stepCollectionImage(1);

      if (!advanced) {
        stopPlayback();
      }
    }, OUTPUT_LIGHTBOX_PLAY_INTERVAL_MS);

    updateTransportControls();
  }

  /**
   * Pause autoplay while keeping the current image visible.
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
   * Stop autoplay entirely while preserving the current image.
   */
  function stopPlayback() {
    state.isPlaying = false;
    clearPlaybackTimer();
    updateTransportControls();
  }

  /**
   * Open the lightbox for a specific image in a specific collection.
   *
   * @param {{image: object, context: string}} params - Open request.
   */
  function open({ image, context }) {
    stopPlayback();
    state.context = context;
    state.currentImageId = image.id;
    state.staticImage = { ...image };

    showLightbox();

    const currentCollectionImage = getCurrentImage();
    renderImage(currentCollectionImage || image);
  }

  /**
   * Close the lightbox and clear its state.
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
   * Determine whether the current focus target should block global hotkeys.
   *
   * @returns {boolean} `true` when keyboard navigation should be ignored.
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
   * Handle document-level keyboard navigation for the active collection.
   *
   * @param {KeyboardEvent} event - Global keydown event from the host app.
   * @returns {boolean} `true` when the event was consumed here.
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

    const advanced = stepCollectionImage(direction);

    if (advanced) {
      event.preventDefault();
      return true;
    }

    return false;
  }

  /**
   * Refresh the current lightbox image after an external state update.
   *
   * @param {string} imageId - Updated image identifier.
   * @param {object} patch - Partial image fields to merge as a fallback.
   */
  function updateImageState(imageId, patch) {
    if (state.currentImageId !== imageId) {
      return;
    }

    const currentImage = getCurrentImage();

    if (currentImage) {
      renderImage(currentImage);
      return;
    }

    if (!state.staticImage) {
      return;
    }

    state.staticImage = { ...state.staticImage, ...patch };
    renderImage(state.staticImage);
  }

  /**
   * React to image removals in the active collection.
   *
   * @param {Array<string>} imageIds - Deleted image identifiers.
   */
  function handleRemovedImages(imageIds) {
    const removedImageIds = new Set(imageIds);

    if (!removedImageIds.has(state.currentImageId)) {
      return;
    }

    const activeCollection = getActiveCollectionImages();

    if (activeCollection.length === 0) {
      close();
      return;
    }

    const fallbackIndex = Math.min(state.lastCollectionIndex, activeCollection.length - 1);
    renderImage(activeCollection[fallbackIndex]);
  }

  /**
   * Reset the lightbox after the Output pane is cleared.
   *
   * Output clearing is a dedicated user action, so the lightbox should close
   * only when it is currently displaying Output images.
   */
  function resetOutputCollection() {
    if (state.context === "output") {
      close();
      return;
    }

    stopPlayback();
  }

  dom.closeButton.addEventListener("click", close);
  dom.backdrop.addEventListener("click", close);
  dom.previousButton.addEventListener("click", () => {
    stepCollectionImage(-1);
  });
  dom.nextButton.addEventListener("click", () => {
    stepCollectionImage(1);
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
