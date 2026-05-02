/**
 * LoRA Tile Lightbox Controller
 * ---------------------------------------------------------------------------
 * Modal viewer for one LoRA dataset tile at full resolution. Mirrors the
 * gallery lightbox idiom (transport controls, H/J/K/L vim-style navigation,
 * keyboard close, click-backdrop close) but operates against an active
 * LoRA run rather than a gallery collection.
 *
 * The active collection is the manifest's slot_order; navigation cycles
 * through tiles in that order. Action buttons (Regen / Exclude / Download)
 * delegate back to the LoRA dataset controller's existing handlers — the
 * lightbox does not own state, it only displays and dispatches.
 */

const PLAY_INTERVAL_MS = 1800;

function resolveDirection(key) {
  const k = String(key || "").toLowerCase();
  if (k === "h" || k === "k") return -1;
  if (k === "j" || k === "l") return 1;
  return 0;
}

function wrappedIndex(currentIndex, direction, count) {
  if (count <= 0) return -1;
  const safe = Number.isInteger(currentIndex) && currentIndex >= 0 ? currentIndex : 0;
  return (safe + direction + count) % count;
}

export function createLoraTileLightboxController({
  toast,
  onRegen,
  onToggleExcluded,
}) {
  const $ = sel => document.querySelector(sel);

  const dom = {
    lightbox: $("#lora-lightbox"),
    backdrop: $("#lora-lightbox-backdrop"),
    closeButton: $("#lora-lightbox-close"),
    image: $("#lora-lb-img"),
    tile: $("#lora-lb-tile"),
    run: $("#lora-lb-run"),
    resolution: $("#lora-lb-resolution"),
    seed: $("#lora-lb-seed"),
    stepsCfg: $("#lora-lb-steps-cfg"),
    tileText: $("#lora-lb-tile-text"),
    prompt: $("#lora-lb-prompt"),
    copyTileTextButton: $("#lora-lb-btn-copy-tile-text"),
    copyPromptButton: $("#lora-lb-btn-copy-prompt"),
    regenButton: $("#lora-lb-btn-regen"),
    excludeButton: $("#lora-lb-btn-exclude"),
    downloadButton: $("#lora-lb-btn-download"),
    previousButton: $("#lora-lb-btn-prev"),
    nextButton: $("#lora-lb-btn-next"),
    playButton: $("#lora-lb-btn-play"),
    pauseButton: $("#lora-lb-btn-pause"),
    stopButton: $("#lora-lb-btn-stop"),
    navStatus: $("#lora-lb-nav-status"),
    navHint: $("#lora-lb-nav-hint"),
  };

  const state = {
    isOpen: false,
    manifest: null,
    currentSlotKey: null,
    isPlaying: false,
    playbackTimerId: null,
    copyResetTimers: { tileText: null, prompt: null },
  };

  function getOrderedSlots() {
    if (!state.manifest) return [];
    const order = state.manifest.slot_order || [];
    const slots = state.manifest.slots || {};
    return order.map(key => ({ key, slot: slots[key] })).filter(entry => entry.slot);
  }

  function getCurrentEntry() {
    return getOrderedSlots().find(entry => entry.key === state.currentSlotKey) || null;
  }

  function clearPlayback() {
    if (state.playbackTimerId !== null) {
      window.clearInterval(state.playbackTimerId);
      state.playbackTimerId = null;
    }
  }

  function renderTransport() {
    const ordered = getOrderedSlots();
    const currentIndex = ordered.findIndex(e => e.key === state.currentSlotKey);
    const total = ordered.length;
    const canNavigate = currentIndex >= 0 && total > 1;

    dom.previousButton.disabled = !canNavigate;
    dom.nextButton.disabled = !canNavigate;
    dom.playButton.disabled = !canNavigate;
    dom.pauseButton.disabled = !state.isPlaying;
    dom.stopButton.disabled = !state.isPlaying;
    dom.playButton.classList.toggle("is-active", state.isPlaying);
    dom.pauseButton.classList.toggle("is-active", state.isPlaying);

    if (currentIndex >= 0) {
      dom.navStatus.textContent = `Tile ${currentIndex + 1} of ${total}`;
      dom.navHint.textContent = "H or K previous · J or L next";
    } else {
      dom.navStatus.textContent = "Tile navigation unavailable";
      dom.navHint.textContent = "Open a tile from a LoRA run to use H/J/K/L navigation.";
    }
  }

  function renderTile(entry) {
    if (!entry) {
      close();
      return;
    }
    const { key, slot } = entry;
    const manifest = state.manifest;
    state.currentSlotKey = key;

    if (slot.image_filename) {
      dom.image.src = `/api/lora-dataset/runs/${encodeURIComponent(
        manifest.run_id
      )}/files/${encodeURIComponent(slot.image_filename)}`;
      dom.image.alt = slot.tile_label || key;
      dom.downloadButton.style.visibility = "visible";
      dom.downloadButton.href = dom.image.src;
      dom.downloadButton.download = `lora_${manifest.run_id.slice(0, 8)}_${key}.png`;
    } else {
      // Slot has no image yet (pending/failed) — clear the image source so
      // the previous slot's image doesn't ghost into the current view.
      dom.image.removeAttribute("src");
      dom.image.alt = `${slot.tile_label || key} (${slot.status})`;
      dom.downloadButton.style.visibility = "hidden";
    }

    const kindLabel =
      slot.tile_kind === "character_sheet"
        ? "Character View"
        : slot.tile_kind === "facial_expression"
          ? "Facial Expression"
          : slot.tile_kind === "body_action"
            ? "Body Action"
            : "Location";
    dom.tile.textContent = `${slot.tile_label || key}  ·  ${kindLabel}  ·  ${slot.status}`;
    dom.run.textContent = manifest.run_id;
    dom.resolution.textContent = `${manifest.params.width} × ${manifest.params.height} px (${manifest.params.aspect_ratio_id})`;
    dom.seed.textContent = slot.seed != null ? String(slot.seed) : "—";
    const schedulerLabel = manifest.params.scheduler ? ` · ${manifest.params.scheduler}` : "";
    dom.stepsCfg.textContent = `${manifest.params.steps} steps · CFG ${manifest.params.guidance}${schedulerLabel}`;
    dom.tileText.textContent = slot.tile_text || "—";
    dom.prompt.textContent = slot.compiled_prompt || "—";

    dom.regenButton.disabled = manifest.status === "running";
    dom.excludeButton.textContent = slot.excluded ? "✓ Include" : "⊘ Exclude";

    renderTransport();
  }

  function step(direction) {
    const ordered = getOrderedSlots();
    const currentIndex = ordered.findIndex(e => e.key === state.currentSlotKey);
    if (currentIndex < 0 || ordered.length <= 1) return false;
    const nextIndex = wrappedIndex(currentIndex, direction, ordered.length);
    renderTile(ordered[nextIndex]);
    return true;
  }

  function startPlayback() {
    if (state.isPlaying) return;
    const ordered = getOrderedSlots();
    if (ordered.length <= 1) return;
    state.isPlaying = true;
    clearPlayback();
    state.playbackTimerId = window.setInterval(() => {
      if (!step(1)) stopPlayback();
    }, PLAY_INTERVAL_MS);
    renderTransport();
  }

  function pausePlayback() {
    if (!state.isPlaying) return;
    state.isPlaying = false;
    clearPlayback();
    renderTransport();
  }

  function stopPlayback() {
    state.isPlaying = false;
    clearPlayback();
    renderTransport();
  }

  function open(manifest, slotKey) {
    state.manifest = manifest;
    state.currentSlotKey = slotKey;
    state.isPlaying = false;
    clearPlayback();
    dom.lightbox.classList.add("is-open");
    document.body.style.overflow = "hidden";
    state.isOpen = true;
    const entry = getCurrentEntry();
    renderTile(entry);
  }

  function close() {
    state.isOpen = false;
    state.manifest = null;
    state.currentSlotKey = null;
    state.isPlaying = false;
    clearPlayback();
    dom.lightbox.classList.remove("is-open");
    document.body.style.overflow = "";
  }

  // Re-render the lightbox against an updated manifest. Called by the LoRA
  // tab controller after a refresh so the lightbox reflects the latest
  // slot status (e.g. status flipped from running → done after a regen).
  function refresh(manifest) {
    if (!state.isOpen || !manifest) return;
    state.manifest = manifest;
    const entry = getCurrentEntry();
    if (!entry) {
      close();
      return;
    }
    renderTile(entry);
  }

  function isTypingTargetActive() {
    const el = document.activeElement;
    if (!el) return false;
    const tag = el.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
  }

  function handleKeydown(event) {
    if (!state.isOpen || event.defaultPrevented) return false;
    if (event.altKey || event.ctrlKey || event.metaKey) return false;
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      return true;
    }
    if (isTypingTargetActive()) return false;
    const direction = resolveDirection(event.key);
    if (direction === 0) return false;
    if (step(direction)) {
      event.preventDefault();
      return true;
    }
    return false;
  }

  async function copyText(target, kind) {
    const value = target.textContent || "";
    if (!value || value === "—") return;
    try {
      await navigator.clipboard.writeText(value);
    } catch (err) {
      toast(`Copy failed: ${err.message || err}`, "err");
      return;
    }
    const button = kind === "tileText" ? dom.copyTileTextButton : dom.copyPromptButton;
    button.textContent = "Copied";
    if (state.copyResetTimers[kind] !== null) {
      window.clearTimeout(state.copyResetTimers[kind]);
    }
    state.copyResetTimers[kind] = window.setTimeout(() => {
      button.textContent = "Copy";
      state.copyResetTimers[kind] = null;
    }, 1200);
  }

  // --- bind once -----------------------------------------------------------
  dom.closeButton.addEventListener("click", close);
  dom.backdrop.addEventListener("click", close);
  dom.previousButton.addEventListener("click", () => step(-1));
  dom.nextButton.addEventListener("click", () => step(1));
  dom.playButton.addEventListener("click", startPlayback);
  dom.pauseButton.addEventListener("click", pausePlayback);
  dom.stopButton.addEventListener("click", stopPlayback);
  dom.copyTileTextButton.addEventListener("click", () => copyText(dom.tileText, "tileText"));
  dom.copyPromptButton.addEventListener("click", () => copyText(dom.prompt, "prompt"));
  dom.regenButton.addEventListener("click", async () => {
    const entry = getCurrentEntry();
    if (!entry || !state.manifest) return;
    await onRegen(state.manifest.run_id, entry.key);
  });
  dom.excludeButton.addEventListener("click", async () => {
    const entry = getCurrentEntry();
    if (!entry || !state.manifest) return;
    await onToggleExcluded(state.manifest.run_id, entry.key, !entry.slot.excluded);
  });

  return { open, close, refresh, handleKeydown };
}
