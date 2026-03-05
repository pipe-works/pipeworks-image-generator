/**
 * Pipe-Works Image Generator — Frontend Application
 * Vanilla JS, no frameworks. Pipe-Works design system.
 */

import { getImageCardBadgeLabel } from "./gallery-context.mjs";
import {
  formatImageCountLabel,
  formatRunCountLabel,
  resolveGalleryPaginationDirection,
} from "./gallery-navigation.mjs";
import { createOutputLightboxController } from "./output-lightbox.mjs";

"use strict";

const MAX_BATCH_SIZE = 1000;
const COPY_FEEDBACK_MS = 1200;
const SECTION_COLLAPSE_STORAGE_PREFIX = "pw-section-collapsed:";

// ── State ──────────────────────────────────────────────────────────────────────

const State = {
  config: null,
  selectedModel: null,
  prependMode: "template",
  promptMode: "manual",
  appendMode: "template",
  batchSize: 1,
  isGenerating: false,
  currentGenerationId: null,
  stopRequested: false,
  outputImages: [],
  galleryImages: [],
  galleryRuns: [],
  galleryTotalImages: 0,
  favouriteImages: [],
  galleryPage: 1,
  galleryPerPage: 20,
  galleryTotal: 0,
  galleryPages: 1,
  galleryModelFilter: "",
  favPage: 1,
  favTotal: 0,
  favPages: 1,
  lightboxImage: null,
  lightboxContext: null,
  theme: "dark",
  tokenCounts: {
    prepend: 0,
    main: 0,
    append: 0,
    total: 0,
    method: "heuristic",
  },
  selectMode: false,
  selectedIds: new Set(),
};

/**
 * The dedicated lightbox controller is created during initialisation so the
 * main application file can delegate Output navigation and slideshow behavior
 * to a smaller, purpose-built module.
 *
 * @type {ReturnType<typeof createOutputLightboxController> | null}
 */
let outputLightboxController = null;


// ── DOM helpers ────────────────────────────────────────────────────────────────

const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "className") e.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(e.style, v);
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (typeof c === "string") e.appendChild(document.createTextNode(c));
    else if (c) e.appendChild(c);
  }
  return e;
}


// ── Toast notifications ────────────────────────────────────────────────────────

function toast(msg, type = "info", duration = 3000) {
  const container = $("#toast-container");
  const t = el("div", { className: `toast toast--${type}` }, msg);
  container.appendChild(t);
  setTimeout(() => {
    t.style.opacity = "0";
    t.style.transition = "opacity 0.3s";
    setTimeout(() => t.remove(), 300);
  }, duration);
}


// ── Status bar ─────────────────────────────────────────────────────────────────

function setStatus(msg, busy = false) {
  $("#status-text").textContent = msg;
  const dot = $("#status-dot");
  dot.classList.toggle("status-dot--busy", busy);
}

function updateStatusBar() {
  if (State.selectedModel) {
    const m = State.config.models.find(m => m.id === State.selectedModel);
    if (m) $("#status-model").textContent = m.label;
  }
  const seed = $("#inp-seed").value;
  const isRandom = $("#chk-random-seed").checked;
  $("#status-seed").textContent = isRandom ? "seed random" : `seed ${seed || "—"}`;
}


// ── Theme ──────────────────────────────────────────────────────────────────────

function applyTheme(theme) {
  State.theme = theme;
  document.documentElement.setAttribute("data-theme", theme === "light" ? "light" : "");
  $("#btn-theme-toggle").textContent = theme === "light" ? "◑ Dark" : "◑ Light";
  localStorage.setItem("pw-theme", theme);
}

function toggleTheme() {
  applyTheme(State.theme === "dark" ? "light" : "dark");
}

function flashButtonLabel(button, activeLabel, defaultLabel, duration = COPY_FEEDBACK_MS) {
  if (!button) return;

  button.textContent = activeLabel;
  window.setTimeout(() => {
    button.textContent = defaultLabel;
  }, duration);
}

function setSectionCollapsed(section, collapsed) {
  const button = section.querySelector("[data-section-toggle]");
  if (!button) return;

  section.classList.toggle("is-collapsed", collapsed);
  button.setAttribute("aria-expanded", collapsed ? "false" : "true");

  const collapseKey = section.dataset.collapseKey;
  if (collapseKey) {
    localStorage.setItem(
      `${SECTION_COLLAPSE_STORAGE_PREFIX}${collapseKey}`,
      collapsed ? "true" : "false",
    );
  }
}

function toggleSection(section) {
  setSectionCollapsed(section, !section.classList.contains("is-collapsed"));
}

function initializeCollapsibleSections() {
  $$(".ctrl-section[data-collapse-key]").forEach(section => {
    const collapseKey = section.dataset.collapseKey;
    const isCollapsed = localStorage.getItem(
      `${SECTION_COLLAPSE_STORAGE_PREFIX}${collapseKey}`,
    ) === "true";
    setSectionCollapsed(section, isCollapsed);
  });
}


// ── Tab navigation ─────────────────────────────────────────────────────────────

function activateTab(tabId) {
  $$(".tab-nav__item").forEach(b => b.classList.toggle("is-active", b.dataset.tab === tabId));
  $$(".tab-content").forEach(c => c.classList.toggle("is-active", c.id === `tab-${tabId}`));

  if (tabId === "gallery") loadGallery();
  if (tabId === "favourites") loadFavourites();
}


// ── Config loading ─────────────────────────────────────────────────────────────

async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    State.config = await res.json();
    populateControls();
    setStatus("Ready");
  } catch (e) {
    setStatus("Config load failed", false);
    toast(`Failed to load config: ${e.message}`, "err");
  }
}

function populateControls() {
  const cfg = State.config;
  const noneOption = { id: "none", label: "— None —" };

  function populateOptionalSelect(selectEl, options) {
    selectEl.innerHTML = "";

    const normalized = [...options];
    if (!normalized.some(option => option.id === "none")) {
      normalized.push(noneOption);
    }

    normalized.forEach(option => {
      const opt = el("option", { value: option.id }, option.label);
      selectEl.appendChild(opt);
    });

    selectEl.value = "none";
  }

  // Models
  const selModel = $("#sel-model");
  selModel.innerHTML = "";
  cfg.models.forEach(m => {
    const optionAttrs = { value: m.id };
    if (m.is_available === false) optionAttrs.disabled = "disabled";
    const opt = el(
      "option",
      optionAttrs,
      m.is_available === false ? `${m.label} (Unavailable)` : m.label,
    );
    selModel.appendChild(opt);
  });

  const firstAvailableModel = cfg.models.find(m => m.is_available !== false);
  if (firstAvailableModel) {
    selModel.value = firstAvailableModel.id;
  }

  // Gallery model filter
  const selGalleryModel = $("#sel-gallery-model");
  cfg.models.forEach(m => {
    const opt = el("option", { value: m.id }, m.label);
    selGalleryModel.appendChild(opt);
  });

  // Prepend prompts
  const selPrepend = $("#sel-prepend");
  populateOptionalSelect(selPrepend, cfg.prepend_prompts);

  // Automated prompts
  const selAuto = $("#sel-auto-prompt");
  populateOptionalSelect(selAuto, cfg.automated_prompts);

  // Append prompts
  const selAppend = $("#sel-append");
  populateOptionalSelect(selAppend, cfg.append_prompts);

  // Version badge in header (populated from API, single source of truth)
  if (cfg.version) {
    $("#app-version").textContent = `V${cfg.version}`;
  }

  // Trigger model change to populate aspect ratios etc.
  onModelChange();
}


// ── Model change ───────────────────────────────────────────────────────────────

function onModelChange() {
  const modelId = $("#sel-model").value;
  State.selectedModel = modelId;

  const model = State.config.models.find(m => m.id === modelId);
  if (!model) return;

  // Update info card
  $("#model-info-name").textContent = model.label;
  $("#model-info-desc").textContent = model.is_available === false
    ? `${model.description} ${model.unavailable_reason || ""}`.trim()
    : model.description;
  $("#model-info-link").href = model.hf_url;

  // Aspect ratios
  const selAspect = $("#sel-aspect");
  selAspect.innerHTML = "";
  model.aspect_ratios.forEach(ar => {
    const opt = el("option", { value: ar.id }, `${ar.label}  (${ar.width}×${ar.height})`);
    selAspect.appendChild(opt);
  });
  // Set default
  const defaultAr = model.aspect_ratios.find(ar => ar.id === model.default_aspect)
    || model.aspect_ratios[0];
  selAspect.value = defaultAr.id;
  onAspectChange();

  // Steps slider
  const rngSteps = $("#rng-steps");
  rngSteps.min = model.min_steps;
  rngSteps.max = model.max_steps;
  rngSteps.value = model.default_steps;
  $("#lbl-steps").textContent = model.default_steps;

  // Guidance slider
  const rngGuidance = $("#rng-guidance");
  rngGuidance.min = model.min_guidance;
  rngGuidance.max = model.max_guidance;
  rngGuidance.step = model.guidance_step;
  rngGuidance.value = model.default_guidance;
  $("#lbl-guidance").textContent = model.default_guidance.toFixed(1);
  const guidanceWrap = $("#guidance-wrap");
  const hasAdjustableGuidance = !(model.min_guidance === 0 && model.max_guidance === 0);
  guidanceWrap.style.display = hasAdjustableGuidance ? "" : "none";

  // Negative prompt visibility
  const negWrap = $("#negative-prompt-wrap");
  negWrap.style.display = model.supports_negative_prompt ? "" : "none";

  // Scheduler dropdown (only visible for models that declare schedulers)
  const schedWrap = $("#scheduler-wrap");
  const selScheduler = $("#sel-scheduler");
  if (model.schedulers && model.schedulers.length > 0) {
    selScheduler.innerHTML = "";
    model.schedulers.forEach(s => {
      selScheduler.appendChild(el("option", { value: s.id }, s.label));
    });
    selScheduler.value = model.default_scheduler || model.schedulers[0].id;
    schedWrap.style.display = "";
  } else {
    selScheduler.innerHTML = "";
    schedWrap.style.display = "none";
  }

  const generateButton = $("#btn-generate");
  generateButton.disabled = model.is_available === false;

  updateStatusBar();
  if (model.is_available === false && model.unavailable_reason) {
    setStatus(model.unavailable_reason, false);
  }
  updatePromptPreview();
  updateTokenCounters();
}

function onAspectChange() {
  const modelId = State.selectedModel;
  const model = State.config.models.find(m => m.id === modelId);
  if (!model) return;

  const aspectId = $("#sel-aspect").value;
  const ar = model.aspect_ratios.find(a => a.id === aspectId);
  if (ar) {
    $("#lbl-resolution").textContent = `${ar.width} × ${ar.height} px`;
  }
}

function setPrependMode(mode) {
  State.prependMode = mode;
  $("#btn-prepend-template").classList.toggle("is-active", mode === "template");
  $("#btn-prepend-manual").classList.toggle("is-active", mode === "manual");
  $("#prepend-template-wrap").style.display = mode === "template" ? "" : "none";
  $("#prepend-manual-wrap").style.display = mode === "manual" ? "" : "none";
  updateTokenCounters();
  schedulePromptPreview();
}

function setMainPromptMode(mode) {
  State.promptMode = mode;
  $("#btn-mode-manual").classList.toggle("is-active", mode === "manual");
  $("#btn-mode-auto").classList.toggle("is-active", mode === "automated");
  $("#prompt-manual-wrap").style.display = mode === "manual" ? "" : "none";
  $("#prompt-auto-wrap").style.display = mode === "automated" ? "" : "none";
  updateTokenCounters();
  schedulePromptPreview();
}

function setAppendMode(mode) {
  State.appendMode = mode;
  $("#btn-append-template").classList.toggle("is-active", mode === "template");
  $("#btn-append-manual").classList.toggle("is-active", mode === "manual");
  $("#append-template-wrap").style.display = mode === "template" ? "" : "none";
  $("#append-manual-wrap").style.display = mode === "manual" ? "" : "none";
  updateTokenCounters();
  schedulePromptPreview();
}


// ── Token estimation ──────────────────────────────────────────────────────────

function estimateTokens(text) {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return Math.ceil(trimmed.length / 4);
}

/**
 * Resolve the current user-entered text for a given prompt section.
 * Returns the raw text string (not including boilerplate).
 */
function getPromptSectionText(section) {
  if (!State.config) return "";

  if (section === "prepend") {
    if (State.prependMode === "manual") {
      return $("#txt-manual-prepend").value;
    }
    // Template mode: resolve selected preset value.
    const id = $("#sel-prepend").value;
    const preset = State.config.prepend_prompts.find(p => p.id === id);
    return preset ? (preset.value || preset.label) : "";
  }

  if (section === "main") {
    if (State.promptMode === "manual") {
      return $("#txt-manual-prompt").value;
    }
    const id = $("#sel-auto-prompt").value;
    const preset = State.config.automated_prompts.find(p => p.id === id);
    return preset ? (preset.value || preset.label) : "";
  }

  if (section === "append") {
    if (State.appendMode === "manual") {
      return $("#txt-manual-append").value;
    }
    const id = $("#sel-append").value;
    if (!id || id === "none") return "";
    const preset = State.config.append_prompts.find(p => p.id === id);
    return preset ? (preset.value || preset.label) : "";
  }

  return "";
}

function getPromptSectionDisplayName(section) {
  if (section === "prepend") return "prepend style";
  if (section === "main") return "main scene";
  if (section === "append") return "append modifier";
  return "prompt section";
}

async function copyPromptSection(section, button) {
  const text = getPromptSectionText(section);
  if (!text.trim()) {
    toast(`No ${getPromptSectionDisplayName(section)} text to copy`, "info");
    return;
  }

  if (!navigator.clipboard?.writeText) {
    toast("Clipboard copy unavailable", "err");
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    flashButtonLabel(button, "Copied", "Copy");
  } catch (_) {
    toast("Clipboard copy failed", "err");
  }
}

/**
 * Update all token counter elements with current estimates.
 * Per-section counters show user text only.  The total counter
 * reflects the full compiled prompt (including boilerplate).
 */
function updateTokenCounters() {
  if (!State.config || !State.selectedModel) return;

  const model = State.config.models.find(m => m.id === State.selectedModel);
  const maxTokens = model ? (model.max_prompt_tokens || 77) : 77;
  const counts = State.tokenCounts || {};

  // Helper to set counter text and warn/over classes.
  function applyCounter(elId, count, limit) {
    const counter = $(elId);
    if (!counter) return;
    counter.textContent = `${count} / ${limit} tokens`;
    counter.classList.remove("token-counter--warn", "token-counter--over");
    if (count > limit) {
      counter.classList.add("token-counter--over");
    } else if (count > limit * 0.85) {
      counter.classList.add("token-counter--warn");
    }
  }

  applyCounter("#prepend-tokens", counts.prepend || 0, maxTokens);
  applyCounter("#main-tokens", counts.main || 0, maxTokens);
  applyCounter("#append-tokens", counts.append || 0, maxTokens);
  applyCounter("#total-tokens", counts.total || 0, maxTokens);
}


// ── Prompt preview ─────────────────────────────────────────────────────────────

let _previewDebounce = null;

function schedulePromptPreview() {
  clearTimeout(_previewDebounce);
  _previewDebounce = setTimeout(updatePromptPreview, 400);
}

async function updatePromptPreview() {
  if (!State.config) return;

  const payload = buildGeneratePayload();
  if (!payload) {
    State.tokenCounts = { prepend: 0, main: 0, append: 0, total: 0, method: "heuristic" };
    updateTokenCounters();
    return;
  }

  try {
    const res = await fetch("/api/prompt/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) return;
    const data = await res.json();
    State.tokenCounts = data.token_counts || {
      prepend: estimateTokens(getPromptSectionText("prepend")),
      main: estimateTokens(getPromptSectionText("main")),
      append: estimateTokens(getPromptSectionText("append")),
      total: estimateTokens(data.compiled_prompt || ""),
      method: "heuristic",
    };
    updateTokenCounters();
  } catch (_) {
    // Silent fail for preview
  }
}


// ── Build generate payload ─────────────────────────────────────────────────────

function buildGeneratePayload() {
  if (!State.config || !State.selectedModel) return null;

  const model = State.config.models.find(m => m.id === State.selectedModel);
  if (!model) return null;

  const aspectId = $("#sel-aspect").value;
  const ar = model.aspect_ratios.find(a => a.id === aspectId);
  if (!ar) return null;

  const isRandom = $("#chk-random-seed").checked;
  const seedVal = isRandom ? null : parseInt($("#inp-seed").value, 10) || null;

  const payload = {
    model_id: State.selectedModel,
    prepend_mode: State.prependMode,
    prompt_mode: State.promptMode,
    append_mode: State.appendMode,
    aspect_ratio_id: aspectId,
    width: ar.width,
    height: ar.height,
    steps: parseInt($("#rng-steps").value, 10),
    guidance: parseFloat($("#rng-guidance").value),
    seed: seedVal,
    batch_size: State.batchSize,
  };

  // Prepend: template or manual
  if (State.prependMode === "manual") {
    payload.manual_prepend = $("#txt-manual-prepend").value.trim();
  } else {
    const prependId = $("#sel-prepend").value;
    payload.prepend_prompt_id = prependId || "none";
  }

  // Main scene: manual or automated
  if (State.promptMode === "manual") {
    payload.manual_prompt = $("#txt-manual-prompt").value.trim();
  } else {
    const automatedId = $("#sel-auto-prompt").value;
    if (automatedId && automatedId !== "none") {
      payload.automated_prompt_id = automatedId;
    }
  }

  // Append: template or manual
  if (State.appendMode === "manual") {
    payload.manual_append = $("#txt-manual-append").value.trim();
  } else {
    const appendId = $("#sel-append").value;
    if (appendId && appendId !== "none") {
      payload.append_prompt_id = appendId;
    }
  }

  if (model.supports_negative_prompt) {
    payload.negative_prompt = $("#txt-negative-prompt").value.trim() || null;
  }

  // Scheduler (only included when the model supports it)
  if (model.schedulers && model.schedulers.length > 0) {
    const schedVal = $("#sel-scheduler").value;
    if (schedVal) payload.scheduler = schedVal;
  }

  return payload;
}


// ── Generate ───────────────────────────────────────────────────────────────────

async function generate() {
  if (State.isGenerating) return;

  const payload = buildGeneratePayload();
  if (!payload) {
    toast("Please finish configuring the generator", "warn");
    return;
  }

  const model = State.config.models.find(m => m.id === State.selectedModel);
  if (model && model.is_available === false) {
    toast(model.unavailable_reason || "Selected model is unavailable in this runtime", "err");
    setStatus("Selected model unavailable", false);
    return;
  }

  State.isGenerating = true;
  State.stopRequested = false;
  State.currentGenerationId = globalThis.crypto?.randomUUID?.() || `gen-${Date.now()}`;
  const btn = $("#btn-generate");
  const stopBtn = $("#btn-stop-generation");
  btn.classList.add("is-loading");
  btn.disabled = true;
  btn.textContent = "◆ Generating…";
  stopBtn.style.display = "";
  stopBtn.disabled = false;
  stopBtn.textContent = "■ Stop After Current Image";

  const progressWrap = $("#progress-wrap");
  progressWrap.style.display = "";

  setStatus(`Generating ${State.batchSize} image(s)…`, true);

  try {
    payload.generation_id = State.currentGenerationId;
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    // Update seed display
    if (payload.seed === null) {
      $("#inp-seed").placeholder = `Last: ${data.batch_seed}`;
    }
    $("#status-seed").textContent = `seed ${data.batch_seed}`;

    // Add images to output
    const placeholder = $("#gen-placeholder");
    if (placeholder) placeholder.remove();

    data.images.forEach(img => {
      State.outputImages.unshift(img);
      const card = createImageCard(img, "output");
      const canvas = $("#gen-canvas");
      canvas.insertBefore(card, canvas.firstChild);
    });

    updateOutputCount();
    if (data.cancelled) {
      const completed = data.completed_count || data.images.length;
      if (completed > 0) {
        toast(`Stopped after ${completed} image(s)`, "info");
        setStatus(`Stopped after ${completed} image(s)`, false);
      } else {
        toast("Stopped before any images completed", "info");
        setStatus("Stopped before any images completed", false);
      }
    } else {
      toast(`Generated ${data.images.length} image(s)`, "ok");
      setStatus(`Done — ${data.images.length} image(s) generated`);
    }

  } catch (e) {
    toast(`Generation failed: ${e.message}`, "err");
    setStatus("Generation failed", false);
  } finally {
    State.isGenerating = false;
    State.currentGenerationId = null;
    State.stopRequested = false;
    btn.classList.remove("is-loading");
    btn.disabled = false;
    btn.textContent = "◆ Generate";
    stopBtn.style.display = "none";
    stopBtn.disabled = true;
    stopBtn.textContent = "■ Stop After Current Image";
    progressWrap.style.display = "none";
  }
}

async function stopGeneration() {
  if (!State.isGenerating || !State.currentGenerationId || State.stopRequested) return;

  State.stopRequested = true;
  const stopBtn = $("#btn-stop-generation");
  stopBtn.disabled = true;
  stopBtn.textContent = "■ Stopping…";

  setStatus("Stopping after current image…", true);

  try {
    const res = await fetch("/api/generate/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ generation_id: State.currentGenerationId }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    toast("Stop requested. Waiting for the current image to finish.", "info");
  } catch (e) {
    State.stopRequested = false;
    stopBtn.disabled = false;
    stopBtn.textContent = "■ Stop After Current Image";
    toast(`Stop request failed: ${e.message}`, "err");
    setStatus("Stop request failed", false);
  }
}


// ── Image card ─────────────────────────────────────────────────────────────────

/**
 * Create an image card for one of the frontend image collections.
 *
 * The card UI is shared between Output, Gallery, and Favourites, but the
 * badge numbering rules differ by context.  Output keeps the original
 * generation batch position, while Gallery and Favourites show the image's
 * current position inside the visible collection.
 *
 * @param {object} img - Image metadata record returned by the backend.
 * @param {string} [context="gallery"] - Collection context for the card.
 * @param {number | null} [collectionIndex=null] - One-based position within
 *     the currently rendered collection, when the collection uses positional
 *     numbering.
 * @returns {HTMLDivElement} Fully wired card element.
 */
function createImageCard(img, context = "gallery", collectionIndex = null) {
  const card = el("div", {
    className: `img-card${img.is_favourite ? " is-favourite" : ""}`,
    style: { width: "200px" },
    "data-id": img.id,
  });

  const image = el("img", {
    className: "img-card__image",
    src: img.url,
    alt: `Generated image — ${img.model_label}`,
    loading: "lazy",
    style: { width: "200px", height: "150px", objectFit: "cover" },
  });

  const star = el("div", { className: "img-card__star" }, "★");

  const badgeLabel = getImageCardBadgeLabel({
    context,
    image: img,
    collectionIndex,
  });
  const batchBadge = el("div", { className: "img-card__batch-badge" }, badgeLabel || "");

  const overlay = el("div", { className: "img-card__overlay" });

  const meta = el("div", { className: "img-card__meta" },
    `${img.model_label} · ${img.width}×${img.height}`
  );

  const favBtn = el("button", {
    className: `img-card__fav-btn${img.is_favourite ? " is-active" : ""}`,
    title: img.is_favourite ? "Remove from favourites" : "Add to favourites",
    onclick: (e) => {
      e.stopPropagation();
      toggleFavourite(img.id, !img.is_favourite, card, favBtn);
    },
  }, img.is_favourite ? "★" : "☆");

  const delBtn = el("button", {
    className: "img-card__del-btn",
    title: "Delete image",
    onclick: (e) => {
      e.stopPropagation();
      deleteImage(img.id, card);
    },
  }, "✕");

  overlay.appendChild(meta);
  overlay.appendChild(favBtn);
  overlay.appendChild(delBtn);

  // Selection checkbox indicator (visible only in select mode via CSS).
  const check = el("div", { className: "img-card__check" });

  card.appendChild(image);
  card.appendChild(star);
  card.appendChild(batchBadge);
  card.appendChild(check);
  card.appendChild(overlay);

  card.addEventListener("click", () => {
    if (State.selectMode) {
      toggleCardSelection(img.id, card);
    } else {
      openLightbox(img, context);
    }
  });

  return card;
}


// ── Collection state synchronisation ──────────────────────────────────────────

/**
 * Apply a partial image update across every in-memory image collection.
 *
 * Output, Gallery, and Favourites are loaded separately, but they can all
 * contain the same image records.  Patching all collections together keeps
 * cards and the lightbox in sync immediately after local actions such as a
 * favourite toggle.
 *
 * @param {string} imageId - Identifier of the image to update.
 * @param {object} patch - Partial image fields to merge into matching items.
 */
function patchImageAcrossCollections(imageId, patch) {
  /**
   * Patch a single image collection immutably.
   *
   * @param {Array<object>} images - Source collection to patch.
   * @returns {Array<object>} Collection with the patch applied.
   */
  function patchCollection(images) {
    return images.map(image => (image.id === imageId ? { ...image, ...patch } : image));
  }

  State.outputImages = patchCollection(State.outputImages);
  State.galleryImages = patchCollection(State.galleryImages);
  State.favouriteImages = patchCollection(State.favouriteImages);
}

/**
 * Remove one or more images from every in-memory image collection.
 *
 * Delete actions can start from any tab or from the lightbox itself.  Removing
 * the deleted IDs from all known collections ensures badge numbering, card
 * lists, and lightbox transport state stay aligned without waiting for a
 * manual refresh.
 *
 * @param {Array<string>} imageIds - Identifiers to remove from all collections.
 */
function removeImagesAcrossCollections(imageIds) {
  const removedImageIds = new Set(imageIds);

  /**
   * Filter a collection down to only still-existing images.
   *
   * @param {Array<object>} images - Source collection.
   * @returns {Array<object>} Collection without the removed IDs.
   */
  function filterCollection(images) {
    return images.filter(image => !removedImageIds.has(image.id));
  }

  State.outputImages = filterCollection(State.outputImages);
  State.galleryImages = filterCollection(State.galleryImages);
  State.favouriteImages = filterCollection(State.favouriteImages);
}


// ── Favourite toggle ───────────────────────────────────────────────────────────

async function toggleFavourite(imageId, isFav, card, btn) {
  try {
    const res = await fetch("/api/gallery/favourite", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: imageId, is_favourite: isFav }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    card.classList.toggle("is-favourite", isFav);
    btn.classList.toggle("is-active", isFav);
    btn.textContent = isFav ? "★" : "☆";
    btn.title = isFav ? "Remove from favourites" : "Add to favourites";

    patchImageAcrossCollections(imageId, { is_favourite: isFav });

    // Keep the dedicated lightbox controller in sync regardless of which
    // surface initiated the favourite change.
    if (outputLightboxController) {
      outputLightboxController.updateImageState(imageId, { is_favourite: isFav });
    }

    toast(isFav ? "Added to favourites" : "Removed from favourites", "ok", 1500);
  } catch (e) {
    toast(`Failed to update favourite: ${e.message}`, "err");
  }
}


// ── Delete image ───────────────────────────────────────────────────────────────

async function deleteImage(imageId, card) {
  if (!confirm("Delete this image? This cannot be undone.")) return;

  try {
    const res = await fetch(`/api/gallery/${imageId}`, { method: "DELETE" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    card.style.opacity = "0";
    card.style.transition = "opacity 0.3s";
    setTimeout(() => card.remove(), 300);

    removeImagesAcrossCollections([imageId]);
    updateOutputCount();

    if (outputLightboxController) {
      outputLightboxController.handleRemovedImages([imageId]);
    }

    await refreshGalleryCollectionsAfterDelete();

    toast("Image deleted", "ok", 1500);
  } catch (e) {
    toast(`Failed to delete: ${e.message}`, "err");
  }
}


// ── Bulk selection mode ────────────────────────────────────────────────────

function toggleSelectMode() {
  State.selectMode = !State.selectMode;
  State.selectedIds.clear();

  const btn = $("#btn-gallery-select");
  const controls = $("#gallery-select-controls");
  const grid = $("#gallery-grid");

  btn.textContent = State.selectMode ? "✕ Cancel" : "☐ Select";
  controls.classList.toggle("is-active", State.selectMode);
  grid.classList.toggle("gallery-grid--selecting", State.selectMode);

  // Clear any existing selections from cards.
  $$(".img-card.is-selected", grid).forEach(c => c.classList.remove("is-selected"));

  updateSelectionUI();
}

function toggleCardSelection(imgId, card) {
  if (State.selectedIds.has(imgId)) {
    State.selectedIds.delete(imgId);
    card.classList.remove("is-selected");
  } else {
    State.selectedIds.add(imgId);
    card.classList.add("is-selected");
  }
  updateSelectionUI();
}

function selectAllVisible() {
  const grid = $("#gallery-grid");
  $$(".img-card", grid).forEach(card => {
    const id = card.getAttribute("data-id");
    if (id && !State.selectedIds.has(id)) {
      State.selectedIds.add(id);
      card.classList.add("is-selected");
    }
  });
  updateSelectionUI();
}

function updateSelectionUI() {
  const count = State.selectedIds.size;
  $("#lbl-select-count").textContent = `${count} selected`;
  $("#btn-delete-selected").disabled = count === 0;
}

async function bulkDelete() {
  const count = State.selectedIds.size;
  if (count === 0) return;

  if (!confirm(`Delete ${count} image${count !== 1 ? "s" : ""}? This cannot be undone.`)) return;

  try {
    const res = await fetch("/api/gallery/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_ids: [...State.selectedIds] }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    removeImagesAcrossCollections(data.deleted);

    // Let the dedicated lightbox controller decide whether to close or
    // advance to the nearest surviving Output image.
    if (outputLightboxController) {
      outputLightboxController.handleRemovedImages(data.deleted);
    }

    await refreshGalleryCollectionsAfterDelete();

    toast(`Deleted ${data.deleted.length} image${data.deleted.length !== 1 ? "s" : ""}`, "ok");

    // Exit select mode and reload gallery.
    toggleSelectMode();
    loadGallery(State.galleryPage);

  } catch (e) {
    toast(`Bulk delete failed: ${e.message}`, "err");
  }
}


// ── Output count ───────────────────────────────────────────────────────────────

function updateOutputCount() {
  const count = $("#gen-canvas").querySelectorAll(".img-card").length;
  $("#lbl-output-count").textContent = formatImageCountLabel(count);
}


// ── Lightbox ───────────────────────────────────────────────────────────────────

function openLightbox(img, context = "gallery") {
  State.lightboxContext = context;
  if (!outputLightboxController) return;
  outputLightboxController.open({ image: img, context });
}

function closeLightbox() {
  State.lightboxContext = null;
  if (!outputLightboxController) return;
  outputLightboxController.close();
}

/**
 * Find the visible card node that corresponds to the active lightbox context.
 *
 * The same image can appear in multiple tabs at once, for example in Gallery
 * and Favourites.  Lightbox actions should update the card in the collection
 * the user is actively browsing rather than whichever duplicate happens to
 * appear first in document order.
 *
 * @param {string} imageId - Identifier of the image to locate.
 * @returns {Element | null} Matching card from the active collection, if any.
 */
function findLightboxContextCard(imageId) {
  if (State.lightboxContext === "output") {
    return $("#gen-canvas")?.querySelector(`.img-card[data-id="${imageId}"]`) || null;
  }

  if (State.lightboxContext === "gallery") {
    return $("#gallery-grid")?.querySelector(`.img-card[data-id="${imageId}"]`) || null;
  }

  if (State.lightboxContext === "favourites") {
    return $("#fav-grid")?.querySelector(`.img-card[data-id="${imageId}"]`) || null;
  }

  return document.querySelector(`.img-card[data-id="${imageId}"]`);
}

/**
 * Refresh gallery-derived collections after a delete mutation.
 *
 * Gallery and Favourites counts are authoritative on the backend because the
 * server may also prune stale metadata for files that were deleted directly
 * from the gallery directory.  Reloading both collections after a delete keeps
 * count badges, page counts, model filters, and navigation state aligned with
 * the reconciled backend view.
 */
async function refreshGalleryCollectionsAfterDelete() {
  await Promise.all([
    loadGallery(State.galleryPage),
    loadFavourites(State.favPage),
  ]);
}

/**
 * Determine whether the user is currently typing into a text-entry control.
 *
 * Global keyboard shortcuts should never steal keystrokes from inputs,
 * textareas, selects, or contenteditable elements.
 *
 * @returns {boolean} `true` when a text-entry element currently owns focus.
 */
function isTypingTargetActive() {
  const activeElement = document.activeElement;

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
 * Return the identifier of the currently active top-level tab.
 *
 * @returns {string | null} Active tab identifier, such as `gallery`.
 */
function getActiveTabId() {
  return $(".tab-nav__item.is-active")?.dataset.tab || null;
}


// ── Gallery ────────────────────────────────────────────────────────────────────

/**
 * Load one page of the gallery grouped by generation runs.
 *
 * Fetches runs from the backend and renders them as grouped sections with
 * date headings, model labels, thumbnail rows, and "Save All" buttons.
 *
 * @param {number} [page=1] - One-based page of runs to load.
 */
async function loadGallery(page = 1) {
  State.galleryPage = page;

  // Exit select mode when gallery reloads (e.g. filter change, page change).
  if (State.selectMode) toggleSelectMode();

  const modelFilter = $("#sel-gallery-model").value;

  try {
    let url = `/api/gallery/runs?page=${page}&per_page=${State.galleryPerPage}`;
    if (modelFilter) url += `&model_id=${encodeURIComponent(modelFilter)}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    State.galleryTotal = data.total_runs;
    State.galleryTotalImages = data.total_images;
    State.galleryPage = data.page;
    State.galleryPages = data.pages;
    State.galleryRuns = data.runs;

    // Build a flat image list from all visible runs for lightbox navigation.
    State.galleryImages = data.runs.flatMap(run => run.images);

    const grid = $("#gallery-grid");
    grid.innerHTML = "";
    grid.classList.add("gallery-grid--runs");

    if (data.runs.length === 0) {
      grid.classList.remove("gallery-grid--runs");
      const empty = el("div", { className: "gallery-empty" },
        el("div", { style: { fontSize: "2rem", opacity: "0.3" } }, "◈"),
        el("div", {}, "No images found"),
        el("div", { className: "u-muted", style: { fontSize: "var(--text-xs)" } },
          modelFilter ? "Try a different filter" : "Generate some images first"
        )
      );
      grid.appendChild(empty);
    } else {
      let lastDate = null;
      let isFirst = true;
      data.runs.forEach(run => {
        // Insert date heading when the date changes.
        if (run.date !== lastDate) {
          lastDate = run.date;
          grid.appendChild(createRunDateHeading(run.date, run.created_at));
        }
        // Expand the first (most recent) run by default.
        grid.appendChild(createRunGroup(run, isFirst));
        isFirst = false;
      });
    }

    $("#lbl-gallery-count").textContent = formatRunCountLabel(data.total_runs, data.total_images);
    $("#lbl-gallery-page").textContent = `Page ${data.page} of ${data.pages}`;
    $("#btn-gallery-prev").disabled = data.page <= 1;
    $("#btn-gallery-next").disabled = data.page >= data.pages;

  } catch (e) {
    toast(`Gallery load failed: ${e.message}`, "err");
  }
}

/**
 * Create a date heading element for the run-grouped gallery.
 *
 * @param {string} dateStr - ISO date string (e.g. "2026-03-05").
 * @param {number} timestamp - Unix timestamp to derive a human-readable date.
 * @returns {HTMLElement} Date heading element.
 */
function createRunDateHeading(dateStr, timestamp) {
  const date = new Date(timestamp * 1000);
  const label = date.toLocaleDateString(undefined, {
    weekday: "short",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
  return el("div", { className: "run-date-heading" }, label);
}

/**
 * Create a DOM group for one generation run.
 *
 * Renders a clickable header bar with chevron, timestamp, model, image count,
 * and a "Save All" button.  The thumbnail row is hidden by default and
 * revealed when the header is clicked (expand/collapse toggle).
 *
 * @param {object} run - Run object from the API response.
 * @param {boolean} [expanded=false] - Whether to start expanded.
 * @returns {HTMLElement} The run group container element.
 */
function createRunGroup(run, expanded = false) {
  const RUN_PAGE_SIZE = 20;
  const group = el("div", { className: `run-group${expanded ? " is-expanded" : ""}` });

  // Internal state for lazy-loaded full run data.
  let allImages = null;
  let runPage = 1;
  let loaded = false;

  // Header bar — clickable to toggle expand/collapse.
  const header = el("div", { className: "run-group__header" });

  const chevron = el("span", { className: "run-group__chevron" }, "\u25B6");

  const time = new Date(run.created_at * 1000);
  const timeLabel = time.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

  header.appendChild(chevron);
  header.appendChild(el("span", { className: "run-group__time" }, timeLabel));
  header.appendChild(el("span", { className: "run-group__model" }, run.model_label));
  header.appendChild(
    el("span", { className: "run-group__count" },
      `${run.total_images} image${run.total_images !== 1 ? "s" : ""}`)
  );
  header.appendChild(el("span", { className: "run-group__spacer" }));

  const zipBtn = el("button", {
    className: "btn btn--secondary btn--sm",
    onclick: (e) => {
      e.stopPropagation();
      downloadRunZip(run.batch_seed, zipBtn);
    },
  }, "Save All");
  header.appendChild(zipBtn);

  header.addEventListener("click", (e) => {
    // Don't toggle if clicking the Save All button.
    if (e.target.closest("button")) return;
    const willExpand = !group.classList.contains("is-expanded");
    group.classList.toggle("is-expanded");
    if (willExpand && !loaded) loadFullRun();
  });

  group.appendChild(header);

  // Thumbnail row — hidden until expanded via CSS.
  const imagesRow = el("div", { className: "run-group__images" });

  // Pagination nav — shown below images when run has >20 images.
  const paginationNav = el("div", { className: "run-group__pagination" });

  /**
   * Render a page of images into the images row.
   */
  function renderRunPage() {
    imagesRow.innerHTML = "";
    const totalPages = Math.ceil(allImages.length / RUN_PAGE_SIZE);
    const start = (runPage - 1) * RUN_PAGE_SIZE;
    const pageImages = allImages.slice(start, start + RUN_PAGE_SIZE);

    pageImages.forEach((img, index) => {
      const collectionIndex = start + index + 1;
      imagesRow.appendChild(createImageCard(img, "gallery", collectionIndex));
    });

    // Update pagination controls.
    if (totalPages > 1) {
      paginationNav.classList.add("is-visible");
      paginationNav.innerHTML = "";

      const prevBtn = el("button", {
        className: "btn btn--secondary btn--sm",
        disabled: runPage <= 1,
        onclick: () => { runPage--; renderRunPage(); },
      }, "\u2190 Prev");

      const nextBtn = el("button", {
        className: "btn btn--secondary btn--sm",
        disabled: runPage >= totalPages,
        onclick: () => { runPage++; renderRunPage(); },
      }, "Next \u2192");

      // Page number buttons.
      const pageNumbers = el("span", { className: "run-group__page-numbers" });
      for (let p = 1; p <= totalPages; p++) {
        const pageBtn = el("button", {
          className: `btn btn--sm ${p === runPage ? "btn--primary" : "btn--ghost"}`,
          onclick: () => { runPage = p; renderRunPage(); },
        }, String(p));
        pageNumbers.appendChild(pageBtn);
      }

      paginationNav.appendChild(prevBtn);
      paginationNav.appendChild(pageNumbers);
      paginationNav.appendChild(nextBtn);
    } else {
      paginationNav.classList.remove("is-visible");
    }
  }

  /**
   * Fetch all images for this run from the API and render them.
   */
  async function loadFullRun() {
    const needsFetch = run.total_images > run.thumbnail_count;
    if (!needsFetch) {
      // All images already present in the initial payload.
      allImages = run.images;
      loaded = true;
      renderRunPage();
      return;
    }

    // Show loading indicator.
    imagesRow.innerHTML = "";
    imagesRow.appendChild(
      el("div", { className: "run-group__loading" }, "Loading all images\u2026")
    );

    try {
      const res = await fetch(`/api/gallery/runs/${run.batch_seed}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      allImages = data.images;
      loaded = true;

      // Update the flat gallery images list so lightbox navigation includes
      // all images from this run.
      const existingIds = new Set(State.galleryImages.map(i => i.id));
      for (const img of allImages) {
        if (!existingIds.has(img.id)) State.galleryImages.push(img);
      }

      renderRunPage();
    } catch (e) {
      imagesRow.innerHTML = "";
      imagesRow.appendChild(
        el("div", { className: "run-group__loading" }, `Failed to load: ${e.message}`)
      );
    }
  }

  // Initial render: show thumbnails or load full run if auto-expanded.
  if (expanded) {
    loadFullRun();
  } else {
    // Render thumbnails for collapsed preview (shown when expanded later
    // only if all images fit in the initial payload).
    allImages = run.images;
    run.images.forEach((img, index) => {
      const collectionIndex = index + 1;
      imagesRow.appendChild(createImageCard(img, "gallery", collectionIndex));
    });

    // "+N more" overflow indicator for thumbnail preview.
    const overflow = run.total_images - run.thumbnail_count;
    if (overflow > 0) {
      const overflowEl = el("div", { className: "run-group__overflow" }, `+${overflow} more`);
      imagesRow.appendChild(overflowEl);
    }
  }

  group.appendChild(imagesRow);
  group.appendChild(paginationNav);

  return group;
}

/**
 * Download a zip of all images in a generation run.
 *
 * @param {number|string} batchSeed - The batch_seed identifying the run.
 * @param {HTMLElement} [btn] - Optional button to show feedback on.
 */
async function downloadRunZip(batchSeed, btn) {
  const originalText = btn ? btn.textContent : "";
  if (btn) {
    btn.textContent = "Downloading…";
    btn.disabled = true;
  }

  try {
    const res = await fetch(`/api/gallery/runs/${batchSeed}/zip`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pipeworks_run_${batchSeed}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    toast("Run zip downloaded", "ok", 1500);
  } catch (e) {
    toast(`Download failed: ${e.message}`, "err");
  } finally {
    if (btn) {
      btn.textContent = originalText;
      btn.disabled = false;
    }
  }
}

/**
 * Load one page of the favourites collection and render it into the grid.
 *
 * The in-memory `State.favouriteImages` array mirrors the rendered favourites
 * page so lightbox navigation stays aligned with the visible favourites list.
 *
 * @param {number} [page=1] - One-based favourites page to load.
 */
async function loadFavourites(page = 1) {
  State.favPage = page;

  try {
    const url = `/api/gallery?page=${page}&per_page=${State.galleryPerPage}&favourites_only=true`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    State.favTotal = data.total;
    State.favPage = data.page;
    State.favPages = data.pages;
    State.favouriteImages = data.images;

    const grid = $("#fav-grid");
    grid.innerHTML = "";

    if (data.images.length === 0) {
      const empty = el("div", { className: "gallery-empty" },
        el("div", { style: { fontSize: "2rem", opacity: "0.3" } }, "★"),
        el("div", {}, "No favourites yet"),
        el("div", { className: "u-muted", style: { fontSize: "var(--text-xs)" } }, "Star images to add them here")
      );
      grid.appendChild(empty);
    } else {
      const favouritesOffset = (data.page - 1) * State.galleryPerPage;
      data.images.forEach((img, index) => {
        const collectionIndex = favouritesOffset + index + 1;
        grid.appendChild(createImageCard(img, "favourites", collectionIndex));
      });
    }

    $("#lbl-fav-count").textContent = formatImageCountLabel(data.total);
    $("#lbl-fav-page").textContent = `Page ${data.page} of ${data.pages}`;
    $("#btn-fav-prev").disabled = data.page <= 1;
    $("#btn-fav-next").disabled = data.page >= data.pages;

  } catch (e) {
    toast(`Favourites load failed: ${e.message}`, "err");
  }
}


// ── Stats modal ────────────────────────────────────────────────────────────────

async function openStatsModal() {
  $("#modal-stats").classList.remove("hidden");

  try {
    const res = await fetch("/api/stats");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const content = $("#modal-stats-content");
    content.innerHTML = "";

    const table = el("table");
    const tbody = el("tbody");

    const rows = [
      ["Total Images", data.total_images],
      ["Favourites", data.total_favourites],
    ];

    if (State.config) {
      State.config.models.forEach(m => {
        rows.push([m.label, data.model_counts[m.id] || 0]);
      });
    }

    rows.forEach(([label, value]) => {
      const tr = el("tr");
      tr.appendChild(el("td", { style: { color: "var(--col-text-muted)", paddingRight: "var(--sp-4)" } }, label));
      tr.appendChild(el("td", { style: { color: "var(--col-accent)", fontWeight: "700" } }, String(value)));
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    content.appendChild(table);

  } catch (e) {
    $("#modal-stats-content").textContent = `Error: ${e.message}`;
  }
}


// ── Seed controls ──────────────────────────────────────────────────────────────

function onRandomSeedChange() {
  const isRandom = $("#chk-random-seed").checked;
  $("#inp-seed").disabled = isRandom;
  $("#btn-new-seed").disabled = isRandom;
  updateStatusBar();
}

function generateRandomSeed() {
  const seed = Math.floor(Math.random() * 4294967295);
  $("#inp-seed").value = seed;
  updateStatusBar();
}


// ── Batch counter ──────────────────────────────────────────────────────────────

function updateBatchDisplay() {
  $("#inp-batch").value = String(State.batchSize);
}

function normalizeBatchSize(value) {
  const parsed = parseInt(value, 10);
  if (Number.isNaN(parsed)) return 1;
  return Math.min(MAX_BATCH_SIZE, Math.max(1, parsed));
}

function onBatchInput() {
  State.batchSize = normalizeBatchSize($("#inp-batch").value);
  updateBatchDisplay();
}

function incBatch() {
  if (State.batchSize < MAX_BATCH_SIZE) {
    State.batchSize++;
    updateBatchDisplay();
  }
}

function decBatch() {
  if (State.batchSize > 1) {
    State.batchSize--;
    updateBatchDisplay();
  }
}


// ── Event wiring ───────────────────────────────────────────────────────────────

function wireEvents() {
  /**
   * Create the dedicated lightbox controller before wiring global handlers so
   * keyboard shortcuts and button flows can delegate to the module cleanly.
   */
  outputLightboxController = createOutputLightboxController({
    getCollectionImages: (context) => {
      if (context === "output") {
        return State.outputImages;
      }

      if (context === "gallery") {
        return State.galleryImages;
      }

      if (context === "favourites") {
        return State.favouriteImages;
      }

      return [];
    },
    onImageChange: (image) => {
      State.lightboxImage = image;
    },
    onClose: () => {
      State.lightboxImage = null;
      State.lightboxContext = null;
    },
    onToggleFavourite: (image) => {
      const card = findLightboxContextCard(image.id);
      const favButton = card ? card.querySelector(".img-card__fav-btn") : null;
      const nextFavouriteState = !image.is_favourite;

      toggleFavourite(
        image.id,
        nextFavouriteState,
        card || { classList: { toggle: () => {} } },
        favButton || {
          classList: { toggle: () => {} },
          textContent: "",
          title: "",
        },
      );
    },
    onDeleteImage: (image) => {
      const card = findLightboxContextCard(image.id);
      deleteImage(image.id, card || { style: {}, remove: () => {} });
    },
  });

  // Theme toggle
  $("#btn-theme-toggle").addEventListener("click", toggleTheme);
  $$("[data-section-toggle]").forEach(button => {
    button.addEventListener("click", () => {
      const section = button.closest(".ctrl-section");
      if (!section) return;
      toggleSection(section);
    });
  });

  // Tab navigation
  $$(".tab-nav__item").forEach(btn => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  // Model change
  $("#sel-model").addEventListener("change", onModelChange);

  // Aspect ratio change
  $("#sel-aspect").addEventListener("change", onAspectChange);

  // Prepend mode toggle
  $("#btn-prepend-template").addEventListener("click", () => setPrependMode("template"));
  $("#btn-prepend-manual").addEventListener("click", () => setPrependMode("manual"));

  // Main scene prompt mode toggle
  $("#btn-mode-manual").addEventListener("click", () => setMainPromptMode("manual"));
  $("#btn-mode-auto").addEventListener("click", () => setMainPromptMode("automated"));

  // Append mode toggle
  $("#btn-append-template").addEventListener("click", () => setAppendMode("template"));
  $("#btn-append-manual").addEventListener("click", () => setAppendMode("manual"));

  // Prompt section clipboard controls
  $("#btn-copy-prepend").addEventListener("click", function () {
    copyPromptSection("prepend", this);
  });
  $("#btn-copy-main").addEventListener("click", function () {
    copyPromptSection("main", this);
  });
  $("#btn-copy-append").addEventListener("click", function () {
    copyPromptSection("append", this);
  });

  // Prompt inputs → live preview + token counters
  $("#txt-manual-prepend").addEventListener("input", () => { updateTokenCounters(); schedulePromptPreview(); });
  $("#txt-manual-prompt").addEventListener("input", () => { updateTokenCounters(); schedulePromptPreview(); });
  $("#txt-manual-append").addEventListener("input", () => { updateTokenCounters(); schedulePromptPreview(); });
  $("#sel-prepend").addEventListener("change", () => { updateTokenCounters(); schedulePromptPreview(); });
  $("#sel-auto-prompt").addEventListener("change", () => { updateTokenCounters(); schedulePromptPreview(); });
  $("#sel-append").addEventListener("change", () => { updateTokenCounters(); schedulePromptPreview(); });

  // Sliders
  $("#rng-steps").addEventListener("input", function () {
    $("#lbl-steps").textContent = this.value;
  });

  $("#rng-guidance").addEventListener("input", function () {
    $("#lbl-guidance").textContent = parseFloat(this.value).toFixed(1);
  });

  // Seed
  $("#chk-random-seed").addEventListener("change", onRandomSeedChange);
  $("#btn-new-seed").addEventListener("click", generateRandomSeed);
  $("#inp-seed").addEventListener("input", updateStatusBar);

  // Batch
  $("#btn-batch-inc").addEventListener("click", incBatch);
  $("#btn-batch-dec").addEventListener("click", decBatch);
  $("#inp-batch").addEventListener("input", onBatchInput);

  // Generate
  $("#btn-generate").addEventListener("click", generate);
  $("#btn-stop-generation").addEventListener("click", stopGeneration);

  // Clear output
  $("#btn-clear-output").addEventListener("click", () => {
    const canvas = $("#gen-canvas");
    canvas.innerHTML = "";
    State.outputImages = [];
    if (outputLightboxController) {
      outputLightboxController.resetOutputCollection();
    }
    const placeholder = el("div", { className: "gen-placeholder", id: "gen-placeholder" },
      el("div", { className: "gen-placeholder__icon" }, "◈"),
      el("div", {}, "Configure your prompt and click "),
      el("div", { className: "u-muted", style: { fontSize: "var(--text-xs)" } }, "Images will appear here")
    );
    canvas.appendChild(placeholder);
    updateOutputCount();
  });

  // Stats modal
  $("#btn-stats").addEventListener("click", openStatsModal);
  $("#modal-stats-close").addEventListener("click", () => $("#modal-stats").classList.add("hidden"));
  $("#modal-stats-close2").addEventListener("click", () => $("#modal-stats").classList.add("hidden"));
  $("#modal-stats-backdrop").addEventListener("click", () => $("#modal-stats").classList.add("hidden"));

  // Gallery
  $("#btn-gallery-refresh").addEventListener("click", () => loadGallery(State.galleryPage));
  $("#btn-gallery-prev").addEventListener("click", () => loadGallery(State.galleryPage - 1));
  $("#btn-gallery-next").addEventListener("click", () => loadGallery(State.galleryPage + 1));
  $("#sel-gallery-model").addEventListener("change", () => {
    State.galleryModelFilter = $("#sel-gallery-model").value;
    loadGallery(1);
  });

  // Gallery bulk selection
  $("#btn-gallery-select").addEventListener("click", toggleSelectMode);
  $("#btn-select-all").addEventListener("click", selectAllVisible);
  $("#btn-delete-selected").addEventListener("click", bulkDelete);

  // Favourites
  $("#btn-fav-refresh").addEventListener("click", () => loadFavourites(State.favPage));
  $("#btn-fav-prev").addEventListener("click", () => loadFavourites(State.favPage - 1));
  $("#btn-fav-next").addEventListener("click", () => loadFavourites(State.favPage + 1));

  // Keyboard shortcuts
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") {
      closeLightbox();
      $("#modal-stats").classList.add("hidden");
      return;
    }

    if (outputLightboxController && outputLightboxController.handleKeydown(e)) {
      return;
    }

    /**
     * Gallery page navigation is intentionally limited to the Gallery tab and
     * is disabled while a modal or text-entry control owns focus.
     */
    if (!e.altKey && !e.ctrlKey && !e.metaKey && !isTypingTargetActive()) {
      const galleryDirection = resolveGalleryPaginationDirection(e.key);
      const activeTabId = getActiveTabId();
      const statsModalOpen = !$("#modal-stats").classList.contains("hidden");

      if (
        galleryDirection !== 0
        && activeTabId === "gallery"
        && !statsModalOpen
      ) {
        if (galleryDirection === -1 && State.galleryPage > 1) {
          e.preventDefault();
          loadGallery(State.galleryPage - 1);
          return;
        }

        if (galleryDirection === 1 && State.galleryPage < State.galleryPages) {
          e.preventDefault();
          loadGallery(State.galleryPage + 1);
          return;
        }
      }
    }

    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      if (!State.isGenerating) generate();
    }
  });
}


// ── Init ───────────────────────────────────────────────────────────────────────

async function init() {
  // Restore theme
  const savedTheme = localStorage.getItem("pw-theme") || "dark";
  applyTheme(savedTheme);
  initializeCollapsibleSections();

  setStatus("Loading config…", true);
  wireEvents();
  await loadConfig();
  updateStatusBar();
}

document.addEventListener("DOMContentLoaded", init);
