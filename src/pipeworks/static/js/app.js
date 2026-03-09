/**
 * Pipe-Works Image Generator — Frontend Application
 * Vanilla JS, no frameworks. Pipe-Works design system.
 */

import { getImageCardBadgeLabel } from "./gallery-context.mjs";
import {
  formatImageCountLabel,
  resolveGalleryPaginationDirection,
} from "./gallery-navigation.mjs";
import { createOutputLightboxController } from "./output-lightbox.mjs";

"use strict";

const MAX_BATCH_SIZE = 1000;
const COPY_FEEDBACK_MS = 1200;
const SECTION_COLLAPSE_STORAGE_PREFIX = "pw-section-collapsed:";
const PROMPT_SECTIONS = ["subject", "setting", "details", "lighting", "atmosphere"];
const PROMPT_SECTION_LABELS = {
  subject: "subject",
  setting: "setting",
  details: "details",
  lighting: "lighting",
  atmosphere: "atmosphere",
};

// ── State ──────────────────────────────────────────────────────────────────────

const State = {
  config: null,
  selectedModel: null,
  sectionModes: {
    subject: "manual",
    setting: "manual",
    details: "manual",
    lighting: "manual",
    atmosphere: "manual",
  },
  batchSize: 1,
  isGenerating: false,
  currentGenerationId: null,
  stopRequested: false,
  outputImages: [],
  galleryImages: [],
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
    subject: 0,
    setting: 0,
    details: 0,
    lighting: 0,
    atmosphere: 0,
    total: 0,
    method: "heuristic",
  },
  selectMode: false,
  selectedIds: new Set(),
  outputSelectMode: false,
  outputSelectedIds: new Set(),
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
  const policyOptions = cfg.policy_prompt_options || [];

  function populatePolicySelect(selectEl, options) {
    if (!selectEl) return;
    selectEl.innerHTML = "";
    selectEl.appendChild(el("option", { value: "" }, "— Add snippet from policies —"));

    const grouped = new Map();
    options.forEach(option => {
      const group = option.group || "policies";
      if (!grouped.has(group)) grouped.set(group, []);
      grouped.get(group).push(option);
    });

    [...grouped.keys()].sort().forEach(group => {
      const optGroup = document.createElement("optgroup");
      optGroup.label = group;
      grouped.get(group).forEach(option => {
        const opt = el("option", { value: option.id }, option.label);
        optGroup.appendChild(opt);
      });
      selectEl.appendChild(optGroup);
    });

    selectEl.value = "";
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

  // Section snippet dropdowns sourced from policies.
  PROMPT_SECTIONS.forEach(section => {
    populatePolicySelect($(`#sel-${section}`), policyOptions);
  });

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

function setSectionPromptMode(section, mode) {
  State.sectionModes[section] = mode;
  $(`#btn-${section}-automated`).classList.toggle("is-active", mode === "automated");
  $(`#btn-${section}-manual`).classList.toggle("is-active", mode === "manual");
  $(`#${section}-auto-wrap`).style.display = mode === "automated" ? "" : "none";
  updateTokenCounters();
  schedulePromptPreview();
}

function appendPolicySnippetToSection(section, optionId) {
  if (!optionId || !State.config) return;
  const option = (State.config.policy_prompt_options || []).find(item => item.id === optionId);
  if (!option || !(option.value || "").trim()) return;

  const textarea = $(`#txt-${section}`);
  if (!textarea) return;

  const existing = textarea.value.trimEnd();
  const snippet = option.value.trim();
  textarea.value = existing ? `${existing}\n${snippet}` : snippet;

  const select = $(`#sel-${section}`);
  if (select) select.value = "";
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
  const textarea = $(`#txt-${section}`);
  return textarea ? textarea.value : "";
}

function getPromptSectionDisplayName(section) {
  return PROMPT_SECTION_LABELS[section] || "prompt section";
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

  applyCounter("#subject-tokens", counts.subject || 0, maxTokens);
  applyCounter("#setting-tokens", counts.setting || 0, maxTokens);
  applyCounter("#details-tokens", counts.details || 0, maxTokens);
  applyCounter("#lighting-tokens", counts.lighting || 0, maxTokens);
  applyCounter("#atmosphere-tokens", counts.atmosphere || 0, maxTokens);
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
    State.tokenCounts = {
      subject: 0,
      setting: 0,
      details: 0,
      lighting: 0,
      atmosphere: 0,
      total: 0,
      method: "heuristic",
    };
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
      subject: estimateTokens(getPromptSectionText("subject")),
      setting: estimateTokens(getPromptSectionText("setting")),
      details: estimateTokens(getPromptSectionText("details")),
      lighting: estimateTokens(getPromptSectionText("lighting")),
      atmosphere: estimateTokens(getPromptSectionText("atmosphere")),
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
    prompt_schema_version: 2,
    aspect_ratio_id: aspectId,
    width: ar.width,
    height: ar.height,
    steps: parseInt($("#rng-steps").value, 10),
    guidance: parseFloat($("#rng-guidance").value),
    seed: seedVal,
    batch_size: State.batchSize,
  };

  // Five independent composer sections.
  PROMPT_SECTIONS.forEach(section => {
    const mode = State.sectionModes[section] || "manual";
    const sectionText = (getPromptSectionText(section) || "").trim();
    const selectedOptionId = $(`#sel-${section}`)?.value || null;

    payload[`${section}_mode`] = mode;
    payload[`manual_${section}`] = sectionText || null;
    if (mode === "automated" && selectedOptionId) {
      payload[`automated_${section}_prompt_id`] = selectedOptionId;
    } else {
      payload[`automated_${section}_prompt_id`] = null;
    }
  });

  if (model.supports_negative_prompt) {
    payload.negative_prompt = $("#txt-negative-prompt").value.trim() || null;
  } else {
    payload.negative_prompt = null;
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
    if (context === "output" && State.outputSelectMode) {
      toggleOutputCardSelection(img.id, card);
    } else if (context !== "output" && State.selectMode) {
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

  imageIds.forEach(imageId => {
    State.selectedIds.delete(imageId);
    State.outputSelectedIds.delete(imageId);
  });

  updateSelectionUI();
  updateOutputSelectionUI();
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
  $("#btn-save-selected").disabled = count === 0;
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

/**
 * Download a zip of the currently selected gallery images.
 */
async function downloadSelectedZip() {
  const count = State.selectedIds.size;
  if (count === 0) return;

  const btn = $("#btn-save-selected");
  const originalText = btn.textContent;
  btn.textContent = "Downloading\u2026";
  btn.disabled = true;

  try {
    const res = await fetch("/api/gallery/bulk-zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_ids: [...State.selectedIds] }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pipeworks_selected_${count}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    toast(`Downloaded ${count} image${count !== 1 ? "s" : ""}`, "ok", 1500);
  } catch (e) {
    toast(`Download failed: ${e.message}`, "err");
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

function toggleOutputSelectMode() {
  State.outputSelectMode = !State.outputSelectMode;
  State.outputSelectedIds.clear();

  const btn = $("#btn-output-select");
  const controls = $("#output-select-controls");
  const canvas = $("#gen-canvas");

  btn.textContent = State.outputSelectMode ? "✕ Cancel" : "☐ Select";
  controls.classList.toggle("is-active", State.outputSelectMode);
  canvas.classList.toggle("gen-output__canvas--selecting", State.outputSelectMode);

  $$(".img-card.is-selected", canvas).forEach(c => c.classList.remove("is-selected"));

  updateOutputSelectionUI();
}

function toggleOutputCardSelection(imgId, card) {
  if (State.outputSelectedIds.has(imgId)) {
    State.outputSelectedIds.delete(imgId);
    card.classList.remove("is-selected");
  } else {
    State.outputSelectedIds.add(imgId);
    card.classList.add("is-selected");
  }
  updateOutputSelectionUI();
}

function selectAllOutputVisible() {
  const canvas = $("#gen-canvas");
  $$(".img-card", canvas).forEach(card => {
    const id = card.getAttribute("data-id");
    if (id && !State.outputSelectedIds.has(id)) {
      State.outputSelectedIds.add(id);
      card.classList.add("is-selected");
    }
  });
  updateOutputSelectionUI();
}

function getVisibleOutputImageIds() {
  const canvas = $("#gen-canvas");
  return [...new Set(
    $$(".img-card", canvas)
      .map(card => card.getAttribute("data-id"))
      .filter(Boolean),
  )];
}

function updateOutputSelectionUI() {
  const selectedCount = State.outputSelectedIds.size;
  const outputCount = getVisibleOutputImageIds().length;
  $("#lbl-output-select-count").textContent = `${selectedCount} selected`;
  $("#btn-output-select-all").disabled = outputCount === 0;
  $("#btn-output-save-all").disabled = outputCount === 0;
}

async function downloadAllOutputZip() {
  const imageIds = getVisibleOutputImageIds();
  const count = imageIds.length;
  if (count === 0) return;

  const btn = $("#btn-output-save-all");
  const originalText = btn.textContent;
  btn.textContent = "Downloading\u2026";
  btn.disabled = true;

  try {
    const res = await fetch("/api/gallery/bulk-zip", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_ids: imageIds }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pipeworks_output_${count}.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    toast(`Downloaded ${count} image${count !== 1 ? "s" : ""}`, "ok", 1500);
  } catch (e) {
    toast(`Download failed: ${e.message}`, "err");
  } finally {
    btn.textContent = originalText;
    updateOutputSelectionUI();
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
 * Load one page of the gallery as a flat image grid.
 *
 * @param {number} [page=1] - One-based page to load.
 */
async function loadGallery(page = 1) {
  State.galleryPage = page;

  // Exit select mode when gallery reloads (e.g. filter change, page change).
  if (State.selectMode) toggleSelectMode();

  const modelFilter = $("#sel-gallery-model").value;

  try {
    let url = `/api/gallery?page=${page}&per_page=${State.galleryPerPage}`;
    if (modelFilter) url += `&model_id=${encodeURIComponent(modelFilter)}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    State.galleryTotal = data.total;
    State.galleryPage = data.page;
    State.galleryPages = data.pages;
    State.galleryImages = data.images;

    const grid = $("#gallery-grid");
    grid.innerHTML = "";

    if (data.images.length === 0) {
      const empty = el("div", { className: "gallery-empty" },
        el("div", { style: { fontSize: "2rem", opacity: "0.3" } }, "◈"),
        el("div", {}, "No images found"),
        el("div", { className: "u-muted", style: { fontSize: "var(--text-xs)" } },
          modelFilter ? "Try a different filter" : "Generate some images first"
        )
      );
      grid.appendChild(empty);
    } else {
      const pageOffset = (data.page - 1) * State.galleryPerPage;
      data.images.forEach((img, index) => {
        const collectionIndex = pageOffset + index + 1;
        grid.appendChild(createImageCard(img, "gallery", collectionIndex));
      });
    }

    $("#lbl-gallery-count").textContent = formatImageCountLabel(data.total);
    $("#lbl-gallery-page").textContent = `Page ${data.page} of ${data.pages}`;
    $("#btn-gallery-prev").disabled = data.page <= 1;
    $("#btn-gallery-next").disabled = data.page >= data.pages;

  } catch (e) {
    toast(`Gallery load failed: ${e.message}`, "err");
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

  // Prompt section controls (mode, copy, textarea input, policy snippet append).
  PROMPT_SECTIONS.forEach(section => {
    $(`#btn-${section}-automated`).addEventListener("click", () => {
      setSectionPromptMode(section, "automated");
    });
    $(`#btn-${section}-manual`).addEventListener("click", () => {
      setSectionPromptMode(section, "manual");
    });
    $(`#btn-copy-${section}`).addEventListener("click", function () {
      copyPromptSection(section, this);
    });
    $(`#txt-${section}`).addEventListener("input", () => {
      updateTokenCounters();
      schedulePromptPreview();
    });
    $(`#sel-${section}`).addEventListener("change", (event) => {
      appendPolicySnippetToSection(section, event.target.value);
    });
  });

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
    if (State.outputSelectMode) {
      toggleOutputSelectMode();
    } else {
      State.outputSelectedIds.clear();
      updateOutputSelectionUI();
    }

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
    updateOutputSelectionUI();
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
  $("#btn-save-selected").addEventListener("click", downloadSelectedZip);
  $("#btn-delete-selected").addEventListener("click", bulkDelete);

  // Output bulk selection
  $("#btn-output-select").addEventListener("click", toggleOutputSelectMode);
  $("#btn-output-select-all").addEventListener("click", selectAllOutputVisible);
  $("#btn-output-save-all").addEventListener("click", downloadAllOutputZip);

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

  updateOutputSelectionUI();
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
