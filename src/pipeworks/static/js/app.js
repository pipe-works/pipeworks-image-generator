/**
 * Pipe-Works Image Generator — Frontend Application
 * Vanilla JS, no frameworks. Pipe-Works design system.
 */

import { createOutputLightboxController } from "./output-lightbox.mjs";

"use strict";

// ── State ──────────────────────────────────────────────────────────────────────

const State = {
  config: null,
  selectedModel: null,
  prependMode: "template",
  promptMode: "manual",
  appendMode: "template",
  batchSize: 1,
  isGenerating: false,
  outputImages: [],
  galleryPage: 1,
  galleryPerPage: 20,
  galleryTotal: 0,
  galleryPages: 1,
  galleryModelFilter: "",
  favPage: 1,
  favTotal: 0,
  favPages: 1,
  lightboxImage: null,
  theme: "dark",
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

  // Models
  const selModel = $("#sel-model");
  selModel.innerHTML = "";
  cfg.models.forEach(m => {
    const opt = el("option", { value: m.id }, m.label);
    selModel.appendChild(opt);
  });

  // Gallery model filter
  const selGalleryModel = $("#sel-gallery-model");
  cfg.models.forEach(m => {
    const opt = el("option", { value: m.id }, m.label);
    selGalleryModel.appendChild(opt);
  });

  // Prepend prompts
  const selPrepend = $("#sel-prepend");
  selPrepend.innerHTML = "";
  cfg.prepend_prompts.forEach(p => {
    const opt = el("option", { value: p.id }, p.label);
    selPrepend.appendChild(opt);
  });

  // Automated prompts
  const selAuto = $("#sel-auto-prompt");
  selAuto.innerHTML = "";
  cfg.automated_prompts.forEach(p => {
    const opt = el("option", { value: p.id }, p.label);
    selAuto.appendChild(opt);
  });

  // Append prompts
  const selAppend = $("#sel-append");
  selAppend.innerHTML = "";
  cfg.append_prompts.forEach(p => {
    const opt = el("option", { value: p.id }, p.label);
    selAppend.appendChild(opt);
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
  $("#model-info-desc").textContent = model.description;
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

  updateStatusBar();
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


// ── Token estimation ──────────────────────────────────────────────────────────

/**
 * Estimate CLIP BPE token count using a ~4 chars/token heuristic.
 * Returns 0 for empty or whitespace-only text.
 */
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

/**
 * Update all token counter elements with current estimates.
 * Per-section counters show user text only.  The total counter
 * reflects the full compiled prompt (including boilerplate).
 */
function updateTokenCounters() {
  if (!State.config || !State.selectedModel) return;

  const model = State.config.models.find(m => m.id === State.selectedModel);
  const maxTokens = model ? (model.max_prompt_tokens || 77) : 77;

  // Per-section estimates (user text only).
  const prependCount = estimateTokens(getPromptSectionText("prepend"));
  const mainCount = estimateTokens(getPromptSectionText("main"));
  const appendCount = estimateTokens(getPromptSectionText("append"));

  // Total estimate from the compiled prompt preview text.
  const previewText = $("#prompt-preview-box").textContent || "";
  const totalCount = estimateTokens(previewText);

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

  applyCounter("#prepend-tokens", prependCount, maxTokens);
  applyCounter("#main-tokens", mainCount, maxTokens);
  applyCounter("#append-tokens", appendCount, maxTokens);
  applyCounter("#total-tokens", totalCount, maxTokens);
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
    $("#prompt-preview-box").textContent = "Fill in the required fields to preview…";
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
    const preview = $("#prompt-preview-box");
    preview.textContent = data.compiled_prompt;
    // Also update modal if open
    const modalText = $("#modal-prompt-text");
    if (modalText) modalText.textContent = data.compiled_prompt;
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
    payload.prepend_prompt_id = $("#sel-prepend").value;
  }

  // Main scene: manual or automated
  if (State.promptMode === "manual") {
    payload.manual_prompt = $("#txt-manual-prompt").value.trim();
  } else {
    payload.automated_prompt_id = $("#sel-auto-prompt").value;
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

  // Validate required fields
  if (State.prependMode === "template" && !$("#sel-prepend").value) {
    toast("Prepend style is required", "warn");
    return;
  }

  if (State.promptMode === "manual") {
    const mp = $("#txt-manual-prompt").value.trim();
    if (!mp) {
      toast("Manual prompt is required", "warn");
      $("#txt-manual-prompt").focus();
      return;
    }
  }

  const payload = buildGeneratePayload();
  if (!payload) {
    toast("Please configure all required settings", "warn");
    return;
  }

  State.isGenerating = true;
  const btn = $("#btn-generate");
  btn.classList.add("is-loading");
  btn.disabled = true;
  btn.textContent = "◆ Generating…";

  const progressWrap = $("#progress-wrap");
  progressWrap.style.display = "";

  setStatus(`Generating ${State.batchSize} image(s)…`, true);

  try {
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
    toast(`Generated ${data.images.length} image(s)`, "ok");
    setStatus(`Done — ${data.images.length} image(s) generated`);

  } catch (e) {
    toast(`Generation failed: ${e.message}`, "err");
    setStatus("Generation failed", false);
  } finally {
    State.isGenerating = false;
    btn.classList.remove("is-loading");
    btn.disabled = false;
    btn.textContent = "◆ Generate";
    progressWrap.style.display = "none";
  }
}


// ── Image card ─────────────────────────────────────────────────────────────────

function createImageCard(img, context = "gallery") {
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

  const batchBadge = el("div", { className: "img-card__batch-badge" },
    `#${(img.batch_index || 0) + 1}`
  );

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

    // Update output images state
    const outImg = State.outputImages.find(i => i.id === imageId);
    if (outImg) outImg.is_favourite = isFav;

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

    State.outputImages = State.outputImages.filter(i => i.id !== imageId);
    updateOutputCount();

    if (outputLightboxController) {
      outputLightboxController.handleRemovedImages([imageId]);
    }

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

    // Also remove from output images state if applicable.
    const deletedSet = new Set(data.deleted);
    State.outputImages = State.outputImages.filter(i => !deletedSet.has(i.id));

    // Let the dedicated lightbox controller decide whether to close or
    // advance to the nearest surviving Output image.
    if (outputLightboxController) {
      outputLightboxController.handleRemovedImages(data.deleted);
    }

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
  $("#lbl-output-count").textContent = `${count} image${count !== 1 ? "s" : ""}`;
}


// ── Lightbox ───────────────────────────────────────────────────────────────────

function openLightbox(img, context = "gallery") {
  if (!outputLightboxController) return;
  outputLightboxController.open({ image: img, context });
}

function closeLightbox() {
  if (!outputLightboxController) return;
  outputLightboxController.close();
}


// ── Gallery ────────────────────────────────────────────────────────────────────

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
    State.galleryPages = data.pages;

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
      data.images.forEach(img => {
        grid.appendChild(createImageCard(img, "gallery"));
      });
    }

    $("#lbl-gallery-count").textContent = `${data.total} image${data.total !== 1 ? "s" : ""}`;
    $("#lbl-gallery-page").textContent = `Page ${page} of ${data.pages}`;
    $("#btn-gallery-prev").disabled = page <= 1;
    $("#btn-gallery-next").disabled = page >= data.pages;

  } catch (e) {
    toast(`Gallery load failed: ${e.message}`, "err");
  }
}

async function loadFavourites(page = 1) {
  State.favPage = page;

  try {
    const url = `/api/gallery?page=${page}&per_page=${State.galleryPerPage}&favourites_only=true`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    State.favTotal = data.total;
    State.favPages = data.pages;

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
      data.images.forEach(img => {
        grid.appendChild(createImageCard(img, "favourites"));
      });
    }

    $("#lbl-fav-count").textContent = `${data.total} image${data.total !== 1 ? "s" : ""}`;
    $("#lbl-fav-page").textContent = `Page ${page} of ${data.pages}`;
    $("#btn-fav-prev").disabled = page <= 1;
    $("#btn-fav-next").disabled = page >= data.pages;

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


// ── Prompt preview modal ───────────────────────────────────────────────────────

async function openPromptModal() {
  await updatePromptPreview();
  const preview = $("#prompt-preview-box").textContent;
  $("#modal-prompt-text").textContent = preview;
  $("#modal-prompt").classList.remove("hidden");
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
  $("#lbl-batch").textContent = State.batchSize;
}

function incBatch() {
  if (State.batchSize < 16) {
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
    getOutputImages: () => State.outputImages,
    onImageChange: (image) => {
      State.lightboxImage = image;
    },
    onClose: () => {
      State.lightboxImage = null;
    },
    onToggleFavourite: (image) => {
      const card = document.querySelector(`.img-card[data-id="${image.id}"]`);
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
      const card = document.querySelector(`.img-card[data-id="${image.id}"]`);
      deleteImage(image.id, card || { style: {}, remove: () => {} });
    },
  });

  // Theme toggle
  $("#btn-theme-toggle").addEventListener("click", toggleTheme);

  // Tab navigation
  $$(".tab-nav__item").forEach(btn => {
    btn.addEventListener("click", () => activateTab(btn.dataset.tab));
  });

  // Model change
  $("#sel-model").addEventListener("change", onModelChange);

  // Aspect ratio change
  $("#sel-aspect").addEventListener("change", onAspectChange);

  // Prepend mode toggle
  $("#btn-prepend-template").addEventListener("click", () => {
    State.prependMode = "template";
    $("#btn-prepend-template").classList.add("is-active");
    $("#btn-prepend-manual").classList.remove("is-active");
    $("#prepend-template-wrap").style.display = "";
    $("#prepend-manual-wrap").style.display = "none";
    schedulePromptPreview();
  });

  $("#btn-prepend-manual").addEventListener("click", () => {
    State.prependMode = "manual";
    $("#btn-prepend-manual").classList.add("is-active");
    $("#btn-prepend-template").classList.remove("is-active");
    $("#prepend-manual-wrap").style.display = "";
    $("#prepend-template-wrap").style.display = "none";
    schedulePromptPreview();
  });

  // Main scene prompt mode toggle
  $("#btn-mode-manual").addEventListener("click", () => {
    State.promptMode = "manual";
    $("#btn-mode-manual").classList.add("is-active");
    $("#btn-mode-auto").classList.remove("is-active");
    $("#prompt-manual-wrap").style.display = "";
    $("#prompt-auto-wrap").style.display = "none";
    schedulePromptPreview();
  });

  $("#btn-mode-auto").addEventListener("click", () => {
    State.promptMode = "automated";
    $("#btn-mode-auto").classList.add("is-active");
    $("#btn-mode-manual").classList.remove("is-active");
    $("#prompt-auto-wrap").style.display = "";
    $("#prompt-manual-wrap").style.display = "none";
    schedulePromptPreview();
  });

  // Append mode toggle
  $("#btn-append-template").addEventListener("click", () => {
    State.appendMode = "template";
    $("#btn-append-template").classList.add("is-active");
    $("#btn-append-manual").classList.remove("is-active");
    $("#append-template-wrap").style.display = "";
    $("#append-manual-wrap").style.display = "none";
    schedulePromptPreview();
  });

  $("#btn-append-manual").addEventListener("click", () => {
    State.appendMode = "manual";
    $("#btn-append-manual").classList.add("is-active");
    $("#btn-append-template").classList.remove("is-active");
    $("#append-manual-wrap").style.display = "";
    $("#append-template-wrap").style.display = "none";
    schedulePromptPreview();
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

  // Generate
  $("#btn-generate").addEventListener("click", generate);

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

  // Prompt preview expand/collapse
  $("#btn-expand-prompt").addEventListener("click", function () {
    const box = $("#prompt-preview-box");
    const isExpanded = box.classList.toggle("prompt-preview--expanded");
    this.textContent = isExpanded ? "collapse" : "expand";
  });

  // Prompt modal
  $("#btn-prompt-preview").addEventListener("click", openPromptModal);
  $("#modal-prompt-close").addEventListener("click", () => $("#modal-prompt").classList.add("hidden"));
  $("#modal-prompt-close2").addEventListener("click", () => $("#modal-prompt").classList.add("hidden"));
  $("#modal-prompt-backdrop").addEventListener("click", () => $("#modal-prompt").classList.add("hidden"));
  $("#btn-copy-prompt").addEventListener("click", () => {
    const text = $("#modal-prompt-text").textContent;
    navigator.clipboard.writeText(text).then(() => toast("Prompt copied", "ok", 1500));
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
      $("#modal-prompt").classList.add("hidden");
      $("#modal-stats").classList.add("hidden");
      return;
    }

    if (outputLightboxController && outputLightboxController.handleKeydown(e)) {
      return;
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

  setStatus("Loading config…", true);
  wireEvents();
  await loadConfig();
  updateStatusBar();
}

document.addEventListener("DOMContentLoaded", init);
