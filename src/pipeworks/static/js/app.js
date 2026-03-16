/**
 * Pipe-Works Image Generator — Frontend Application
 * Bootstrap/composition root. Feature logic lives in app/*.mjs modules.
 */

import { getImageCardBadgeLabel } from "./gallery-context.mjs";
import {
  formatImageCountLabel,
  resolveGalleryPaginationDirection,
} from "./gallery-navigation.mjs";
import { createOutputLightboxController } from "./output-lightbox.mjs";
import { createApiClient } from "./app/api-client.mjs";
import { $, $$, el, fetchJson } from "./app/dom-utils.mjs";
import { createGenerationFlow } from "./app/generation-flow.mjs";
import { createGalleryManager } from "./app/gallery-manager.mjs";
import { createPromptComposer } from "./app/prompt-composer.mjs";
import { createRuntimeGpuController } from "./app/runtime-gpu-controller.mjs";
import {
  COPY_FEEDBACK_MS,
  MAX_BATCH_SIZE,
  PROMPT_SECTIONS,
  SECTION_COLLAPSE_STORAGE_PREFIX,
  State,
} from "./app/state.mjs";

"use strict";

let outputLightboxController = null;

const apiClient = createApiClient({ fetchJson });

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

function setStatus(msg, busy = false) {
  $("#status-text").textContent = msg;
  const dot = $("#status-dot");
  dot.classList.toggle("status-dot--busy", busy);
}

function updateStatusBar() {
  if (State.selectedModel) {
    const model = State.config.models.find(m => m.id === State.selectedModel);
    if (model) $("#status-model").textContent = model.label;
  }
  const seed = $("#inp-seed").value;
  const isRandom = $("#chk-random-seed").checked;
  $("#status-seed").textContent = isRandom ? "seed random" : `seed ${seed || "—"}`;
}

function setGpuSettingsStatus(message) {
  const status = $("#gpu-settings-status");
  if (status) {
    status.textContent = message;
  }
}

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

let promptComposer = null;

const runtimeGpuController = createRuntimeGpuController({
  state: State,
  $,
  el,
  apiClient,
  setStatus,
  toast,
  updateStatusBar,
  updatePromptPreview: () => promptComposer?.updatePromptPreview(),
  updateTokenCounters: () => promptComposer?.updateTokenCounters(),
  setGpuSettingsStatus,
});

const galleryManager = createGalleryManager({
  state: State,
  $,
  $$,
  el,
  apiClient,
  toast,
  formatImageCountLabel,
  getImageCardBadgeLabel,
  getOutputLightboxController: () => outputLightboxController,
});

promptComposer = createPromptComposer({
  state: State,
  $,
  apiClient,
  toast,
  flashButtonLabel,
  getSelectedGpuWorker: runtimeGpuController.getSelectedGpuWorker,
});

const generationFlow = createGenerationFlow({
  state: State,
  $,
  apiClient,
  toast,
  setStatus,
  buildGeneratePayload: promptComposer.buildGeneratePayload,
  selectedGpuWorkerLabel: runtimeGpuController.selectedGpuWorkerLabel,
  createImageCard: galleryManager.createImageCard,
  updateOutputCount: galleryManager.updateOutputCount,
});

function activateTab(tabId) {
  $$(".tab-nav__item").forEach(button => button.classList.toggle("is-active", button.dataset.tab === tabId));
  $$(".tab-content").forEach(content => content.classList.toggle("is-active", content.id === `tab-${tabId}`));

  if (tabId === "gallery") galleryManager.loadGallery();
  if (tabId === "favourites") galleryManager.loadFavourites();
}

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

function wireEvents() {
  outputLightboxController = createOutputLightboxController({
    getCollectionImages: context => {
      if (context === "output") return State.outputImages;
      if (context === "gallery") return State.galleryImages;
      if (context === "favourites") return State.favouriteImages;
      return [];
    },
    onClose: () => {
      State.lightboxContext = null;
    },
    onToggleFavourite: image => {
      const card = galleryManager.findLightboxContextCard(image.id);
      const favButton = card ? card.querySelector(".img-card__fav-btn") : null;
      const nextFavouriteState = !image.is_favourite;
      galleryManager.toggleFavourite(
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
    onDeleteImage: image => {
      const card = galleryManager.findLightboxContextCard(image.id);
      galleryManager.deleteImage(image.id, card || { style: {}, remove: () => {} });
    },
  });

  $("#btn-theme-toggle").addEventListener("click", toggleTheme);

  $$("[data-section-toggle]").forEach(button => {
    button.addEventListener("click", () => {
      const section = button.closest(".ctrl-section");
      if (!section) return;
      toggleSection(section);
    });
  });

  $$(".tab-nav__item").forEach(button => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });

  $("#runtime-mode-select")?.addEventListener("change", async event => {
    const nextMode = event.target.value;
    try {
      setStatus(`Switching snippet source to ${nextMode}...`, true);
      await runtimeGpuController.setRuntimeMode(nextMode);
      setStatus(`Snippet source switched to ${runtimeGpuController.runtimeModeLabel()}.`);
    } catch (error) {
      toast(`Runtime mode switch failed: ${error.message}`, "err");
      setStatus(`Runtime mode switch failed: ${error.message}`);
      await runtimeGpuController.loadRuntimeMode();
      await runtimeGpuController.refreshRuntimeAuthState({ silent: true });
      await runtimeGpuController.loadPolicyPrompts({ silent: true });
    }
  });

  $("#runtime-mode-url")?.addEventListener("input", () => {
    const applyButton = $("#runtime-mode-apply");
    if (!applyButton) return;
    applyButton.disabled = !($("#runtime-mode-url")?.value || "").trim();
  });

  $("#runtime-mode-apply")?.addEventListener("click", async () => {
    const modeKey = $("#runtime-mode-select")?.value;
    const serverUrl = ($("#runtime-mode-url")?.value || "").trim();
    if (!modeKey) return;
    try {
      setStatus(`Applying snippet source URL for ${runtimeGpuController.runtimeModeLabel()}...`, true);
      await runtimeGpuController.setRuntimeMode(modeKey, { explicitServerUrl: serverUrl });
      setStatus(`Snippet source URL updated for ${runtimeGpuController.runtimeModeLabel()}.`);
    } catch (error) {
      toast(`Failed to apply URL: ${error.message}`, "err");
      setStatus(`Failed to apply URL: ${error.message}`);
    }
  });

  $("#runtime-login-username")?.addEventListener("input", runtimeGpuController.applyRuntimeControls);
  $("#runtime-login-password")?.addEventListener("input", runtimeGpuController.applyRuntimeControls);

  $("#runtime-login-apply")?.addEventListener("click", async () => {
    try {
      if (runtimeGpuController.isRuntimeSessionAuthorized()) {
        await runtimeGpuController.logoutRuntimeSession();
      } else {
        await runtimeGpuController.loginRuntimeSession();
      }
    } catch (error) {
      toast(`Runtime session action failed: ${error.message}`, "err");
      setStatus(`Runtime session action failed: ${error.message}`);
    }
  });

  $("#chk-use-remote-gpu")?.addEventListener("change", () => {
    runtimeGpuController.syncGpuSettingsControls();
    setGpuSettingsStatus($("#chk-use-remote-gpu").checked ? "Remote GPU enabled." : "Remote GPU disabled.");
  });

  $("#btn-gpu-token-generate")?.addEventListener("click", runtimeGpuController.generateGpuTokenInField);
  $("#btn-gpu-test")?.addEventListener("click", runtimeGpuController.testGpuSettingsConnection);
  $("#btn-gpu-save")?.addEventListener("click", async () => {
    try {
      await runtimeGpuController.saveGpuSettings();
    } catch (error) {
      setGpuSettingsStatus(`Failed to save GPU settings: ${error.message}`);
      toast(`Failed to save GPU settings: ${error.message}`, "err");
    }
  });

  $("#sel-model").addEventListener("change", runtimeGpuController.onModelChange);
  $("#sel-gpu-worker")?.addEventListener("change", () => {
    State.selectedGpuWorkerId = $("#sel-gpu-worker").value;
    setStatus(`GPU machine set to ${runtimeGpuController.selectedGpuWorkerLabel()}.`);
  });

  $("#sel-aspect").addEventListener("change", runtimeGpuController.onAspectChange);

  PROMPT_SECTIONS.forEach(section => {
    $(`#btn-${section}-automated`).addEventListener("click", () => {
      promptComposer.setSectionPromptMode(section, "automated");
    });
    $(`#btn-${section}-manual`).addEventListener("click", () => {
      promptComposer.setSectionPromptMode(section, "manual");
    });
    $(`#btn-copy-${section}`).addEventListener("click", function () {
      promptComposer.copyPromptSection(section, this);
    });
    $(`#txt-${section}`).addEventListener("input", () => {
      promptComposer.updateTokenCounters();
      promptComposer.schedulePromptPreview();
    });
    $(`#sel-${section}`).addEventListener("change", event => {
      promptComposer.appendPolicySnippetToSection(section, event.target.value);
    });
  });

  $("#rng-steps").addEventListener("input", function () {
    $("#lbl-steps").textContent = this.value;
  });

  $("#rng-guidance").addEventListener("input", function () {
    $("#lbl-guidance").textContent = parseFloat(this.value).toFixed(1);
  });

  $("#chk-random-seed").addEventListener("change", onRandomSeedChange);
  $("#btn-new-seed").addEventListener("click", generateRandomSeed);
  $("#inp-seed").addEventListener("input", updateStatusBar);

  $("#btn-batch-inc").addEventListener("click", incBatch);
  $("#btn-batch-dec").addEventListener("click", decBatch);
  $("#inp-batch").addEventListener("input", onBatchInput);

  $("#btn-generate").addEventListener("click", generationFlow.generate);
  $("#btn-stop-generation").addEventListener("click", generationFlow.stopGeneration);

  $("#btn-clear-output").addEventListener("click", () => {
    if (State.outputSelectMode) {
      galleryManager.toggleOutputSelectMode();
    } else {
      State.outputSelectedIds.clear();
      galleryManager.updateOutputSelectionUI();
    }

    const canvas = $("#gen-canvas");
    canvas.innerHTML = "";
    State.outputImages = [];
    if (outputLightboxController) {
      outputLightboxController.resetOutputCollection();
    }
    const placeholder = el(
      "div",
      { className: "gen-placeholder", id: "gen-placeholder" },
      el("div", { className: "gen-placeholder__icon" }, "◈"),
      el("div", {}, "Configure your prompt and click "),
      el("div", { className: "u-muted", style: { fontSize: "var(--text-xs)" } }, "Images will appear here"),
    );
    canvas.appendChild(placeholder);
    galleryManager.updateOutputCount();
    galleryManager.updateOutputSelectionUI();
  });

  $("#btn-stats").addEventListener("click", galleryManager.openStatsModal);
  $("#modal-stats-close").addEventListener("click", () => $("#modal-stats").classList.add("hidden"));
  $("#modal-stats-close2").addEventListener("click", () => $("#modal-stats").classList.add("hidden"));
  $("#modal-stats-backdrop").addEventListener("click", () => $("#modal-stats").classList.add("hidden"));

  $("#btn-gallery-refresh").addEventListener("click", () => galleryManager.loadGallery(State.galleryPage));
  $("#btn-gallery-prev").addEventListener("click", () => galleryManager.loadGallery(State.galleryPage - 1));
  $("#btn-gallery-next").addEventListener("click", () => galleryManager.loadGallery(State.galleryPage + 1));
  $("#sel-gallery-model").addEventListener("change", () => {
    galleryManager.loadGallery(1);
  });

  $("#btn-gallery-select").addEventListener("click", galleryManager.toggleSelectMode);
  $("#btn-select-all").addEventListener("click", galleryManager.selectAllVisible);
  $("#btn-save-selected").addEventListener("click", galleryManager.downloadSelectedZip);
  $("#btn-delete-selected").addEventListener("click", galleryManager.bulkDelete);

  $("#btn-output-select").addEventListener("click", galleryManager.toggleOutputSelectMode);
  $("#btn-output-select-all").addEventListener("click", galleryManager.selectAllOutputVisible);
  $("#btn-output-save-all").addEventListener("click", galleryManager.downloadAllOutputZip);

  $("#btn-fav-refresh").addEventListener("click", () => galleryManager.loadFavourites(State.favPage));
  $("#btn-fav-prev").addEventListener("click", () => galleryManager.loadFavourites(State.favPage - 1));
  $("#btn-fav-next").addEventListener("click", () => galleryManager.loadFavourites(State.favPage + 1));

  document.addEventListener("keydown", event => {
    if (event.key === "Escape") {
      galleryManager.closeLightbox();
      $("#modal-stats").classList.add("hidden");
      return;
    }

    if (outputLightboxController && outputLightboxController.handleKeydown(event)) {
      return;
    }

    if (!event.altKey && !event.ctrlKey && !event.metaKey && !galleryManager.isTypingTargetActive()) {
      const galleryDirection = resolveGalleryPaginationDirection(event.key);
      const activeTabId = galleryManager.getActiveTabId();
      const statsModalOpen = !$("#modal-stats").classList.contains("hidden");

      if (galleryDirection !== 0 && activeTabId === "gallery" && !statsModalOpen) {
        if (galleryDirection === -1 && State.galleryPage > 1) {
          event.preventDefault();
          galleryManager.loadGallery(State.galleryPage - 1);
          return;
        }

        if (galleryDirection === 1 && State.galleryPage < State.galleryPages) {
          event.preventDefault();
          galleryManager.loadGallery(State.galleryPage + 1);
          return;
        }
      }
    }

    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      if (!State.isGenerating) generationFlow.generate();
    }
  });

  galleryManager.updateOutputSelectionUI();
}

async function init() {
  const savedTheme = localStorage.getItem("pw-theme") || "dark";
  applyTheme(savedTheme);
  initializeCollapsibleSections();

  setStatus("Loading config…", true);
  wireEvents();

  await runtimeGpuController.loadConfig();
  await runtimeGpuController.loadGpuSettings({ silent: true });

  try {
    if (!State.runtimeMode) {
      await runtimeGpuController.loadRuntimeMode();
    }
    if (!State.runtimeAuth) {
      await runtimeGpuController.refreshRuntimeAuthState({ silent: true });
    }
    runtimeGpuController.applyRuntimeControls();
    runtimeGpuController.applyPolicyPromptDropdowns();
  } catch (error) {
    toast(`Runtime controls unavailable: ${error.message}`, "warn");
  }

  setStatus("Ready");
  updateStatusBar();
}

document.addEventListener("DOMContentLoaded", init);
