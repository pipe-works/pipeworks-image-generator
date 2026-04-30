import {
  PROMPT_SECTIONS,
  PROMPT_SECTION_LABELS,
  emptyTokenCounts,
} from "./state.mjs";

export function createPromptComposer({
  state,
  $,
  apiClient,
  toast,
  flashButtonLabel,
  getSelectedGpuWorker,
}) {
  let previewDebounce = null;

  function setSectionPromptMode(section, mode) {
    state.sectionModes[section] = mode;
    $(`#btn-${section}-automated`).classList.toggle("is-active", mode === "automated");
    $(`#btn-${section}-manual`).classList.toggle("is-active", mode === "manual");
    $(`#${section}-auto-wrap`).style.display = mode === "automated" ? "" : "none";
    updateTokenCounters();
    schedulePromptPreview();
  }

  function appendPolicySnippetToSection(section, optionId) {
    if (!optionId) return;
    const option = (state.policyPromptOptions || []).find(item => item.id === optionId);
    if (!option || !(option.value || "").trim()) return;

    const textarea = $(`#txt-${section}`);
    if (!textarea) return;

    const existing = textarea.value.trimEnd();
    const snippet = option.value.trim();
    textarea.value = existing ? `${existing}\n${snippet}` : snippet;

    const select = $(`#sel-${section}`);
    if (select) select.value = optionId;
    updateTokenCounters();
    schedulePromptPreview();
  }

  function estimateTokens(text) {
    const trimmed = text.trim();
    if (!trimmed) return 0;
    return Math.ceil(trimmed.length / 4);
  }

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

  function updateTokenCounters() {
    if (!state.config || !state.selectedModel) return;

    const model = state.config.models.find(m => m.id === state.selectedModel);
    const maxTokens = model ? (model.max_prompt_tokens || 77) : 77;
    const counts = state.tokenCounts || {};

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

  function schedulePromptPreview() {
    clearTimeout(previewDebounce);
    previewDebounce = setTimeout(updatePromptPreview, 400);
  }

  async function updatePromptPreview() {
    if (!state.config) return;

    const payload = buildGeneratePayload();
    if (!payload) {
      state.tokenCounts = emptyTokenCounts();
      updateTokenCounters();
      return;
    }

    try {
      const data = await apiClient.compilePrompt(payload);
      state.tokenCounts = data.token_counts || {
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
      // Silent fail for preview.
    }
  }

  function buildGeneratePayload() {
    if (!state.config || !state.selectedModel) return null;

    const model = state.config.models.find(m => m.id === state.selectedModel);
    if (!model) return null;

    const aspectId = $("#sel-aspect").value;
    const ar = model.aspect_ratios.find(a => a.id === aspectId);
    if (!ar) return null;

    const isRandom = $("#chk-random-seed").checked;
    const seedVal = isRandom ? null : parseInt($("#inp-seed").value, 10) || null;

    const payload = {
      model_id: state.selectedModel,
      gpu_worker_id: getSelectedGpuWorker()?.id || null,
      prompt_schema_version: 2,
      aspect_ratio_id: aspectId,
      width: ar.width,
      height: ar.height,
      steps: parseInt($("#rng-steps").value, 10),
      guidance: parseFloat($("#rng-guidance").value),
      seed: seedVal,
      batch_size: state.batchSize,
    };

    PROMPT_SECTIONS.forEach(section => {
      const mode = state.sectionModes[section] || "manual";
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

    if (model.schedulers && model.schedulers.length > 0) {
      const schedVal = $("#sel-scheduler").value;
      if (schedVal) payload.scheduler = schedVal;
    }

    return payload;
  }

  return {
    setSectionPromptMode,
    appendPolicySnippetToSection,
    copyPromptSection,
    updateTokenCounters,
    schedulePromptPreview,
    updatePromptPreview,
    buildGeneratePayload,
  };
}
