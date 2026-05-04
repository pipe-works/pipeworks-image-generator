import {
  DEFAULT_SLOT_LABEL,
  PROMPT_SCHEMA_VERSION,
  defaultSlot,
  emptyTokenCounts,
  nextSlotId,
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
  let dragSourceId = null;

  const slotsContainer = $("#composer-slots");
  const slotTemplate = $("#composer-slot-template");
  const addSlotButton = $("#btn-add-slot");

  function findSlot(slotId) {
    return state.sections.find(s => s.id === slotId) || null;
  }

  function findSlotIndex(slotId) {
    return state.sections.findIndex(s => s.id === slotId);
  }

  function getSlotElement(slotId) {
    return slotsContainer.querySelector(`[data-slot-id="${slotId}"]`);
  }

  function ensureFloor() {
    if (state.sections.length === 0) {
      state.sections.push(defaultSlot());
    }
  }

  function optionMatchesSlotKind(option, slotKind) {
    if (!slotKind) return true;
    const kinds = Array.isArray(option.slot_kinds) ? option.slot_kinds : [];
    return kinds.includes(slotKind);
  }

  function renderPolicyOptions(selectEl, currentValue, slotKind) {
    selectEl.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = slotKind
      ? `— Add ${slotKind} snippet —`
      : "— Add snippet from policies —";
    selectEl.appendChild(placeholder);

    const options = (state.policyPromptOptions || []).filter(option =>
      optionMatchesSlotKind(option, slotKind)
    );

    if (slotKind) {
      // Filtered to a single slot kind: skip optgroup chrome — the list is
      // already coherent and short.
      if (options.length === 0) {
        const empty = document.createElement("option");
        empty.value = "";
        empty.disabled = true;
        empty.textContent = "— No snippets for this slot kind —";
        selectEl.appendChild(empty);
      } else {
        options.forEach(option => {
          const opt = document.createElement("option");
          opt.value = option.id;
          opt.textContent = option.label || option.id;
          selectEl.appendChild(opt);
        });
      }
      selectEl.value =
        currentValue && options.some(option => option.id === currentValue) ? currentValue : "";
      return;
    }

    const declaredGroups = state.policyPromptGroups || [];
    const grouped = new Map();
    declaredGroups.forEach(group => {
      if (group && !grouped.has(group)) grouped.set(group, []);
    });
    options.forEach(option => {
      const group = option.group || "policies";
      if (!grouped.has(group)) grouped.set(group, []);
      grouped.get(group).push(option);
    });

    [...grouped.keys()].sort().forEach(groupName => {
      const optgroup = document.createElement("optgroup");
      optgroup.label = groupName;
      const entries = grouped.get(groupName) || [];
      if (entries.length === 0) {
        const placeholderOpt = document.createElement("option");
        placeholderOpt.value = "";
        placeholderOpt.disabled = true;
        placeholderOpt.textContent = "— No prompt snippets in this directory —";
        optgroup.appendChild(placeholderOpt);
      } else {
        entries.forEach(option => {
          const opt = document.createElement("option");
          opt.value = option.id;
          opt.textContent = option.label || option.id;
          optgroup.appendChild(opt);
        });
      }
      selectEl.appendChild(optgroup);
    });

    selectEl.value = currentValue || "";
  }

  function renderSlotKindOptions(selectEl, currentValue) {
    selectEl.innerHTML = "";
    const freeForm = document.createElement("option");
    freeForm.value = "";
    freeForm.textContent = "Free-form (all snippets)";
    selectEl.appendChild(freeForm);
    (state.policyPromptSlotKinds || []).forEach(kind => {
      const opt = document.createElement("option");
      opt.value = kind;
      opt.textContent = kind;
      selectEl.appendChild(opt);
    });
    selectEl.value = currentValue || "";
  }

  function applySlotMode(slotEl, mode) {
    slotEl.querySelectorAll(".composer-slot__mode").forEach(btn => {
      btn.classList.toggle("is-active", btn.dataset.mode === mode);
    });
    slotEl.querySelector(".composer-slot__auto-wrap").style.display =
      mode === "automated" ? "" : "none";
  }

  function renderSlot(slot) {
    const fragment = slotTemplate.content.cloneNode(true);
    const slotEl = fragment.querySelector(".composer-slot");
    slotEl.dataset.slotId = slot.id;
    slotEl.draggable = true;

    const labelInput = slotEl.querySelector(".composer-slot__label-input");
    labelInput.value = slot.label;

    const textarea = slotEl.querySelector(".composer-slot__textarea");
    textarea.value = slot.manualText || "";

    const kindSelect = slotEl.querySelector(".composer-slot__kind-select");
    if (kindSelect) {
      renderSlotKindOptions(kindSelect, slot.slotKind);
    }

    const select = slotEl.querySelector(".composer-slot__select");
    renderPolicyOptions(select, slot.selectedPolicyId, slot.slotKind);

    applySlotMode(slotEl, slot.mode);

    bindSlotEvents(slotEl, slot);
    return slotEl;
  }

  function rerenderSlots() {
    slotsContainer.innerHTML = "";
    state.sections.forEach(slot => {
      slotsContainer.appendChild(renderSlot(slot));
    });
    updateDeleteButtons();
  }

  function updateDeleteButtons() {
    const onlyOne = state.sections.length <= 1;
    slotsContainer.querySelectorAll(".composer-slot__delete").forEach(btn => {
      btn.disabled = onlyOne;
      btn.title = onlyOne ? "At least one slot is required" : "Remove slot";
    });
  }

  function bindSlotEvents(slotEl, slot) {
    const labelInput = slotEl.querySelector(".composer-slot__label-input");
    labelInput.addEventListener("input", () => {
      slot.label = labelInput.value;
      schedulePromptPreview();
    });

    slotEl.querySelectorAll(".composer-slot__mode").forEach(btn => {
      btn.addEventListener("click", () => {
        slot.mode = btn.dataset.mode;
        applySlotMode(slotEl, slot.mode);
        schedulePromptPreview();
      });
    });

    const kindSelect = slotEl.querySelector(".composer-slot__kind-select");
    if (kindSelect) {
      kindSelect.addEventListener("change", () => {
        const nextKind = (kindSelect.value || "").trim() || null;
        slot.slotKind = nextKind;
        const select = slotEl.querySelector(".composer-slot__select");
        renderPolicyOptions(select, slot.selectedPolicyId, slot.slotKind);
        if (slot.selectedPolicyId) {
          const stillVisible = (state.policyPromptOptions || []).some(
            option =>
              option.id === slot.selectedPolicyId &&
              optionMatchesSlotKind(option, slot.slotKind)
          );
          if (!stillVisible) {
            slot.selectedPolicyId = null;
            select.value = "";
          }
        }
      });
    }

    const select = slotEl.querySelector(".composer-slot__select");
    select.addEventListener("change", () => {
      const optionId = select.value;
      if (!optionId) {
        slot.selectedPolicyId = null;
        return;
      }
      slot.selectedPolicyId = optionId;
      const option = (state.policyPromptOptions || []).find(item => item.id === optionId);
      if (option) {
        const snippet = (option.value || "").trim();
        const textarea = slotEl.querySelector(".composer-slot__textarea");
        textarea.value = snippet;
        slot.manualText = snippet;
        const optionLabel = (option.label || "").trim();
        if (optionLabel) {
          slot.label = optionLabel;
          slotEl.querySelector(".composer-slot__label-input").value = optionLabel;
        }
      }
      schedulePromptPreview();
    });

    const textarea = slotEl.querySelector(".composer-slot__textarea");
    textarea.addEventListener("input", () => {
      slot.manualText = textarea.value;
      schedulePromptPreview();
    });

    slotEl.querySelector(".composer-slot__copy").addEventListener("click", async event => {
      const text = (slot.manualText || "").trim();
      if (!text) {
        toast(`No ${slot.label || DEFAULT_SLOT_LABEL} text to copy`, "info");
        return;
      }
      if (!navigator.clipboard?.writeText) {
        toast("Clipboard copy unavailable", "err");
        return;
      }
      try {
        await navigator.clipboard.writeText(text);
        flashButtonLabel(event.currentTarget, "Copied", "Copy");
      } catch (_) {
        toast("Clipboard copy failed", "err");
      }
    });

    slotEl.querySelector(".composer-slot__delete").addEventListener("click", () => {
      if (state.sections.length <= 1) return;
      const idx = findSlotIndex(slot.id);
      if (idx >= 0) {
        state.sections.splice(idx, 1);
        rerenderSlots();
        schedulePromptPreview();
      }
    });

    slotEl.addEventListener("dragstart", event => {
      dragSourceId = slot.id;
      slotEl.classList.add("composer-slot--dragging");
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", slot.id);
    });

    slotEl.addEventListener("dragend", () => {
      dragSourceId = null;
      slotEl.classList.remove("composer-slot--dragging");
      slotsContainer
        .querySelectorAll(".composer-slot--drop-target")
        .forEach(el => el.classList.remove("composer-slot--drop-target"));
    });

    slotEl.addEventListener("dragover", event => {
      if (!dragSourceId || dragSourceId === slot.id) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      slotEl.classList.add("composer-slot--drop-target");
    });

    slotEl.addEventListener("dragleave", () => {
      slotEl.classList.remove("composer-slot--drop-target");
    });

    slotEl.addEventListener("drop", event => {
      event.preventDefault();
      const sourceId = dragSourceId || event.dataTransfer.getData("text/plain");
      slotEl.classList.remove("composer-slot--drop-target");
      if (!sourceId || sourceId === slot.id) return;
      const fromIdx = findSlotIndex(sourceId);
      const toIdx = findSlotIndex(slot.id);
      if (fromIdx < 0 || toIdx < 0) return;
      const [moved] = state.sections.splice(fromIdx, 1);
      state.sections.splice(toIdx, 0, moved);
      rerenderSlots();
      schedulePromptPreview();
    });
  }

  function bindAddSlot() {
    if (!addSlotButton) return;
    addSlotButton.addEventListener("click", () => {
      state.sections.push(defaultSlot());
      rerenderSlots();
      schedulePromptPreview();
    });
  }

  function refreshPolicyDropdowns() {
    state.sections.forEach(slot => {
      const slotEl = getSlotElement(slot.id);
      if (!slotEl) return;
      const kindSelect = slotEl.querySelector(".composer-slot__kind-select");
      if (kindSelect) {
        renderSlotKindOptions(kindSelect, slot.slotKind);
      }
      const select = slotEl.querySelector(".composer-slot__select");
      renderPolicyOptions(select, slot.selectedPolicyId, slot.slotKind);
    });
  }

  function estimateTokens(text) {
    const trimmed = (text || "").trim();
    if (!trimmed) return 0;
    return Math.ceil(trimmed.length / 4);
  }

  function applyTotalCounter(count, limit) {
    const counter = $("#total-tokens");
    if (!counter) return;
    counter.textContent = `${count} / ${limit} tokens`;
    counter.classList.remove("token-counter--warn", "token-counter--over");
    if (count > limit) {
      counter.classList.add("token-counter--over");
    } else if (count > limit * 0.85) {
      counter.classList.add("token-counter--warn");
    }
  }

  function applySlotCounter(slotId, count, limit) {
    const slotEl = getSlotElement(slotId);
    if (!slotEl) return;
    const counter = slotEl.querySelector(".composer-slot__tokens");
    if (!counter) return;
    counter.textContent = `${count} / ${limit} tokens`;
    counter.classList.remove("token-counter--warn", "token-counter--over");
    if (count > limit) {
      counter.classList.add("token-counter--over");
    } else if (count > limit * 0.85) {
      counter.classList.add("token-counter--warn");
    }
  }

  function updateTokenCounters() {
    if (!state.config || !state.selectedModel) return;
    const model = state.config.models.find(m => m.id === state.selectedModel);
    const maxTokens = model ? model.max_prompt_tokens || 77 : 77;
    const sectionCounts = (state.tokenCounts && state.tokenCounts.sections) || [];

    state.sections.forEach((slot, index) => {
      const entry = sectionCounts[index];
      const tokens = entry?.tokens ?? estimateTokens(slot.manualText);
      slot.tokens = tokens;
      applySlotCounter(slot.id, tokens, maxTokens);
    });

    const total = state.tokenCounts?.total ?? 0;
    applyTotalCounter(total, maxTokens);
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
      const counts = data.token_counts || {};
      state.tokenCounts = {
        sections: Array.isArray(counts.sections) ? counts.sections : [],
        total: counts.total ?? estimateTokens(data.compiled_prompt || ""),
        method: counts.method || "heuristic",
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

    const sections = state.sections.map(slot => ({
      label: (slot.label || DEFAULT_SLOT_LABEL).trim() || DEFAULT_SLOT_LABEL,
      mode: slot.mode || "manual",
      manual_text: (slot.manualText || "").trim() || null,
      automated_prompt_id:
        slot.mode === "automated" && slot.selectedPolicyId ? slot.selectedPolicyId : null,
    }));

    const payload = {
      model_id: state.selectedModel,
      gpu_worker_id: getSelectedGpuWorker()?.id || null,
      prompt_schema_version: PROMPT_SCHEMA_VERSION,
      aspect_ratio_id: aspectId,
      width: ar.width,
      height: ar.height,
      steps: parseInt($("#rng-steps").value, 10),
      guidance: parseFloat($("#rng-guidance").value),
      seed: seedVal,
      batch_size: state.batchSize,
      sections,
    };

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

  function init() {
    ensureFloor();
    rerenderSlots();
    bindAddSlot();
  }

  init();

  // Reset slot ids on next render so dynamically created slots don't collide
  // when this module is re-initialised in tests.
  void nextSlotId;

  return {
    rerenderSlots,
    refreshPolicyDropdowns,
    updateTokenCounters,
    schedulePromptPreview,
    updatePromptPreview,
    buildGeneratePayload,
  };
}
