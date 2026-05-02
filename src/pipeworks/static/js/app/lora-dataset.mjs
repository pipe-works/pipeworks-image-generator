/**
 * LoRA Dataset tab controller.
 *
 * Three-step flow on a single tab:
 *   1. snapshot the prompt-v3 composer state (the consistency stack);
 *   2. select canonical `location` snippets to use as the diversity axis;
 *   3. create a run and drive per-tile generation, regen, exclusion, export.
 *
 * The snapshot model is intentionally manual (operator clicks "snapshot"
 * rather than the tab continuously mirroring the composer). It's the
 * cheapest UX that captures intent without coupling state across tabs;
 * we'll iterate after real use shows what's actually annoying.
 */

const POLL_INTERVAL_MS = 1500;

export function createLoraDatasetController({ apiClient, toast, buildGeneratePayload }) {
  const $ = sel => document.querySelector(sel);

  let snapshotPayload = null;
  let availableLocations = [];
  const selectedLocationIds = new Set();
  let activeRunId = null;
  let pollHandle = null;
  let pollGeneration = 0;

  function setText(sel, value) {
    const node = $(sel);
    if (node) node.textContent = value;
  }

  function showActiveRunPanel(visible) {
    const panel = $("#lora-active-run-panel");
    if (panel) panel.style.display = visible ? "" : "none";
  }

  function refreshCreateButtonEnabled() {
    const btn = $("#lora-btn-create");
    if (!btn) return;
    btn.disabled = !snapshotPayload || selectedLocationIds.size === 0;
  }

  /* ------------------------- snapshot ------------------------- */

  function takeSnapshot() {
    const payload = buildGeneratePayload();
    if (!payload) {
      toast("Composer is not ready — pick a model and aspect ratio first.", "err");
      return;
    }
    snapshotPayload = payload;
    setText("#lora-snapshot-status", `${(payload.sections || []).length} section(s) snapshotted`);
    renderSnapshotSummary();
    refreshCreateButtonEnabled();
  }

  function renderSnapshotSummary() {
    const root = $("#lora-snapshot-summary");
    if (!root) return;
    root.innerHTML = "";
    if (!snapshotPayload) return;

    const sections = snapshotPayload.sections || [];
    if (sections.length === 0) {
      const empty = document.createElement("div");
      empty.className = "lora-snapshot-empty";
      empty.textContent = "Snapshot has no sections — the composer was empty.";
      root.appendChild(empty);
      return;
    }

    sections.forEach(section => {
      const row = document.createElement("div");
      row.className = "lora-snapshot-row";

      const label = document.createElement("span");
      label.className = "lora-snapshot-row__label";
      label.textContent = section.label || "Policy";

      const text = document.createElement("span");
      text.className = "lora-snapshot-row__text";
      text.textContent = (section.manual_text || "").trim() || "(empty)";

      row.appendChild(label);
      row.appendChild(text);
      root.appendChild(row);
    });
  }

  /* ------------------------- locations ------------------------- */

  async function loadLocations() {
    setText("#lora-locations-status", "Loading…");
    let data;
    try {
      data = await apiClient.fetchPolicyPrompts();
    } catch (err) {
      setText("#lora-locations-status", "Failed to load policy snippets.");
      toast(`Could not load locations: ${err.message || err}`, "err");
      return;
    }
    const options = data?.policy_prompt_options || [];
    availableLocations = options
      .filter(opt => typeof opt.id === "string" && opt.id.startsWith("location:"))
      .sort((a, b) => (a.label || "").localeCompare(b.label || ""));

    if (availableLocations.length === 0) {
      setText(
        "#lora-locations-status",
        "No location snippets visible. Sign in to mud-server and ensure locations are published."
      );
    } else {
      setText("#lora-locations-status", `${availableLocations.length} location(s) available`);
    }

    selectedLocationIds.clear();
    renderLocationsList();
    refreshCreateButtonEnabled();
  }

  function renderLocationsList() {
    const root = $("#lora-locations-list");
    if (!root) return;
    root.innerHTML = "";

    availableLocations.forEach(opt => {
      const row = document.createElement("label");
      row.className = "lora-location-row";

      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = opt.id;
      checkbox.checked = selectedLocationIds.has(opt.id);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) selectedLocationIds.add(opt.id);
        else selectedLocationIds.delete(opt.id);
        refreshCreateButtonEnabled();
      });

      const label = document.createElement("span");
      label.className = "lora-location-row__label";
      label.textContent = opt.label || opt.id;

      const preview = document.createElement("span");
      preview.className = "lora-location-row__preview";
      preview.textContent = opt.value || "";

      row.appendChild(checkbox);
      row.appendChild(label);
      row.appendChild(preview);
      root.appendChild(row);
    });
  }

  function selectAllLocations() {
    availableLocations.forEach(opt => selectedLocationIds.add(opt.id));
    renderLocationsList();
    refreshCreateButtonEnabled();
  }

  function clearLocationSelection() {
    selectedLocationIds.clear();
    renderLocationsList();
    refreshCreateButtonEnabled();
  }

  /* ------------------------- runs ------------------------- */

  async function createRun() {
    if (!snapshotPayload) {
      toast("Snapshot the composer first.", "err");
      return;
    }
    if (selectedLocationIds.size === 0) {
      toast("Select at least one location.", "err");
      return;
    }

    const body = {
      model_id: snapshotPayload.model_id,
      aspect_ratio_id: snapshotPayload.aspect_ratio_id,
      width: snapshotPayload.width,
      height: snapshotPayload.height,
      steps: snapshotPayload.steps,
      guidance: snapshotPayload.guidance,
      scheduler: snapshotPayload.scheduler || null,
      seed: snapshotPayload.seed ?? null,
      negative_prompt: snapshotPayload.negative_prompt || null,
      pinned_sections: snapshotPayload.sections || [],
      location_section_label: "Location",
      location_policy_ids: Array.from(selectedLocationIds),
    };

    setText("#lora-create-status", "Creating run…");
    let manifest;
    try {
      manifest = await apiClient.createLoraRun(body);
    } catch (err) {
      setText("#lora-create-status", "Failed.");
      toast(`Run creation failed: ${err.message || err}`, "err");
      return;
    }
    setText("#lora-create-status", `Run ${manifest.run_id} created.`);
    await openRun(manifest.run_id);
    await loadRunList();
  }

  async function openRun(runId) {
    activeRunId = runId;
    showActiveRunPanel(true);
    setText("#lora-active-run-id", runId);
    await refreshActiveRun();
  }

  async function refreshActiveRun() {
    if (!activeRunId) return;
    let manifest;
    try {
      manifest = await apiClient.getLoraRun(activeRunId);
    } catch (err) {
      toast(`Could not load run: ${err.message || err}`, "err");
      return;
    }
    renderActiveRun(manifest);
  }

  function renderActiveRun(manifest) {
    setText("#lora-active-run-status", manifest.status);
    const order = manifest.slot_order || [];
    const slots = manifest.slots || {};
    const done = order.filter(k => slots[k]?.status === "done").length;
    const total = order.length;
    setText("#lora-active-run-progress", `${done} / ${total} tile(s) done`);

    const cancelBtn = $("#lora-btn-cancel");
    if (cancelBtn) cancelBtn.disabled = manifest.status !== "running";
    const generateBtn = $("#lora-btn-generate");
    if (generateBtn) generateBtn.disabled = manifest.status === "running";
    // Reflect the manifest's seed strategy on the toggle and disable while
    // the run is generating — toggling mid-flight would race the per-tile
    // seed reads in the backend loop.
    const seedToggle = $("#lora-chk-shared-seed");
    if (seedToggle) {
      seedToggle.checked = manifest.params?.share_seed_across_tiles !== false;
      seedToggle.disabled = manifest.status === "running";
    }

    const grid = $("#lora-tile-grid");
    if (!grid) return;
    grid.innerHTML = "";
    order.forEach((key, index) => {
      const slot = slots[key];
      if (!slot) return;
      grid.appendChild(renderTile(manifest, index, key, slot));
    });
  }

  function renderTile(manifest, index, key, slot) {
    const card = document.createElement("div");
    card.className = "lora-tile";
    card.dataset.slotKey = key;
    card.dataset.status = slot.status;
    if (slot.excluded) card.classList.add("is-excluded");

    const header = document.createElement("div");
    header.className = "lora-tile__header";
    header.innerHTML = `<strong>${String(index + 1).padStart(2, "0")} · ${slot.location_label}</strong>
      <span class="lora-tile__status">${slot.status}</span>`;

    const body = document.createElement("div");
    body.className = "lora-tile__body";
    if (slot.image_filename && slot.status === "done") {
      const img = document.createElement("img");
      img.alt = slot.location_label;
      img.src = `/api/lora-dataset/runs/${manifest.run_id}/files/${encodeURIComponent(
        slot.image_filename
      )}`;
      body.appendChild(img);
    } else if (slot.error) {
      const err = document.createElement("div");
      err.className = "lora-tile__error";
      err.textContent = slot.error;
      body.appendChild(err);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "lora-tile__placeholder";
      placeholder.textContent = slot.status === "running" ? "Generating…" : "Pending";
      body.appendChild(placeholder);
    }

    // Surface the per-tile seed inline so the operator can correlate
    // visible drift with seed bumps after a regen, and so a "bad" tile
    // can be reproduced or rejected with seed-level evidence.
    const seedLine = document.createElement("div");
    seedLine.className = "lora-tile__seed";
    seedLine.textContent = slot.seed != null ? `seed ${slot.seed}` : "seed —";

    const caption = document.createElement("div");
    caption.className = "lora-tile__caption";
    caption.textContent = slot.location_text;

    const actions = document.createElement("div");
    actions.className = "lora-tile__actions";

    const regenBtn = document.createElement("button");
    regenBtn.className = "btn btn--secondary btn--sm";
    regenBtn.type = "button";
    regenBtn.textContent = "Regen";
    regenBtn.disabled = manifest.status === "running";
    regenBtn.addEventListener("click", () => regenerateTile(key));

    const excludeBtn = document.createElement("button");
    excludeBtn.className = "btn btn--secondary btn--sm";
    excludeBtn.type = "button";
    excludeBtn.textContent = slot.excluded ? "Include" : "Exclude";
    excludeBtn.addEventListener("click", () => togglePropExcluded(key, !slot.excluded));

    actions.appendChild(regenBtn);
    actions.appendChild(excludeBtn);

    card.appendChild(header);
    card.appendChild(body);
    card.appendChild(seedLine);
    card.appendChild(caption);
    card.appendChild(actions);
    return card;
  }

  async function generateActiveRun() {
    if (!activeRunId) return;
    setText("#lora-active-run-status", "running");
    startPolling();
    try {
      await apiClient.generateLoraRun(activeRunId);
    } catch (err) {
      toast(`Generation failed: ${err.message || err}`, "err");
    } finally {
      stopPolling();
      await refreshActiveRun();
    }
  }

  async function cancelActiveRun() {
    if (!activeRunId) return;
    try {
      await apiClient.cancelLoraRun(activeRunId);
      toast("Cancellation requested.", "info");
    } catch (err) {
      toast(`Cancel failed: ${err.message || err}`, "err");
    }
    await refreshActiveRun();
  }

  async function regenerateTile(slotKey) {
    if (!activeRunId) return;
    startPolling();
    try {
      await apiClient.regenerateLoraSlot(activeRunId, slotKey);
    } catch (err) {
      toast(`Regen failed: ${err.message || err}`, "err");
    } finally {
      stopPolling();
      await refreshActiveRun();
    }
  }

  async function togglePropExcluded(slotKey, excluded) {
    if (!activeRunId) return;
    try {
      await apiClient.patchLoraSlot(activeRunId, slotKey, { excluded });
    } catch (err) {
      toast(`Update failed: ${err.message || err}`, "err");
      return;
    }
    await refreshActiveRun();
  }

  async function exportActiveDataset() {
    if (!activeRunId) return;
    try {
      const result = await apiClient.exportLoraDataset(activeRunId);
      toast(
        `Exported ${result.pairs_copied} pair(s); ` +
          `excluded ${result.excluded}; skipped ${result.skipped}.`,
        "ok"
      );
    } catch (err) {
      toast(`Export failed: ${err.message || err}`, "err");
    }
  }

  /* ------------------------- polling ------------------------- */

  function startPolling() {
    pollGeneration += 1;
    const myGen = pollGeneration;
    if (pollHandle) clearInterval(pollHandle);
    pollHandle = setInterval(() => {
      if (myGen !== pollGeneration) return;
      refreshActiveRun().catch(() => {});
    }, POLL_INTERVAL_MS);
  }

  function stopPolling() {
    if (pollHandle) {
      clearInterval(pollHandle);
      pollHandle = null;
    }
  }

  /* ------------------------- run list ------------------------- */

  async function loadRunList() {
    let data;
    try {
      data = await apiClient.listLoraRuns();
    } catch (err) {
      toast(`Could not list runs: ${err.message || err}`, "err");
      return;
    }
    const root = $("#lora-run-list");
    if (!root) return;
    root.innerHTML = "";
    const runs = data?.runs || [];
    if (runs.length === 0) {
      const empty = document.createElement("div");
      empty.className = "lora-run-list__empty";
      empty.textContent = "No runs yet.";
      root.appendChild(empty);
      return;
    }
    runs.forEach(manifest => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "lora-run-list__item";
      const created = new Date((manifest.created_at || 0) * 1000).toLocaleString();
      const slotCount = (manifest.slot_order || []).length;
      item.textContent = `${manifest.run_id.slice(0, 8)} · ${manifest.status} · ${slotCount} tile(s) · ${created}`;
      item.addEventListener("click", () => openRun(manifest.run_id));
      root.appendChild(item);
    });
  }

  /* ------------------------- bind ------------------------- */

  async function onSharedSeedToggleChange(event) {
    if (!activeRunId) return;
    const desired = !!event.target.checked;
    try {
      await apiClient.patchLoraRun(activeRunId, { share_seed_across_tiles: desired });
    } catch (err) {
      toast(`Could not change seed strategy: ${err.message || err}`, "err");
      return;
    }
    await refreshActiveRun();
  }

  function bind() {
    $("#lora-btn-snapshot")?.addEventListener("click", takeSnapshot);
    $("#lora-btn-select-all")?.addEventListener("click", selectAllLocations);
    $("#lora-btn-select-none")?.addEventListener("click", clearLocationSelection);
    $("#lora-btn-create")?.addEventListener("click", createRun);
    $("#lora-btn-generate")?.addEventListener("click", generateActiveRun);
    $("#lora-btn-cancel")?.addEventListener("click", cancelActiveRun);
    $("#lora-btn-export")?.addEventListener("click", exportActiveDataset);
    $("#lora-chk-shared-seed")?.addEventListener("change", onSharedSeedToggleChange);
  }

  bind();

  return {
    onTabActivated() {
      loadLocations();
      loadRunList();
    },
  };
}
