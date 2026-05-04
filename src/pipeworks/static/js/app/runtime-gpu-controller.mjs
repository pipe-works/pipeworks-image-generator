import { DEFAULT_MODEL_ID } from "./state.mjs";

export function createRuntimeGpuController({
  state,
  $,
  el,
  apiClient,
  setStatus,
  toast,
  updateStatusBar,
  updatePromptPreview,
  updateTokenCounters,
  setGpuSettingsStatus,
  refreshComposerPolicyDropdowns,
}) {
  function syncGpuSettingsControls() {
    const useRemote = $("#chk-use-remote-gpu")?.checked || false;
    [
      "inp-remote-gpu-url",
      "inp-remote-gpu-token",
      "btn-gpu-token-generate",
      "btn-gpu-test",
    ].forEach(id => {
      const control = $(`#${id}`);
      if (control) control.disabled = !useRemote;
    });
  }

  function randomGpuToken() {
    const bytes = new Uint8Array(24);
    (globalThis.crypto || window.crypto).getRandomValues(bytes);
    return [...bytes].map(b => b.toString(16).padStart(2, "0")).join("");
  }

  function applyGpuSettingsForm(payload) {
    const useRemote = Boolean(payload?.use_remote_gpu);
    const checkbox = $("#chk-use-remote-gpu");
    const remoteUrl = $("#inp-remote-gpu-url");
    const remoteToken = $("#inp-remote-gpu-token");
    if (checkbox) checkbox.checked = useRemote;
    if (remoteUrl) remoteUrl.value = payload?.remote_base_url || "";

    if (remoteToken) {
      if (payload?.generated_bearer_token) {
        remoteToken.value = payload.generated_bearer_token;
        remoteToken.placeholder = "Generated token (copy to remote worker)";
      } else if (payload?.has_bearer_token) {
        remoteToken.value = "";
        remoteToken.placeholder = "Saved token hidden (leave blank to keep)";
      } else {
        remoteToken.value = "";
        remoteToken.placeholder = "Bearer token";
      }
    }

    syncGpuSettingsControls();
  }

  function applyPolicyPromptDropdowns() {
    if (typeof refreshComposerPolicyDropdowns === "function") {
      refreshComposerPolicyDropdowns();
    }
  }

  function runtimeModeLabel() {
    const modeKey = state.runtimeMode?.mode_key || "";
    const option = (state.runtimeMode?.options || []).find(candidate => candidate.mode_key === modeKey);
    return option?.label || modeKey || "Unknown";
  }

  function runtimeActiveServerUrl() {
    const runtimeMode = state.runtimeMode;
    const activeUrl = (runtimeMode?.active_server_url || "").trim();
    if (activeUrl) {
      return activeUrl;
    }
    const activeOption = (runtimeMode?.options || []).find(
      option => option.mode_key === runtimeMode?.mode_key,
    );
    return (activeOption?.default_server_url || "").trim();
  }

  function runtimeAuthStatus() {
    return String(state.runtimeAuth?.status || "");
  }

  function describeRuntimeServer(url) {
    const normalized = String(url || "").trim();
    if (!normalized) return "Policy API URL unavailable";
    try {
      const parsed = new URL(normalized);
      if (["127.0.0.1", "localhost", "::1"].includes(parsed.hostname)) {
        return `Luminal-local mud API ${normalized}`;
      }
    } catch {
      // Keep the raw URL if parsing fails.
    }
    return `Policy API ${normalized}`;
  }

  function isRuntimeSessionAuthorized() {
    return Boolean(state.runtimeAuth?.access_granted);
  }

  function updateRuntimeSourceStatusLine() {
    const sourceStatus = $("#runtime-source-status");
    if (!sourceStatus) return;

    const serverUrl = runtimeActiveServerUrl();
    const snippetCount = (state.policyPromptOptions || []).length;
    const snippetText = `${snippetCount} snippet${snippetCount === 1 ? "" : "s"}`;
    sourceStatus.textContent = `${describeRuntimeServer(serverUrl)} · ${snippetText}`;
  }

  function applyRuntimeControls() {
    const modeSelect = $("#runtime-mode-select");
    const modeUrl = $("#runtime-mode-url");
    const modeApply = $("#runtime-mode-apply");
    const loginUsername = $("#runtime-login-username");
    const loginPassword = $("#runtime-login-password");
    const loginApply = $("#runtime-login-apply");
    const modeBadge = $("#runtime-mode-badge");
    const authBadge = $("#runtime-auth-badge");

    if (!modeSelect || !modeUrl || !modeApply || !loginUsername || !loginPassword || !loginApply) {
      updateRuntimeSourceStatusLine();
      return;
    }

    const runtimeMode = state.runtimeMode;
    modeSelect.innerHTML = "";
    (runtimeMode?.options || []).forEach(option => {
      modeSelect.appendChild(el("option", { value: option.mode_key }, option.label));
    });
    if (runtimeMode?.mode_key && modeSelect.querySelector(`option[value="${runtimeMode.mode_key}"]`)) {
      modeSelect.value = runtimeMode.mode_key;
    }

    const activeOption = (runtimeMode?.options || []).find(
      option => option.mode_key === runtimeMode?.mode_key,
    );
    const activeServerUrl = (runtimeMode?.active_server_url || "").trim();
    const defaultServerUrl = (activeOption?.default_server_url || "").trim();
    modeUrl.value = activeServerUrl || defaultServerUrl;
    modeApply.disabled = !(modeUrl.value || "").trim();

    if (isRuntimeSessionAuthorized()) {
      loginApply.textContent = "Logout";
      loginApply.disabled = false;
    } else {
      loginApply.textContent = "Login";
      loginApply.disabled = !(loginUsername.value || "").trim() || !(loginPassword.value || "").trim();
    }

    if (modeBadge) {
      modeBadge.classList.remove("badge--muted", "badge--active", "badge--info");
      modeBadge.classList.add(runtimeMode?.mode_key === "server_dev" ? "badge--active" : "badge--info");
      modeBadge.textContent = runtimeMode ? `${runtimeModeLabel()} · Policy API` : "Mode unavailable";
    }

    if (authBadge) {
      authBadge.classList.remove(
        "badge--muted",
        "badge--active",
        "badge--info",
        "badge--warn",
        "badge--err",
      );
      const authStatus = runtimeAuthStatus();
      if (!authStatus) {
        authBadge.classList.add("badge--muted");
        authBadge.textContent = "Session Pending";
      } else if (authStatus === "authorized") {
        authBadge.classList.add("badge--active");
        authBadge.textContent = "Session Ready";
      } else if (authStatus === "missing_session") {
        authBadge.classList.add("badge--warn");
        authBadge.textContent = "No Runtime Session";
      } else if (authStatus === "forbidden") {
        authBadge.classList.add("badge--err");
        authBadge.textContent = "Runtime Role Denied";
      } else if (authStatus === "unauthenticated") {
        authBadge.classList.add("badge--warn");
        authBadge.textContent = "Runtime Session Invalid";
      } else {
        authBadge.classList.add("badge--err");
        authBadge.textContent = "Runtime Auth Error";
      }
    }

    updateRuntimeSourceStatusLine();
  }

  async function loadRuntimeMode() {
    state.runtimeMode = await apiClient.getRuntimeMode();
    applyRuntimeControls();
  }

  async function refreshRuntimeAuthState({ silent = false } = {}) {
    try {
      state.runtimeAuth = await apiClient.getRuntimeAuth();
    } catch (error) {
      state.runtimeAuth = {
        status: "error",
        access_granted: false,
        detail: error.message,
      };
      if (!silent) {
        setStatus(`Runtime auth failed: ${error.message}`);
      }
    }
    applyRuntimeControls();
    return state.runtimeAuth;
  }

  async function loadPolicyPrompts({ silent = false } = {}) {
    try {
      const payload = await apiClient.getPolicyPrompts();
      state.policyPromptOptions = payload.policy_prompt_options || [];
      state.policyPromptGroups = payload.policy_prompt_groups || [];
      state.policyPromptSlotKinds = payload.policy_prompt_slot_kinds || [];
      if (payload.runtime_auth) {
        state.runtimeAuth = payload.runtime_auth;
      }
      applyPolicyPromptDropdowns();
      applyRuntimeControls();
      if (!silent) {
        const source = runtimeModeLabel();
        setStatus(`Loaded ${state.policyPromptOptions.length} snippet(s) from ${source}.`);
      }
    } catch (error) {
      state.policyPromptOptions = [];
      state.policyPromptGroups = [];
      state.policyPromptSlotKinds = [];
      applyPolicyPromptDropdowns();
      applyRuntimeControls();
      if (!silent) {
        setStatus(`Policy snippets unavailable: ${error.message}`);
      }
    }
  }

  async function setRuntimeMode(modeKey, { explicitServerUrl = null } = {}) {
    const requestPayload = { mode_key: modeKey };
    if (explicitServerUrl !== null) {
      const serverUrl = String(explicitServerUrl || "").trim();
      if (serverUrl) requestPayload.server_url = serverUrl;
    }
    state.runtimeMode = await apiClient.setRuntimeMode(requestPayload);
    await refreshRuntimeAuthState({ silent: true });
    await loadPolicyPrompts({ silent: true });
    applyRuntimeControls();
  }

  async function loginRuntimeSession() {
    const username = ($("#runtime-login-username")?.value || "").trim();
    const password = ($("#runtime-login-password")?.value || "").trim();
    if (!username || !password) {
      setStatus("Username and password are required for runtime login.");
      return;
    }

    setStatus(`Logging in to ${runtimeModeLabel()}...`, true);
    const payload = await apiClient.loginRuntime({ username, password });

    if ($("#runtime-login-password")) {
      $("#runtime-login-password").value = "";
    }
    await refreshRuntimeAuthState({ silent: true });
    await loadPolicyPrompts({ silent: true });
    applyRuntimeControls();

    if (payload.success) {
      setStatus(`Login successful as ${payload.role}.`);
    } else {
      setStatus(payload.detail || "Login succeeded, but role is not authorized.");
    }
  }

  async function logoutRuntimeSession() {
    await apiClient.logoutRuntime();
    if ($("#runtime-login-password")) {
      $("#runtime-login-password").value = "";
    }
    await refreshRuntimeAuthState({ silent: true });
    await loadPolicyPrompts({ silent: true });
    applyRuntimeControls();
    setStatus(`Logged out from ${runtimeModeLabel()}.`);
  }

  function getSelectedGpuWorker() {
    if (!state.config) return null;
    const workers = state.config.gpu_workers || [];
    if (workers.length === 0) return null;

    const selectedId = ($("#sel-gpu-worker")?.value || state.selectedGpuWorkerId || "").trim();
    const selected = workers.find(worker => worker.id === selectedId && worker.enabled !== false);
    if (selected) return selected;

    const defaultId = (state.config.default_gpu_worker_id || "").trim();
    const defaultWorker = workers.find(worker => worker.id === defaultId && worker.enabled !== false);
    if (defaultWorker) return defaultWorker;

    return workers.find(worker => worker.enabled !== false) || null;
  }

  function selectedGpuWorkerLabel() {
    return getSelectedGpuWorker()?.label || "GPU worker";
  }

  async function loadConfig() {
    try {
      const payload = await apiClient.getConfig();
      state.config = payload;
      if (payload.runtime_mode) {
        state.runtimeMode = payload.runtime_mode;
      }
      if (payload.runtime_auth) {
        state.runtimeAuth = payload.runtime_auth;
      }
      populateControls();
      applyRuntimeControls();
    } catch (error) {
      setStatus("Config load failed", false);
      toast(`Failed to load config: ${error.message}`, "err");
    }
  }

  function populateControls() {
    const cfg = state.config;
    state.policyPromptOptions = cfg.policy_prompt_options || [];
    state.policyPromptGroups = cfg.policy_prompt_groups || [];
    state.policyPromptSlotKinds = cfg.policy_prompt_slot_kinds || [];

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

    const preferredAvailableModel = cfg.models.find(
      m => m.id === DEFAULT_MODEL_ID && m.is_available !== false,
    );
    const firstAvailableModel = preferredAvailableModel || cfg.models.find(m => m.is_available !== false);
    if (firstAvailableModel) {
      selModel.value = firstAvailableModel.id;
    }

    const selGpuWorker = $("#sel-gpu-worker");
    if (selGpuWorker) {
      selGpuWorker.innerHTML = "";
      (cfg.gpu_workers || []).forEach(worker => {
        if (worker.enabled === false) return;
        selGpuWorker.appendChild(el("option", { value: worker.id }, worker.label));
      });

      const preferredWorker = (cfg.default_gpu_worker_id || "").trim();
      const hasPreferred = preferredWorker
        && selGpuWorker.querySelector(`option[value="${preferredWorker}"]`);
      if (hasPreferred) {
        selGpuWorker.value = preferredWorker;
        state.selectedGpuWorkerId = preferredWorker;
      } else if (selGpuWorker.options.length > 0) {
        selGpuWorker.selectedIndex = 0;
        state.selectedGpuWorkerId = selGpuWorker.value;
      } else {
        state.selectedGpuWorkerId = null;
      }
    }

    const selGalleryModel = $("#sel-gallery-model");
    selGalleryModel.innerHTML = "";
    cfg.models.forEach(m => {
      const opt = el("option", { value: m.id }, m.label);
      selGalleryModel.appendChild(opt);
    });

    applyPolicyPromptDropdowns();

    if (cfg.version) {
      $("#app-version").textContent = `V${cfg.version}`;
    }

    onModelChange();
  }

  async function loadGpuSettings({ silent = false } = {}) {
    try {
      const payload = await apiClient.getGpuSettings();
      applyGpuSettingsForm(payload);
      if (!silent) {
        const stateLabel = payload.use_remote_gpu ? "enabled" : "disabled";
        setGpuSettingsStatus(`Remote worker ${stateLabel}.`);
      }
    } catch (error) {
      if (!silent) {
        setGpuSettingsStatus(`GPU settings unavailable: ${error.message}`);
      }
    }
  }

  async function saveGpuSettings() {
    const useRemote = $("#chk-use-remote-gpu")?.checked || false;
    const remoteUrl = ($("#inp-remote-gpu-url")?.value || "").trim();
    const remoteToken = ($("#inp-remote-gpu-token")?.value || "").trim();

    if (useRemote && !remoteUrl) {
      toast("Remote worker URL is required", "warn");
      setGpuSettingsStatus("Remote worker URL required.");
      return;
    }

    setGpuSettingsStatus("Saving worker settings...");
    const payload = await apiClient.updateGpuSettings({
      use_remote_gpu: useRemote,
      remote_base_url: remoteUrl || null,
      bearer_token: remoteToken || null,
      default_to_remote: false,
      timeout_seconds: 240,
    });

    applyGpuSettingsForm(payload);
    await loadConfig();
    setGpuSettingsStatus("Worker settings saved.");
    setStatus("Worker settings saved.");
    if (payload.generated_bearer_token) {
      toast("Generated a new worker bearer token. Copy it to the remote worker.", "info", 5000);
    } else {
      toast("Worker settings saved", "ok");
    }
  }

  async function testGpuSettingsConnection() {
    const useRemote = $("#chk-use-remote-gpu")?.checked || false;
    const remoteUrl = ($("#inp-remote-gpu-url")?.value || "").trim();
    const remoteToken = ($("#inp-remote-gpu-token")?.value || "").trim();
    if (!useRemote) {
      setGpuSettingsStatus("Enable remote worker first.");
      return;
    }
    if (!remoteUrl) {
      setGpuSettingsStatus("Remote worker URL required.");
      return;
    }

    setGpuSettingsStatus("Testing remote worker...");
    try {
      await apiClient.testGpuSettings({
        remote_base_url: remoteUrl,
        bearer_token: remoteToken || null,
        timeout_seconds: 8,
      });
      setGpuSettingsStatus("Remote worker health check succeeded.");
      toast("Remote worker is reachable", "ok");
    } catch (error) {
      setGpuSettingsStatus(`Remote worker check failed: ${error.message}`);
      toast(`Remote worker check failed: ${error.message}`, "err");
    }
  }

  function generateGpuTokenInField() {
    const tokenField = $("#inp-remote-gpu-token");
    if (!tokenField) return;
    tokenField.value = randomGpuToken();
    tokenField.placeholder = "Generated token (copy to remote worker)";
    setGpuSettingsStatus("Generated worker token. Save settings to persist.");
  }

  function onModelChange() {
    const modelId = $("#sel-model").value;
    state.selectedModel = modelId;

    const model = state.config.models.find(m => m.id === modelId);
    if (!model) return;

    $("#model-info-name").textContent = model.label;
    $("#model-info-desc").textContent = model.is_available === false
      ? `${model.description} ${model.unavailable_reason || ""}`.trim()
      : model.description;
    $("#model-info-link").href = model.hf_url;

    const selAspect = $("#sel-aspect");
    selAspect.innerHTML = "";
    model.aspect_ratios.forEach(ar => {
      const opt = el("option", { value: ar.id }, `${ar.label}  (${ar.width}×${ar.height})`);
      selAspect.appendChild(opt);
    });

    const defaultAr = model.aspect_ratios.find(ar => ar.id === model.default_aspect) || model.aspect_ratios[0];
    selAspect.value = defaultAr.id;
    onAspectChange();

    const rngSteps = $("#rng-steps");
    rngSteps.min = model.min_steps;
    rngSteps.max = model.max_steps;
    rngSteps.value = model.default_steps;
    $("#lbl-steps").textContent = model.default_steps;

    const rngGuidance = $("#rng-guidance");
    rngGuidance.min = model.min_guidance;
    rngGuidance.max = model.max_guidance;
    rngGuidance.step = model.guidance_step;
    rngGuidance.value = model.default_guidance;
    $("#lbl-guidance").textContent = model.default_guidance.toFixed(1);
    const guidanceWrap = $("#guidance-wrap");
    const hasAdjustableGuidance = !(model.min_guidance === 0 && model.max_guidance === 0);
    guidanceWrap.style.display = hasAdjustableGuidance ? "" : "none";

    const negWrap = $("#negative-prompt-wrap");
    negWrap.style.display = model.supports_negative_prompt ? "" : "none";

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
    const modelId = state.selectedModel;
    const model = state.config.models.find(m => m.id === modelId);
    if (!model) return;

    const aspectId = $("#sel-aspect").value;
    const ar = model.aspect_ratios.find(a => a.id === aspectId);
    if (ar) {
      $("#lbl-resolution").textContent = `${ar.width} × ${ar.height} px`;
    }
  }

  return {
    syncGpuSettingsControls,
    applyPolicyPromptDropdowns,
    applyRuntimeControls,
    runtimeModeLabel,
    isRuntimeSessionAuthorized,
    loadRuntimeMode,
    refreshRuntimeAuthState,
    loadPolicyPrompts,
    setRuntimeMode,
    loginRuntimeSession,
    logoutRuntimeSession,
    getSelectedGpuWorker,
    selectedGpuWorkerLabel,
    loadConfig,
    loadGpuSettings,
    saveGpuSettings,
    testGpuSettingsConnection,
    generateGpuTokenInField,
    onModelChange,
    onAspectChange,
  };
}
