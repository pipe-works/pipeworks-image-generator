export function createGenerationFlow({
  state,
  $,
  apiClient,
  toast,
  setStatus,
  buildGeneratePayload,
  selectedGpuWorkerLabel,
  createImageCard,
  updateOutputCount,
}) {
  function sleep(ms) {
    return new Promise(resolve => {
      window.setTimeout(resolve, ms);
    });
  }

  function startGenerationStatusPolling({ generationId }) {
    let stopped = false;

    const poll = async () => {
      while (!stopped) {
        try {
          const status = await apiClient.getGenerationStatus(generationId);
          if (status?.message) {
            setStatus(status.message, status.done !== true);
          }
          if (status?.done === true) {
            stopped = true;
            break;
          }
        } catch (error) {
          if (stopped) break;
        }
        await sleep(750);
      }
    };

    const pollingPromise = poll();
    return {
      stop: async () => {
        stopped = true;
        await pollingPromise;
      },
    };
  }

  async function generate() {
    if (state.isGenerating) return;

    const payload = buildGeneratePayload();
    if (!payload) {
      toast("Please finish configuring the generator", "warn");
      return;
    }

    const model = state.config.models.find(m => m.id === state.selectedModel);
    if (model && model.is_available === false) {
      toast(model.unavailable_reason || "Selected model is unavailable in this runtime", "err");
      setStatus("Selected model unavailable", false);
      return;
    }

    state.isGenerating = true;
    state.stopRequested = false;
    const workerLabel = selectedGpuWorkerLabel();
    state.currentGenerationId = globalThis.crypto?.randomUUID?.() || `gen-${Date.now()}`;
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

    setStatus(`Preparing ${state.batchSize} image(s) on ${workerLabel}…`, true);
    const statusPolling = startGenerationStatusPolling({
      generationId: state.currentGenerationId,
    });

    try {
      payload.generation_id = state.currentGenerationId;
      const data = await apiClient.generate(payload);
      await statusPolling.stop();

      if (payload.seed === null) {
        $("#inp-seed").placeholder = `Last: ${data.batch_seed}`;
      }
      $("#status-seed").textContent = `seed ${data.batch_seed}`;

      const placeholder = $("#gen-placeholder");
      if (placeholder) placeholder.remove();

      data.images.forEach(img => {
        state.outputImages.unshift(img);
        const card = createImageCard(img, "output");
        const canvas = $("#gen-canvas");
        canvas.insertBefore(card, canvas.firstChild);
      });

      updateOutputCount();
      if (data.cancelled) {
        const completed = data.completed_count || data.images.length;
        if (completed > 0) {
          toast(`Stopped after ${completed} image(s)`, "info");
          setStatus(`Stopped after ${completed} image(s) on ${workerLabel}`, false);
        } else {
          toast("Stopped before any images completed", "info");
          setStatus(`Stopped before any images completed on ${workerLabel}`, false);
        }
      } else {
        toast(`Generated ${data.images.length} image(s)`, "ok");
        setStatus(`Done — ${data.images.length} image(s) generated on ${workerLabel}`);
      }
    } catch (error) {
      await statusPolling.stop();
      toast(`Generation failed on ${workerLabel}: ${error.message}`, "err");
      setStatus(`Generation failed on ${workerLabel}`, false);
    } finally {
      state.isGenerating = false;
      state.currentGenerationId = null;
      state.stopRequested = false;
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
    if (!state.isGenerating || !state.currentGenerationId || state.stopRequested) return;

    state.stopRequested = true;
    const stopBtn = $("#btn-stop-generation");
    stopBtn.disabled = true;
    stopBtn.textContent = "■ Stopping…";

    setStatus("Stopping after current image…", true);

    try {
      await apiClient.cancelGeneration(state.currentGenerationId);
      toast("Stop requested. Waiting for the current image to finish.", "info");
    } catch (error) {
      state.stopRequested = false;
      stopBtn.disabled = false;
      stopBtn.textContent = "■ Stop After Current Image";
      toast(`Stop request failed: ${error.message}`, "err");
      setStatus("Stop request failed", false);
    }
  }

  return {
    generate,
    stopGeneration,
  };
}
