export function createApiClient({ fetchJson }) {
  return {
    getConfig() {
      return fetchJson("/api/config");
    },
    getRuntimeMode() {
      return fetchJson("/api/runtime-mode");
    },
    setRuntimeMode(payload) {
      return fetchJson("/api/runtime-mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    getRuntimeAuth() {
      return fetchJson("/api/runtime-auth");
    },
    getPolicyPrompts() {
      return fetchJson("/api/policy-prompts");
    },
    loginRuntime(payload) {
      return fetchJson("/api/runtime-login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    logoutRuntime() {
      return fetchJson("/api/runtime-logout", { method: "POST" });
    },
    getGpuSettings() {
      return fetchJson("/api/gpu-settings");
    },
    updateGpuSettings(payload) {
      return fetchJson("/api/gpu-settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    testGpuSettings(payload) {
      return fetchJson("/api/gpu-settings/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    compilePrompt(payload) {
      return fetchJson("/api/prompt/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    generate(payload) {
      return fetchJson("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    getGenerationStatus(generationId) {
      return fetchJson(`/api/generate/status/${encodeURIComponent(generationId)}`);
    },
    cancelGeneration(generationId) {
      return fetchJson("/api/generate/cancel", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generation_id: generationId }),
      });
    },
    getGallery({ page, perPage, favouritesOnly, modelId }) {
      const query = new URLSearchParams();
      query.set("page", String(page));
      query.set("per_page", String(perPage));
      if (favouritesOnly) query.set("favourites_only", "true");
      if (modelId) query.set("model_id", modelId);
      return fetchJson(`/api/gallery?${query.toString()}`);
    },
    getStats() {
      return fetchJson("/api/stats");
    },
    toggleFavourite(imageId, isFavourite) {
      return fetchJson("/api/gallery/favourite", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: imageId, is_favourite: isFavourite }),
      });
    },
    deleteImage(imageId) {
      return fetchJson(`/api/gallery/${encodeURIComponent(imageId)}`, {
        method: "DELETE",
      });
    },
    bulkDelete(imageIds) {
      return fetchJson("/api/gallery/bulk-delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_ids: imageIds }),
      });
    },
    async bulkZipBlob(imageIds) {
      const response = await fetch("/api/gallery/bulk-zip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_ids: imageIds }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `HTTP ${response.status}`);
      }
      return response.blob();
    },
    fetchPolicyPrompts() {
      return fetchJson("/api/policy-prompts");
    },
    fetchLoraTilePacks() {
      return fetchJson("/api/lora-dataset/tile-packs");
    },
    listLoraRuns() {
      return fetchJson("/api/lora-dataset/runs");
    },
    getLoraRun(runId) {
      return fetchJson(`/api/lora-dataset/runs/${encodeURIComponent(runId)}`);
    },
    createLoraRun(payload) {
      return fetchJson("/api/lora-dataset/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    },
    patchLoraRun(runId, patch) {
      return fetchJson(`/api/lora-dataset/runs/${encodeURIComponent(runId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
    },
    generateLoraRun(runId) {
      return fetchJson(`/api/lora-dataset/runs/${encodeURIComponent(runId)}/generate`, {
        method: "POST",
      });
    },
    cancelLoraRun(runId) {
      return fetchJson(`/api/lora-dataset/runs/${encodeURIComponent(runId)}/cancel`, {
        method: "POST",
      });
    },
    regenerateLoraSlot(runId, slotKey) {
      return fetchJson(
        `/api/lora-dataset/runs/${encodeURIComponent(runId)}/slots/${encodeURIComponent(
          slotKey
        )}/regenerate`,
        { method: "POST" }
      );
    },
    patchLoraSlot(runId, slotKey, patch) {
      return fetchJson(
        `/api/lora-dataset/runs/${encodeURIComponent(runId)}/slots/${encodeURIComponent(slotKey)}`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patch),
        }
      );
    },
    exportLoraDataset(runId) {
      return fetchJson(`/api/lora-dataset/runs/${encodeURIComponent(runId)}/dataset`, {
        method: "POST",
      });
    },
  };
}
