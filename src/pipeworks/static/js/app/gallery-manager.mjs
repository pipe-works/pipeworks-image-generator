export function createGalleryManager({
  state,
  $,
  $$,
  el,
  apiClient,
  toast,
  formatImageCountLabel,
  getImageCardBadgeLabel,
  getOutputLightboxController,
}) {
  function patchImageAcrossCollections(imageId, patch) {
    function patchCollection(images) {
      return images.map(image => (image.id === imageId ? { ...image, ...patch } : image));
    }

    state.outputImages = patchCollection(state.outputImages);
    state.galleryImages = patchCollection(state.galleryImages);
    state.favouriteImages = patchCollection(state.favouriteImages);
  }

  function removeImagesAcrossCollections(imageIds) {
    const removedImageIds = new Set(imageIds);

    function filterCollection(images) {
      return images.filter(image => !removedImageIds.has(image.id));
    }

    state.outputImages = filterCollection(state.outputImages);
    state.galleryImages = filterCollection(state.galleryImages);
    state.favouriteImages = filterCollection(state.favouriteImages);

    imageIds.forEach(imageId => {
      state.selectedIds.delete(imageId);
      state.outputSelectedIds.delete(imageId);
    });

    updateSelectionUI();
    updateOutputSelectionUI();
  }

  async function toggleFavourite(imageId, isFav, card, btn) {
    try {
      await apiClient.toggleFavourite(imageId, isFav);

      card.classList.toggle("is-favourite", isFav);
      btn.classList.toggle("is-active", isFav);
      btn.textContent = isFav ? "★" : "☆";
      btn.title = isFav ? "Remove from favourites" : "Add to favourites";

      patchImageAcrossCollections(imageId, { is_favourite: isFav });

      const controller = getOutputLightboxController();
      if (controller) {
        controller.updateImageState(imageId, { is_favourite: isFav });
      }

      toast(isFav ? "Added to favourites" : "Removed from favourites", "ok", 1500);
    } catch (error) {
      toast(`Failed to update favourite: ${error.message}`, "err");
    }
  }

  async function deleteImage(imageId, card) {
    if (!confirm("Delete this image? This cannot be undone.")) return;

    try {
      await apiClient.deleteImage(imageId);

      card.style.opacity = "0";
      card.style.transition = "opacity 0.3s";
      setTimeout(() => card.remove(), 300);

      removeImagesAcrossCollections([imageId]);
      updateOutputCount();

      const controller = getOutputLightboxController();
      if (controller) {
        controller.handleRemovedImages([imageId]);
      }

      await refreshGalleryCollectionsAfterDelete();

      toast("Image deleted", "ok", 1500);
    } catch (error) {
      toast(`Failed to delete: ${error.message}`, "err");
    }
  }

  function openLightbox(img, context = "gallery") {
    state.lightboxContext = context;
    const controller = getOutputLightboxController();
    if (!controller) return;
    controller.open({ image: img, context });
  }

  function closeLightbox() {
    state.lightboxContext = null;
    const controller = getOutputLightboxController();
    if (!controller) return;
    controller.close();
  }

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

    const meta = el(
      "div",
      { className: "img-card__meta" },
      `${img.model_label} · ${img.width}×${img.height}`,
    );

    const favBtn = el(
      "button",
      {
        className: `img-card__fav-btn${img.is_favourite ? " is-active" : ""}`,
        title: img.is_favourite ? "Remove from favourites" : "Add to favourites",
        onclick: event => {
          event.stopPropagation();
          toggleFavourite(img.id, !img.is_favourite, card, favBtn);
        },
      },
      img.is_favourite ? "★" : "☆",
    );

    const delBtn = el(
      "button",
      {
        className: "img-card__del-btn",
        title: "Delete image",
        onclick: event => {
          event.stopPropagation();
          deleteImage(img.id, card);
        },
      },
      "✕",
    );

    overlay.appendChild(meta);
    overlay.appendChild(favBtn);
    overlay.appendChild(delBtn);

    const check = el("div", { className: "img-card__check" });

    card.appendChild(image);
    card.appendChild(star);
    card.appendChild(batchBadge);
    card.appendChild(check);
    card.appendChild(overlay);

    card.addEventListener("click", () => {
      if (context === "output" && state.outputSelectMode) {
        toggleOutputCardSelection(img.id, card);
      } else if (context !== "output" && state.selectMode) {
        toggleCardSelection(img.id, card);
      } else {
        openLightbox(img, context);
      }
    });

    return card;
  }

  function toggleSelectMode() {
    state.selectMode = !state.selectMode;
    state.selectedIds.clear();

    const btn = $("#btn-gallery-select");
    const controls = $("#gallery-select-controls");
    const grid = $("#gallery-grid");

    btn.textContent = state.selectMode ? "✕ Cancel" : "☐ Select";
    controls.classList.toggle("is-active", state.selectMode);
    grid.classList.toggle("gallery-grid--selecting", state.selectMode);

    $$(".img-card.is-selected", grid).forEach(card => card.classList.remove("is-selected"));

    updateSelectionUI();
  }

  function toggleCardSelection(imgId, card) {
    if (state.selectedIds.has(imgId)) {
      state.selectedIds.delete(imgId);
      card.classList.remove("is-selected");
    } else {
      state.selectedIds.add(imgId);
      card.classList.add("is-selected");
    }
    updateSelectionUI();
  }

  function selectAllVisible() {
    const grid = $("#gallery-grid");
    $$(".img-card", grid).forEach(card => {
      const id = card.getAttribute("data-id");
      if (id && !state.selectedIds.has(id)) {
        state.selectedIds.add(id);
        card.classList.add("is-selected");
      }
    });
    updateSelectionUI();
  }

  function updateSelectionUI() {
    const count = state.selectedIds.size;
    $("#lbl-select-count").textContent = `${count} selected`;
    $("#btn-delete-selected").disabled = count === 0;
    $("#btn-save-selected").disabled = count === 0;
  }

  async function bulkDelete() {
    const count = state.selectedIds.size;
    if (count === 0) return;

    if (!confirm(`Delete ${count} image${count !== 1 ? "s" : ""}? This cannot be undone.`)) return;

    try {
      const data = await apiClient.bulkDelete([...state.selectedIds]);

      removeImagesAcrossCollections(data.deleted);

      const controller = getOutputLightboxController();
      if (controller) {
        controller.handleRemovedImages(data.deleted);
      }

      await refreshGalleryCollectionsAfterDelete();

      toast(`Deleted ${data.deleted.length} image${data.deleted.length !== 1 ? "s" : ""}`, "ok");

      toggleSelectMode();
      loadGallery(state.galleryPage);
    } catch (error) {
      toast(`Bulk delete failed: ${error.message}`, "err");
    }
  }

  async function downloadSelectedZip() {
    const count = state.selectedIds.size;
    if (count === 0) return;

    const btn = $("#btn-save-selected");
    const originalText = btn.textContent;
    btn.textContent = "Downloading…";
    btn.disabled = true;

    try {
      const blob = await apiClient.bulkZipBlob([...state.selectedIds]);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `pipeworks_selected_${count}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);

      toast(`Downloaded ${count} image${count !== 1 ? "s" : ""}`, "ok", 1500);
    } catch (error) {
      toast(`Download failed: ${error.message}`, "err");
    } finally {
      btn.textContent = originalText;
      btn.disabled = false;
    }
  }

  function toggleOutputSelectMode() {
    state.outputSelectMode = !state.outputSelectMode;
    state.outputSelectedIds.clear();

    const btn = $("#btn-output-select");
    const controls = $("#output-select-controls");
    const canvas = $("#gen-canvas");

    btn.textContent = state.outputSelectMode ? "✕ Cancel" : "☐ Select";
    controls.classList.toggle("is-active", state.outputSelectMode);
    canvas.classList.toggle("gen-output__canvas--selecting", state.outputSelectMode);

    $$(".img-card.is-selected", canvas).forEach(card => card.classList.remove("is-selected"));

    updateOutputSelectionUI();
  }

  function toggleOutputCardSelection(imgId, card) {
    if (state.outputSelectedIds.has(imgId)) {
      state.outputSelectedIds.delete(imgId);
      card.classList.remove("is-selected");
    } else {
      state.outputSelectedIds.add(imgId);
      card.classList.add("is-selected");
    }
    updateOutputSelectionUI();
  }

  function selectAllOutputVisible() {
    const canvas = $("#gen-canvas");
    $$(".img-card", canvas).forEach(card => {
      const id = card.getAttribute("data-id");
      if (id && !state.outputSelectedIds.has(id)) {
        state.outputSelectedIds.add(id);
        card.classList.add("is-selected");
      }
    });
    updateOutputSelectionUI();
  }

  function getVisibleOutputImageIds() {
    const canvas = $("#gen-canvas");
    return [
      ...new Set(
        $$(".img-card", canvas)
          .map(card => card.getAttribute("data-id"))
          .filter(Boolean),
      ),
    ];
  }

  function updateOutputSelectionUI() {
    const selectedCount = state.outputSelectedIds.size;
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
    btn.textContent = "Downloading…";
    btn.disabled = true;

    try {
      const blob = await apiClient.bulkZipBlob(imageIds);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `pipeworks_output_${count}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);

      toast(`Downloaded ${count} image${count !== 1 ? "s" : ""}`, "ok", 1500);
    } catch (error) {
      toast(`Download failed: ${error.message}`, "err");
    } finally {
      btn.textContent = originalText;
      updateOutputSelectionUI();
    }
  }

  function updateOutputCount() {
    const count = $("#gen-canvas").querySelectorAll(".img-card").length;
    $("#lbl-output-count").textContent = formatImageCountLabel(count);
  }

  function findLightboxContextCard(imageId) {
    if (state.lightboxContext === "output") {
      return $("#gen-canvas")?.querySelector(`.img-card[data-id="${imageId}"]`) || null;
    }

    if (state.lightboxContext === "gallery") {
      return $("#gallery-grid")?.querySelector(`.img-card[data-id="${imageId}"]`) || null;
    }

    if (state.lightboxContext === "favourites") {
      return $("#fav-grid")?.querySelector(`.img-card[data-id="${imageId}"]`) || null;
    }

    return document.querySelector(`.img-card[data-id="${imageId}"]`);
  }

  async function refreshGalleryCollectionsAfterDelete() {
    await Promise.all([loadGallery(state.galleryPage), loadFavourites(state.favPage)]);
  }

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

  function getActiveTabId() {
    return $(".tab-nav__item.is-active")?.dataset.tab || null;
  }

  async function loadGallery(page = 1) {
    state.galleryPage = page;

    if (state.selectMode) toggleSelectMode();

    const modelFilter = $("#sel-gallery-model").value;

    try {
      const data = await apiClient.getGallery({
        page,
        perPage: state.galleryPerPage,
        favouritesOnly: false,
        modelId: modelFilter || null,
      });

      state.galleryPage = data.page;
      state.galleryPages = data.pages;
      state.galleryImages = data.images;

      const grid = $("#gallery-grid");
      grid.innerHTML = "";

      if (data.images.length === 0) {
        const empty = el(
          "div",
          { className: "gallery-empty" },
          el("div", { style: { fontSize: "2rem", opacity: "0.3" } }, "◈"),
          el("div", {}, "No images found"),
          el(
            "div",
            { className: "u-muted", style: { fontSize: "var(--text-xs)" } },
            modelFilter ? "Try a different filter" : "Generate some images first",
          ),
        );
        grid.appendChild(empty);
      } else {
        const pageOffset = (data.page - 1) * state.galleryPerPage;
        data.images.forEach((img, index) => {
          const collectionIndex = pageOffset + index + 1;
          grid.appendChild(createImageCard(img, "gallery", collectionIndex));
        });
      }

      $("#lbl-gallery-count").textContent = formatImageCountLabel(data.total);
      $("#lbl-gallery-page").textContent = `Page ${data.page} of ${data.pages}`;
      $("#btn-gallery-prev").disabled = data.page <= 1;
      $("#btn-gallery-next").disabled = data.page >= data.pages;
    } catch (error) {
      toast(`Gallery load failed: ${error.message}`, "err");
    }
  }

  async function loadFavourites(page = 1) {
    state.favPage = page;

    try {
      const data = await apiClient.getGallery({
        page,
        perPage: state.galleryPerPage,
        favouritesOnly: true,
      });

      state.favPage = data.page;
      state.favouriteImages = data.images;

      const grid = $("#fav-grid");
      grid.innerHTML = "";

      if (data.images.length === 0) {
        const empty = el(
          "div",
          { className: "gallery-empty" },
          el("div", { style: { fontSize: "2rem", opacity: "0.3" } }, "★"),
          el("div", {}, "No favourites yet"),
          el(
            "div",
            { className: "u-muted", style: { fontSize: "var(--text-xs)" } },
            "Star images to add them here",
          ),
        );
        grid.appendChild(empty);
      } else {
        const favouritesOffset = (data.page - 1) * state.galleryPerPage;
        data.images.forEach((img, index) => {
          const collectionIndex = favouritesOffset + index + 1;
          grid.appendChild(createImageCard(img, "favourites", collectionIndex));
        });
      }

      $("#lbl-fav-count").textContent = formatImageCountLabel(data.total);
      $("#lbl-fav-page").textContent = `Page ${data.page} of ${data.pages}`;
      $("#btn-fav-prev").disabled = data.page <= 1;
      $("#btn-fav-next").disabled = data.page >= data.pages;
    } catch (error) {
      toast(`Favourites load failed: ${error.message}`, "err");
    }
  }

  async function openStatsModal() {
    $("#modal-stats").classList.remove("hidden");

    try {
      const data = await apiClient.getStats();

      const content = $("#modal-stats-content");
      content.innerHTML = "";

      const table = el("table");
      const tbody = el("tbody");

      const rows = [
        ["Total Images", data.total_images],
        ["Favourites", data.total_favourites],
      ];

      if (state.config) {
        state.config.models.forEach(m => {
          rows.push([m.label, data.model_counts[m.id] || 0]);
        });
      }

      rows.forEach(([label, value]) => {
        const tr = el("tr");
        tr.appendChild(
          el(
            "td",
            { style: { color: "var(--col-text-muted)", paddingRight: "var(--sp-4)" } },
            label,
          ),
        );
        tr.appendChild(
          el("td", { style: { color: "var(--col-accent)", fontWeight: "700" } }, String(value)),
        );
        tbody.appendChild(tr);
      });

      table.appendChild(tbody);
      content.appendChild(table);
    } catch (error) {
      $("#modal-stats-content").textContent = `Error: ${error.message}`;
    }
  }

  return {
    createImageCard,
    toggleFavourite,
    deleteImage,
    toggleSelectMode,
    toggleCardSelection,
    selectAllVisible,
    updateSelectionUI,
    bulkDelete,
    downloadSelectedZip,
    toggleOutputSelectMode,
    toggleOutputCardSelection,
    selectAllOutputVisible,
    getVisibleOutputImageIds,
    updateOutputSelectionUI,
    downloadAllOutputZip,
    updateOutputCount,
    openLightbox,
    closeLightbox,
    findLightboxContextCard,
    refreshGalleryCollectionsAfterDelete,
    isTypingTargetActive,
    getActiveTabId,
    loadGallery,
    loadFavourites,
    openStatsModal,
  };
}
