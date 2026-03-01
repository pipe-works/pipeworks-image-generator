/**
 * Gallery / Lightbox Collection Helpers
 * ---------------------------------------------------------------------------
 * This module contains pure collection-presentation helpers that are shared by
 * the gallery card renderer and the lightbox controller.
 *
 * Keeping these rules in a dedicated module avoids duplicating collection
 * naming, badge formatting, and transport status text across multiple frontend
 * files.  The functions are deliberately DOM-free so they can be covered by
 * lightweight Node tests.
 */

/**
 * Convert an internal collection context identifier into the user-facing label
 * that appears in badges and lightbox transport status text.
 *
 * @param {string | null | undefined} context - Internal collection identifier.
 * @returns {string} Stable display label for the collection.
 */
export function getCollectionDisplayLabel(context) {
  if (context === "output") {
    return "Output";
  }

  if (context === "gallery") {
    return "Gallery";
  }

  if (context === "favourites") {
    return "Favourites";
  }

  return "Images";
}

/**
 * Build the badge text for an image card in the requested collection context.
 *
 * Output cards intentionally retain their generation-local numbering because
 * that badge identifies the image's position inside the generation batch.
 * Gallery and Favourites cards instead display their collection position so
 * the badge reflects what the user is currently browsing, not the original
 * batch that produced the image.
 *
 * @param {object} params - Badge inputs.
 * @param {string} params.context - Card collection context.
 * @param {object} params.image - Image metadata record.
 * @param {number | null | undefined} params.collectionIndex - One-based index
 *     within the currently rendered collection, when applicable.
 * @returns {string | null} Badge text such as `#3`, or `null` if unavailable.
 */
export function getImageCardBadgeLabel({ context, image, collectionIndex }) {
  if (context === "output") {
    return `#${(image.batch_index || 0) + 1}`;
  }

  if (Number.isInteger(collectionIndex) && collectionIndex > 0) {
    return `#${collectionIndex}`;
  }

  return null;
}

/**
 * Build the lightbox transport status text for the active collection.
 *
 * @param {object} params - Status inputs.
 * @param {string} params.context - Active collection context.
 * @param {number} params.currentIndex - Zero-based active index.
 * @param {number} params.totalImages - Total images in the active collection.
 * @returns {string} Status text shown above the image metadata.
 */
export function getCollectionStatusText({ context, currentIndex, totalImages }) {
  const label = getCollectionDisplayLabel(context);
  return `${label} ${currentIndex + 1} / ${totalImages}`;
}

/**
 * Provide the shared keyboard-navigation hint shown in the lightbox.
 *
 * The same `h/j/k/l` navigation works for Output, Gallery, and Favourites, so
 * the hint text stays stable across contexts.
 *
 * @returns {string} Keyboard-help text for the lightbox transport footer.
 */
export function getCollectionNavigationHint() {
  return "H/K previous · J/L next";
}

/**
 * Provide the status text shown when the current lightbox context cannot be
 * navigated because it has no resolved collection or only a single image.
 *
 * @param {string | null | undefined} context - Active collection context.
 * @returns {string} Disabled-state status text for the lightbox.
 */
export function getUnavailableCollectionStatusText(context) {
  return `${getCollectionDisplayLabel(context)} navigation unavailable`;
}
