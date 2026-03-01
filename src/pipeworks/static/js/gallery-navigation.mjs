/**
 * Gallery Navigation Helpers
 * ---------------------------------------------------------------------------
 * This module contains pure helpers for gallery pagination controls and count
 * labels.  Keeping these rules out of the main frontend bootstrap file makes
 * the behavior easier to test and avoids growing `app.js` further.
 */

/**
 * Format a user-facing image count label.
 *
 * @param {number} count - Number of images represented by the label.
 * @returns {string} Label such as `0 images` or `1 image`.
 */
export function formatImageCountLabel(count) {
  return `${count} image${count !== 1 ? "s" : ""}`;
}

/**
 * Resolve gallery page hotkeys into pagination directions.
 *
 * Only `h` and `l` are handled here because the request is scoped to the main
 * Gallery pagination controls, not the Favourites list or the lightbox.
 *
 * @param {string} key - Raw `KeyboardEvent.key` value.
 * @returns {-1 | 0 | 1} `-1` for previous page, `1` for next page, `0` otherwise.
 */
export function resolveGalleryPaginationDirection(key) {
  const normalisedKey = String(key || "").toLowerCase();

  if (normalisedKey === "h") {
    return -1;
  }

  if (normalisedKey === "l") {
    return 1;
  }

  return 0;
}
