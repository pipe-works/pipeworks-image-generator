/**
 * Frontend unit tests for the Output lightbox navigation helpers.
 *
 * These tests intentionally target the pure helper exports so they can run in
 * Node without a browser DOM or additional dependencies.  The browser-facing
 * DOM wiring remains covered separately by the server-rendered HTML test.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  getOutputTransportState,
  getWrappedImageIndex,
  resolveOutputNavigationDirection,
} from "../../src/pipeworks/static/js/output-lightbox.mjs";

/**
 * Verify that each requested vim-style key maps to the expected movement
 * direction and unrelated keys remain inert.
 */
test("resolveOutputNavigationDirection maps hjkl keys correctly", () => {
  assert.equal(resolveOutputNavigationDirection("h"), -1);
  assert.equal(resolveOutputNavigationDirection("k"), -1);
  assert.equal(resolveOutputNavigationDirection("j"), 1);
  assert.equal(resolveOutputNavigationDirection("l"), 1);
  assert.equal(resolveOutputNavigationDirection("x"), 0);
});

/**
 * Wrapped navigation is important for both autoplay and keyboard movement so
 * the lightbox can cycle through Output images continuously.
 */
test("getWrappedImageIndex wraps at both sequence boundaries", () => {
  assert.equal(getWrappedImageIndex(0, -1, 3), 2);
  assert.equal(getWrappedImageIndex(2, 1, 3), 0);
  assert.equal(getWrappedImageIndex(1, 1, 3), 2);
  assert.equal(getWrappedImageIndex(0, 1, 0), -1);
});

/**
 * Transport controls should only activate for Output images when the current
 * image resolves successfully and there is more than one Output image.
 */
test("getOutputTransportState enables navigation only for multi-image Output context", () => {
  const outputImages = [
    { id: "img-1" },
    { id: "img-2" },
    { id: "img-3" },
  ];

  const activeState = getOutputTransportState({
    context: "output",
    outputImages,
    currentImageId: "img-2",
    isPlaying: true,
  });

  assert.equal(activeState.isOutputContext, true);
  assert.equal(activeState.currentIndex, 1);
  assert.equal(activeState.canNavigate, true);
  assert.equal(activeState.canPlay, true);
  assert.equal(activeState.isPlaying, true);

  const singleImageState = getOutputTransportState({
    context: "output",
    outputImages: [{ id: "img-1" }],
    currentImageId: "img-1",
    isPlaying: true,
  });

  assert.equal(singleImageState.canNavigate, false);
  assert.equal(singleImageState.canPlay, false);
  assert.equal(singleImageState.isPlaying, false);

  const galleryState = getOutputTransportState({
    context: "gallery",
    outputImages,
    currentImageId: "img-2",
    isPlaying: true,
  });

  assert.equal(galleryState.isOutputContext, false);
  assert.equal(galleryState.currentIndex, -1);
  assert.equal(galleryState.canNavigate, false);
  assert.equal(galleryState.canPlay, false);
  assert.equal(galleryState.isPlaying, false);
});
