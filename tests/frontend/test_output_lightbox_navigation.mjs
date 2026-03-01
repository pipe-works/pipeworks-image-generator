/**
 * Frontend unit tests for collection-aware gallery helpers and lightbox
 * transport behavior.
 *
 * These tests intentionally target pure helper exports so they can run in the
 * Node test runner without requiring a browser DOM.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
  getCollectionDisplayLabel,
  getCollectionNavigationHint,
  getCollectionStatusText,
  getImageCardBadgeLabel,
  getUnavailableCollectionStatusText,
} from "../../src/pipeworks/static/js/gallery-context.mjs";
import {
  getLightboxTransportState,
  getWrappedImageIndex,
  resolveOutputNavigationDirection,
} from "../../src/pipeworks/static/js/output-lightbox.mjs";

/**
 * The vim-style transport bindings should remain stable across all collections.
 */
test("resolveOutputNavigationDirection maps hjkl keys correctly", () => {
  assert.equal(resolveOutputNavigationDirection("h"), -1);
  assert.equal(resolveOutputNavigationDirection("k"), -1);
  assert.equal(resolveOutputNavigationDirection("j"), 1);
  assert.equal(resolveOutputNavigationDirection("l"), 1);
  assert.equal(resolveOutputNavigationDirection("x"), 0);
});

/**
 * Wrapped navigation is shared by keyboard navigation and slideshow playback.
 */
test("getWrappedImageIndex wraps at both sequence boundaries", () => {
  assert.equal(getWrappedImageIndex(0, -1, 3), 2);
  assert.equal(getWrappedImageIndex(2, 1, 3), 0);
  assert.equal(getWrappedImageIndex(1, 1, 3), 2);
  assert.equal(getWrappedImageIndex(0, 1, 0), -1);
});

/**
 * Collection transport should activate whenever the current image resolves in
 * the active collection and the collection contains more than one image.
 */
test("getLightboxTransportState enables navigation for any multi-image collection", () => {
  const collectionImages = [
    { id: "img-1" },
    { id: "img-2" },
    { id: "img-3" },
  ];

  const galleryState = getLightboxTransportState({
    context: "gallery",
    collectionImages,
    currentImageId: "img-2",
    isPlaying: true,
  });

  assert.equal(galleryState.currentIndex, 1);
  assert.equal(galleryState.canNavigate, true);
  assert.equal(galleryState.canPlay, true);
  assert.equal(galleryState.isPlaying, true);

  const singleImageState = getLightboxTransportState({
    context: "favourites",
    collectionImages: [{ id: "img-1" }],
    currentImageId: "img-1",
    isPlaying: true,
  });

  assert.equal(singleImageState.canNavigate, false);
  assert.equal(singleImageState.canPlay, false);
  assert.equal(singleImageState.isPlaying, false);

  const missingImageState = getLightboxTransportState({
    context: "output",
    collectionImages,
    currentImageId: "missing",
    isPlaying: true,
  });

  assert.equal(missingImageState.currentIndex, -1);
  assert.equal(missingImageState.canNavigate, false);
});

/**
 * Gallery and favourites badges should reflect the current collection position
 * instead of leaking the original generation batch index.
 */
test("getImageCardBadgeLabel uses collection positions outside Output", () => {
  const image = { batch_index: 4 };

  assert.equal(
    getImageCardBadgeLabel({
      context: "output",
      image,
      collectionIndex: 9,
    }),
    "#5",
  );

  assert.equal(
    getImageCardBadgeLabel({
      context: "gallery",
      image,
      collectionIndex: 2,
    }),
    "#2",
  );

  assert.equal(
    getImageCardBadgeLabel({
      context: "favourites",
      image,
      collectionIndex: 7,
    }),
    "#7",
  );
});

/**
 * The collection text helpers should produce stable lightbox labels and hints.
 */
test("collection helpers return consistent display text", () => {
  assert.equal(getCollectionDisplayLabel("gallery"), "Gallery");
  assert.equal(
    getCollectionStatusText({
      context: "favourites",
      currentIndex: 1,
      totalImages: 4,
    }),
    "Favourites 2 / 4",
  );
  assert.equal(getCollectionNavigationHint(), "H/K previous · J/L next");
  assert.equal(getUnavailableCollectionStatusText("output"), "Output navigation unavailable");
});
