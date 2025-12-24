"""Unit tests for gallery handler functions."""

from unittest.mock import MagicMock

import pytest

from pipeworks.core.catalog_manager import CatalogManager
from pipeworks.core.favorites_db import FavoritesDB
from pipeworks.core.gallery_browser import GalleryBrowser
from pipeworks.ui.handlers.gallery import (
    apply_gallery_filter,
    refresh_gallery,
    select_gallery_image,
    toggle_favorite,
    toggle_metadata_format,
)
from pipeworks.ui.models import UIState


@pytest.fixture
def mock_gallery_browser(temp_dir):
    """Create a mock GalleryBrowser with test data."""
    outputs_dir = temp_dir / "outputs"
    outputs_dir.mkdir(parents=True)

    # Create some test images
    (outputs_dir / "image1.png").touch()
    (outputs_dir / "image2.png").touch()
    (outputs_dir / "image3.png").touch()

    # Create metadata files
    (outputs_dir / "image1.txt").write_text("Prompt for image 1")
    (outputs_dir / "image2.txt").write_text("Prompt for image 2")

    return GalleryBrowser(outputs_dir)


@pytest.fixture
def mock_favorites_db(temp_dir):
    """Create a mock FavoritesDB."""
    db_path = temp_dir / "test_favorites.db"
    return FavoritesDB(str(db_path))


@pytest.fixture
def initialized_state(mock_gallery_browser, mock_favorites_db, temp_dir):
    """Create an initialized UIState with gallery components."""
    outputs_dir = temp_dir / "outputs"
    catalog_dir = temp_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)

    state = UIState()
    state.gallery_browser = mock_gallery_browser
    state.favorites_db = mock_favorites_db
    state.catalog_manager = CatalogManager(outputs_dir, catalog_dir, mock_favorites_db)
    state.gallery_current_path = ""
    state.gallery_images = []
    state.gallery_selected_index = None
    state.gallery_initialized = True
    return state


# ============================================================================
# refresh_gallery Tests
# ============================================================================


class TestRefreshGallery:
    """Tests for refresh_gallery handler."""

    def test_refresh_gallery_updates_image_list(self, initialized_state, temp_dir):
        """Test that refresh_gallery updates the gallery image list."""
        # Initially empty
        assert initialized_state.gallery_images == []

        # Refresh
        images, state = refresh_gallery("", initialized_state)

        # Should now have 3 images
        assert len(images) == 3
        assert len(state.gallery_images) == 3
        assert all("image" in img for img in images)

    def test_refresh_gallery_clears_selected_index(self, initialized_state):
        """Test that refresh_gallery clears the selected index."""
        # Set a selected index
        initialized_state.gallery_selected_index = 1

        # Refresh
        images, state = refresh_gallery("", initialized_state)

        # Selected index should be None after refresh
        assert state.gallery_selected_index is None

    def test_refresh_gallery_detects_new_images(self, initialized_state, temp_dir):
        """Test that refresh_gallery detects newly added images."""
        outputs_dir = temp_dir / "outputs"

        # Initial refresh
        images, state = refresh_gallery("", initialized_state)
        assert len(images) == 3

        # Add a new image
        (outputs_dir / "image4.png").touch()

        # Refresh again
        images, state = refresh_gallery("", state)
        assert len(images) == 4

    def test_refresh_gallery_detects_removed_images(self, initialized_state, temp_dir):
        """Test that refresh_gallery detects removed images."""
        outputs_dir = temp_dir / "outputs"

        # Initial refresh
        images, state = refresh_gallery("", initialized_state)
        assert len(images) == 3

        # Remove an image
        (outputs_dir / "image1.png").unlink()

        # Refresh again
        images, state = refresh_gallery("", state)
        assert len(images) == 2

    def test_refresh_gallery_with_subdirectory(self, initialized_state, temp_dir):
        """Test refresh_gallery in a subdirectory."""
        outputs_dir = temp_dir / "outputs"
        subdir = outputs_dir / "subfolder"
        subdir.mkdir()
        (subdir / "subimage.png").touch()

        # Refresh in subdirectory
        images, state = refresh_gallery("subfolder", initialized_state)

        assert len(images) == 1
        assert "subimage.png" in images[0]

    def test_refresh_gallery_handles_uninitialized_state(self):
        """Test refresh_gallery handles uninitialized state gracefully."""
        state = UIState()
        # Don't call initialize_ui_state() - start with uninitialized state
        # Set gallery_browser to None manually
        state.gallery_browser = None
        state.initialized = False

        images, state = refresh_gallery("", state)

        # Should return empty list when gallery_browser is None after initialization
        assert images == []
        # After calling handler, state will be initialized but gallery_browser may still be None
        # if initialization failed (which is expected when config is not set up properly)
        # So we just check that images is empty
        assert len(images) == 0

    def test_refresh_gallery_preserves_other_state(self, initialized_state):
        """Test that refresh_gallery preserves other state attributes."""
        # Set some other state attributes
        initialized_state.gallery_filter = "favorites"
        initialized_state.gallery_root = "catalog"

        images, state = refresh_gallery("", initialized_state)

        # Other attributes should be preserved
        assert state.gallery_filter == "favorites"
        assert state.gallery_root == "catalog"


# ============================================================================
# select_gallery_image Tests
# ============================================================================


class TestSelectGalleryImage:
    """Tests for select_gallery_image handler."""

    def test_select_gallery_image_with_txt_metadata(self, initialized_state, temp_dir):
        """Test selecting an image with .txt metadata."""
        outputs_dir = temp_dir / "outputs"

        # Populate gallery images
        initialized_state.gallery_images = [
            str(outputs_dir / "image1.png"),
            str(outputs_dir / "image2.png"),
        ]

        # Create mock SelectData event
        evt = MagicMock()
        evt.index = 0  # Select first image

        # Select image
        image_path, metadata, fav_btn, state = select_gallery_image(
            evt, "Text (.txt)", initialized_state
        )

        assert image_path == str(outputs_dir / "image1.png")
        assert "image1.png" in metadata
        assert "Prompt for image 1" in metadata
        assert state.gallery_selected_index == 0

    def test_select_gallery_image_with_json_metadata(self, initialized_state, temp_dir):
        """Test selecting an image with .json metadata."""
        outputs_dir = temp_dir / "outputs"

        # Create JSON metadata
        import json

        (outputs_dir / "image1.json").write_text(json.dumps({"prompt": "JSON prompt", "seed": 42}))

        initialized_state.gallery_images = [str(outputs_dir / "image1.png")]

        evt = MagicMock()
        evt.index = 0

        image_path, metadata, fav_btn, state = select_gallery_image(
            evt, "JSON (.json)", initialized_state
        )

        assert image_path == str(outputs_dir / "image1.png")
        assert "image1.png" in metadata
        assert "JSON prompt" in metadata
        assert "42" in metadata

    def test_select_gallery_image_without_metadata(self, initialized_state, temp_dir):
        """Test selecting an image without metadata files."""
        outputs_dir = temp_dir / "outputs"

        # image3 has no metadata
        initialized_state.gallery_images = [str(outputs_dir / "image3.png")]

        evt = MagicMock()
        evt.index = 0

        image_path, metadata, fav_btn, state = select_gallery_image(
            evt, "Text (.txt)", initialized_state
        )

        assert image_path == str(outputs_dir / "image3.png")
        assert "No .txt metadata found" in metadata

    def test_select_gallery_image_out_of_bounds(self, initialized_state, temp_dir):
        """Test selecting an invalid image index."""
        outputs_dir = temp_dir / "outputs"
        initialized_state.gallery_images = [str(outputs_dir / "image1.png")]

        evt = MagicMock()
        evt.index = 999  # Out of bounds

        image_path, metadata, fav_btn, state = select_gallery_image(
            evt, "Text (.txt)", initialized_state
        )

        assert image_path == ""
        assert "No image selected" in metadata

    def test_select_gallery_image_updates_favorite_button(self, initialized_state, temp_dir):
        """Test that favorite button label reflects favorite status."""
        outputs_dir = temp_dir / "outputs"
        image_path = str(outputs_dir / "image1.png")

        initialized_state.gallery_images = [image_path]

        # Mark as favorite
        initialized_state.favorites_db.add_favorite(image_path)

        evt = MagicMock()
        evt.index = 0

        _, _, fav_btn, _ = select_gallery_image(evt, "Text (.txt)", initialized_state)

        assert fav_btn == "⭐ Unfavorite"

    def test_select_gallery_image_not_favorited(self, initialized_state, temp_dir):
        """Test favorite button label for non-favorited image."""
        outputs_dir = temp_dir / "outputs"
        initialized_state.gallery_images = [str(outputs_dir / "image1.png")]

        evt = MagicMock()
        evt.index = 0

        _, _, fav_btn, _ = select_gallery_image(evt, "Text (.txt)", initialized_state)

        assert fav_btn == "☆ Favorite"


# ============================================================================
# toggle_metadata_format Tests
# ============================================================================


class TestToggleMetadataFormat:
    """Tests for toggle_metadata_format handler."""

    def test_toggle_to_json_format(self, initialized_state, temp_dir):
        """Test toggling from txt to json format."""
        outputs_dir = temp_dir / "outputs"

        # Create JSON metadata
        import json

        (outputs_dir / "image1.json").write_text(json.dumps({"prompt": "JSON prompt", "seed": 42}))

        initialized_state.gallery_images = [str(outputs_dir / "image1.png")]
        initialized_state.gallery_selected_index = 0

        metadata, state = toggle_metadata_format("JSON (.json)", initialized_state)

        assert "JSON prompt" in metadata
        assert "42" in metadata

    def test_toggle_to_txt_format(self, initialized_state, temp_dir):
        """Test toggling from json to txt format."""
        outputs_dir = temp_dir / "outputs"
        initialized_state.gallery_images = [str(outputs_dir / "image1.png")]
        initialized_state.gallery_selected_index = 0

        metadata, state = toggle_metadata_format("Text (.txt)", initialized_state)

        assert "Prompt for image 1" in metadata

    def test_toggle_metadata_with_no_selection(self, initialized_state):
        """Test toggling metadata format with no image selected."""
        initialized_state.gallery_selected_index = None

        metadata, state = toggle_metadata_format("JSON (.json)", initialized_state)

        assert "Select an image" in metadata

    def test_toggle_metadata_preserves_state(self, initialized_state, temp_dir):
        """Test that toggle preserves state."""
        outputs_dir = temp_dir / "outputs"
        initialized_state.gallery_images = [str(outputs_dir / "image1.png")]
        initialized_state.gallery_selected_index = 0

        _, state = toggle_metadata_format("Text (.txt)", initialized_state)

        # Selected index should be preserved
        assert state.gallery_selected_index == 0


# ============================================================================
# toggle_favorite Tests
# ============================================================================


class TestToggleFavorite:
    """Tests for toggle_favorite handler."""

    def test_toggle_favorite_adds_favorite(self, initialized_state, temp_dir):
        """Test toggling favorite adds to favorites."""
        outputs_dir = temp_dir / "outputs"
        image_path = str(outputs_dir / "image1.png")

        initialized_state.gallery_images = [image_path]
        initialized_state.gallery_selected_index = 0

        # Toggle on
        button_label, info_msg, state = toggle_favorite(initialized_state)

        assert button_label == "⭐ Unfavorite"
        assert "Added to favorites" in info_msg
        assert initialized_state.favorites_db.is_favorite(image_path)

    def test_toggle_favorite_removes_favorite(self, initialized_state, temp_dir):
        """Test toggling favorite removes from favorites."""
        outputs_dir = temp_dir / "outputs"
        image_path = str(outputs_dir / "image1.png")

        initialized_state.gallery_images = [image_path]
        initialized_state.gallery_selected_index = 0

        # Add to favorites first
        initialized_state.favorites_db.add_favorite(image_path)

        # Toggle off
        button_label, info_msg, state = toggle_favorite(initialized_state)

        assert button_label == "☆ Favorite"
        assert "Removed from favorites" in info_msg
        assert not initialized_state.favorites_db.is_favorite(image_path)

    def test_toggle_favorite_with_no_selection(self, initialized_state):
        """Test toggle_favorite with no image selected."""
        initialized_state.gallery_selected_index = None

        button_label, info_msg, state = toggle_favorite(initialized_state)

        assert button_label == "☆ Favorite"
        assert "No image selected" in info_msg


# ============================================================================
# apply_gallery_filter Tests
# ============================================================================


class TestApplyGalleryFilter:
    """Tests for apply_gallery_filter handler."""

    def test_filter_to_favorites_only(self, initialized_state, temp_dir):
        """Test filtering to show only favorites."""
        outputs_dir = temp_dir / "outputs"

        # Mark one image as favorite
        image1_path = str(outputs_dir / "image1.png")
        initialized_state.favorites_db.add_favorite(image1_path)

        # Apply filter
        filtered_images, state = apply_gallery_filter("Favorites Only", "", initialized_state)

        assert len(filtered_images) == 1
        assert image1_path in filtered_images
        assert state.gallery_filter == "favorites"

    def test_filter_to_all_images(self, initialized_state):
        """Test filtering to show all images."""
        filtered_images, state = apply_gallery_filter("All Images", "", initialized_state)

        assert len(filtered_images) == 3
        assert state.gallery_filter == "all"

    def test_filter_preserves_state(self, initialized_state):
        """Test that filtering preserves other state."""
        initialized_state.gallery_selected_index = 1

        _, state = apply_gallery_filter("All Images", "", initialized_state)

        # Selected index should be preserved
        assert state.gallery_selected_index == 1


# ============================================================================
# initialize_gallery_browser Tests
# ============================================================================


class TestInitializeGalleryBrowser:
    """Tests for initialize_gallery_browser handler."""

    def test_initialize_always_rescans(self, initialized_state, temp_dir):
        """Test that initialize_gallery_browser always rescans even if already initialized."""
        from pipeworks.ui.handlers.gallery import initialize_gallery_browser

        outputs_dir = temp_dir / "outputs"

        # Initial call should scan and find 3 images
        dropdown, path, images, state = initialize_gallery_browser(initialized_state)
        assert len(images) == 3
        assert state.gallery_initialized is True

        # Add a new image
        (outputs_dir / "image4.png").touch()

        # Call again - should rescan and find 4 images (not return cached 3)
        dropdown, path, images, state = initialize_gallery_browser(state)
        assert len(images) == 4, "Should rescan and find new image"

    def test_initialize_preserves_current_path(self, initialized_state, temp_dir):
        """Test that initialize_gallery_browser preserves current path on re-initialization."""
        from pipeworks.ui.handlers.gallery import initialize_gallery_browser

        outputs_dir = temp_dir / "outputs"
        subdir = outputs_dir / "subfolder"
        subdir.mkdir()
        (subdir / "subimage.png").touch()

        # Navigate to subfolder
        initialized_state.gallery_current_path = "subfolder"

        # Initialize should use current path
        dropdown, path, images, state = initialize_gallery_browser(initialized_state)
        assert path == "subfolder"
        assert len(images) == 1
        assert "subimage.png" in images[0]


# ============================================================================
# Integration Tests
# ============================================================================


class TestGalleryHandlerIntegration:
    """Integration tests for gallery handlers working together."""

    def test_refresh_then_select_workflow(self, initialized_state, temp_dir):
        """Test typical workflow: refresh then select image."""
        outputs_dir = temp_dir / "outputs"

        # Step 1: Refresh gallery
        images, state = refresh_gallery("", initialized_state)
        assert len(images) == 3
        assert state.gallery_selected_index is None

        # Step 2: Select an image
        evt = MagicMock()
        evt.index = 1
        image_path, metadata, _, state = select_gallery_image(evt, "Text (.txt)", state)

        assert state.gallery_selected_index == 1
        assert "image2.png" in image_path
        assert "Prompt for image 2" in metadata

    def test_select_favorite_refresh_workflow(self, initialized_state, temp_dir):
        """Test workflow: select, favorite, refresh."""
        outputs_dir = temp_dir / "outputs"

        # Refresh to populate
        images, state = refresh_gallery("", initialized_state)
        state.gallery_images = images

        # Select image
        state.gallery_selected_index = 0

        # Favorite it
        button_label, _, state = toggle_favorite(state)
        assert button_label == "⭐ Unfavorite"

        # Refresh (should clear selection)
        images, state = refresh_gallery("", state)
        assert state.gallery_selected_index is None

        # Image should still be favorited
        image_path = images[0]
        assert state.favorites_db.is_favorite(image_path)

    def test_filter_after_adding_images(self, initialized_state, temp_dir):
        """Test that filtering works after adding new images."""
        outputs_dir = temp_dir / "outputs"

        # Initial state
        images, state = refresh_gallery("", initialized_state)
        original_count = len(images)

        # Add new image
        new_image = outputs_dir / "image4.png"
        new_image.touch()

        # Refresh
        images, state = refresh_gallery("", state)
        assert len(images) == original_count + 1

        # Mark new image as favorite
        state.favorites_db.add_favorite(str(new_image))

        # Filter to favorites
        filtered, state = apply_gallery_filter("Favorites Only", "", state)
        assert len(filtered) == 1
        assert str(new_image) in filtered[0]
