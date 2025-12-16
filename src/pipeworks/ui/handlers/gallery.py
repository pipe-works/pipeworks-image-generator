"""Gallery browser, favorites, and catalog handlers."""

import logging
from pathlib import Path

import gradio as gr

from ..models import UIState
from ..state import initialize_ui_state

logger = logging.getLogger(__name__)


def load_gallery_folder(
    selected_item: str, current_path: str, state: UIState
) -> tuple[gr.Dropdown, str, list[str], UIState]:
    """Navigate folders or load images from current path.

    Args:
        selected_item: Item selected from dropdown (folder or image)
        current_path: Current path being browsed
        state: UI state

    Returns:
        Tuple of (dropdown_update, new_path, gallery_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Handle folder navigation
        if selected_item and selected_item.startswith("📁"):
            folder_name = selected_item[2:].strip()  # Remove emoji prefix

            # Skip if it's just a placeholder
            if not folder_name or folder_name in ["(No folders)", "(Error)", "- Select folder --"]:
                return gr.update(), current_path, state.gallery_images, state

            if folder_name == "..":
                # Go up one level
                if current_path:
                    new_path = str(Path(current_path).parent)
                    if new_path == ".":
                        new_path = ""
                else:
                    new_path = ""
            else:
                # Navigate into folder
                new_path = str(Path(current_path) / folder_name) if current_path else folder_name

            # Update current path in state
            state.gallery_current_path = new_path

            # Get items in new path
            folders, _ = state.gallery_browser.get_items_in_path(new_path)

            # Build dropdown choices
            choices = ["-- Select folder --"]  # Neutral first choice
            if new_path:  # Add parent navigation if not at root
                choices.append("📁 ..")

            # Add folders with emoji
            for folder in folders:
                choices.append(f"📁 {folder}")

            # Scan for images in new path
            images = state.gallery_browser.scan_images(new_path)
            state.gallery_images = images

            # Update dropdown with new choices and set to neutral selection
            logger.info(f"Navigated to: {new_path}, {len(images)} images, {len(folders)} folders")
            return gr.update(choices=choices, value="-- Select folder --"), new_path, images, state

        else:
            # Not a folder selection, just return current state
            return gr.update(), current_path, state.gallery_images, state

    except Exception as e:
        logger.error(f"Error loading gallery folder: {e}", exc_info=True)
        return gr.update(), current_path, state.gallery_images, state


def select_gallery_image(
    evt: gr.SelectData, show_json: str, state: UIState
) -> tuple[str, str, str, UIState]:
    """Display selected image with its metadata.

    Args:
        evt: Gradio SelectData event containing selected index
        show_json: Which metadata format to show ("Text (.txt)" or "JSON (.json)")
        state: UI state

    Returns:
        Tuple of (image_path, metadata_markdown, favorite_button_label, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Get selected index from event
        selected_index = evt.index

        # Check if we have images cached
        if not state.gallery_images or selected_index >= len(state.gallery_images):
            return None, "*No image selected*", "☆ Favorite", state

        # Get image path
        image_path = state.gallery_images[selected_index]
        image_name = Path(image_path).name

        # Store selected index in state
        state.gallery_selected_index = selected_index

        # Check if image is favorited
        is_favorited = state.favorites_db.is_favorite(image_path)
        favorite_button_label = "⭐ Unfavorite" if is_favorited else "☆ Favorite"

        # Read metadata based on toggle
        if "JSON" in show_json:
            json_data = state.gallery_browser.read_json_metadata(image_path)
            metadata_md = state.gallery_browser.format_metadata_json(json_data, image_name)
        else:
            txt_content = state.gallery_browser.read_txt_metadata(image_path)
            metadata_md = state.gallery_browser.format_metadata_txt(txt_content, image_name)

        return image_path, metadata_md, favorite_button_label, state

    except Exception as e:
        logger.error(f"Error selecting gallery image: {e}", exc_info=True)
        return None, f"*Error loading image: {str(e)}*", "☆ Favorite", state


def refresh_gallery(current_path: str, state: UIState) -> tuple[list[str], UIState]:
    """Refresh image list in current path.

    Args:
        current_path: Current path being browsed
        state: UI state

    Returns:
        Tuple of (gallery_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Scan for images
        images = state.gallery_browser.scan_images(current_path)
        state.gallery_images = images

        return images, state

    except Exception as e:
        logger.error(f"Error refreshing gallery: {e}", exc_info=True)
        return [], state


def toggle_metadata_format(show_json: str, state: UIState) -> tuple[str, UIState]:
    """Switch between .txt and .json metadata view.

    Args:
        show_json: Which metadata format to show ("Text (.txt)" or "JSON (.json)")
        state: UI state

    Returns:
        Tuple of (metadata_markdown, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Check if we have a selected image
        if state.gallery_selected_index is None or not state.gallery_images:
            return "*Select an image to view metadata*", state

        if state.gallery_selected_index >= len(state.gallery_images):
            return "*No image selected*", state

        # Get selected image
        image_path = state.gallery_images[state.gallery_selected_index]
        image_name = Path(image_path).name

        # Read metadata based on toggle
        if "JSON" in show_json:
            json_data = state.gallery_browser.read_json_metadata(image_path)
            metadata_md = state.gallery_browser.format_metadata_json(json_data, image_name)
        else:
            txt_content = state.gallery_browser.read_txt_metadata(image_path)
            metadata_md = state.gallery_browser.format_metadata_txt(txt_content, image_name)

        return metadata_md, state

    except Exception as e:
        logger.error(f"Error toggling metadata format: {e}", exc_info=True)
        return f"*Error loading metadata: {str(e)}*", state


def initialize_gallery_browser(state: UIState) -> tuple[gr.Dropdown, str, list[str], UIState]:
    """Initialize gallery browser on tab load.

    Args:
        state: UI state

    Returns:
        Tuple of (dropdown_update, current_path, gallery_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Skip if already initialized to prevent infinite loop
        if state.gallery_initialized:
            return (
                gr.update(),
                state.gallery_current_path,
                state.gallery_images,
                state,
            )

        # Mark as initialized
        state.gallery_initialized = True

        # Start at root of outputs directory
        current_path = ""
        state.gallery_current_path = current_path

        # Get folders and images at root
        folders, _ = state.gallery_browser.get_items_in_path(current_path)

        # Build dropdown choices
        choices = ["-- Select folder --"]  # Neutral first choice
        for folder in folders:
            choices.append(f"📁 {folder}")

        # Scan for images at root
        images = state.gallery_browser.scan_images(current_path)
        state.gallery_images = images

        logger.info(f"Initialized gallery: {len(images)} images, {len(folders)} folders at root")
        return gr.update(choices=choices, value="-- Select folder --"), current_path, images, state

    except Exception as e:
        logger.error(f"Error initializing gallery browser: {e}", exc_info=True)
        return gr.update(choices=["(Error)"]), "", [], state


def toggle_favorite(state: UIState) -> tuple[str, str, UIState]:
    """Toggle favorite status of currently selected image.

    Args:
        state: UI state

    Returns:
        Tuple of (favorite_button_label, info_message, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Get selected index from state
        selected_index = state.gallery_selected_index

        # Get image path
        if selected_index is None or not state.gallery_images:
            return "☆ Favorite", "*No image selected*", state

        if selected_index >= len(state.gallery_images):
            return "☆ Favorite", "*Invalid image index*", state

        image_path = state.gallery_images[selected_index]

        # Toggle favorite status
        is_now_favorited = state.favorites_db.toggle_favorite(image_path)

        # Update button label
        if is_now_favorited:
            button_label = "⭐ Unfavorite"
            info_message = "*Added to favorites*"
        else:
            button_label = "☆ Favorite"
            info_message = "*Removed from favorites*"

        logger.info(f"Toggled favorite for {image_path}: {is_now_favorited}")
        return button_label, info_message, state

    except Exception as e:
        logger.error(f"Error toggling favorite: {e}", exc_info=True)
        return "☆ Favorite", f"*Error: {str(e)}*", state


def apply_gallery_filter(
    filter_mode: str, current_path: str, state: UIState
) -> tuple[list[str], UIState]:
    """Filter gallery by favorites.

    Args:
        filter_mode: "All Images" or "Favorites Only"
        current_path: Current path being browsed
        state: UI state

    Returns:
        Tuple of (filtered_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Get all images in current path
        all_images = state.gallery_browser.scan_images(current_path)

        # Apply filter
        if filter_mode == "Favorites Only":
            # Filter to only favorited images
            filtered_images = [
                img for img in all_images if state.favorites_db.is_favorite(img)
            ]
            state.gallery_filter = "favorites"
            logger.info(
                f"Filtered to favorites: {len(filtered_images)} / {len(all_images)} images"
            )
        else:
            # Show all images
            filtered_images = all_images
            state.gallery_filter = "all"
            logger.info(f"Showing all images: {len(filtered_images)}")

        # Update cached images
        state.gallery_images = filtered_images

        return filtered_images, state

    except Exception as e:
        logger.error(f"Error applying gallery filter: {e}", exc_info=True)
        return [], state


def move_favorites_to_catalog(state: UIState) -> tuple[str, list[str], UIState]:
    """Move all favorited images to catalog.

    Args:
        state: UI state

    Returns:
        Tuple of (info_message, refreshed_gallery_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Get count before move
        favorite_count = state.favorites_db.get_favorite_count()

        if favorite_count == 0:
            return "*No favorites to move*", state.gallery_images, state

        # Perform move operation
        logger.info(f"Starting move of {favorite_count} favorites to catalog")
        stats = state.catalog_manager.move_favorites_to_catalog()

        # Format result message
        if stats["moved"] > 0:
            msg = f"**Moved {stats['moved']} image(s) to catalog**"

            if stats["skipped"] > 0:
                msg += f"\n\n*Skipped {stats['skipped']} (already moved or missing)*"

            if stats["failed"] > 0:
                msg += f"\n\n*Failed to move {stats['failed']} image(s)*"

            if stats["errors"]:
                # Show first few errors
                error_list = "\n".join(f"- {err}" for err in stats["errors"][:3])
                msg += f"\n\n**Errors:**\n{error_list}"
                if len(stats["errors"]) > 3:
                    msg += f"\n- ... and {len(stats['errors']) - 3} more"

        elif stats["skipped"] > 0:
            msg = f"*All {stats['skipped']} favorite(s) already moved or missing*"
        else:
            msg = "*Failed to move favorites. Check logs for details.*"

        # Refresh gallery view
        refreshed_images = state.gallery_browser.scan_images(state.gallery_current_path)
        state.gallery_images = refreshed_images

        logger.info(f"Move complete: {stats}")
        return msg, refreshed_images, state

    except Exception as e:
        logger.error(f"Error moving favorites to catalog: {e}", exc_info=True)
        return f"*Error: {str(e)}*", state.gallery_images, state


def switch_gallery_root(
    root_choice: str, state: UIState
) -> tuple[gr.Dropdown, str, list[str], UIState]:
    """Switch between outputs and catalog browsing.

    Args:
        root_choice: "📁 outputs" or "📁 catalog"
        state: UI state

    Returns:
        Tuple of (dropdown_update, current_path, gallery_images, updated_state)
    """
    try:
        # Initialize state if needed
        state = initialize_ui_state(state)

        # Extract root name (remove emoji prefix)
        root_name = root_choice.replace("📁 ", "").strip()

        # Switch gallery browser root
        state.gallery_browser.set_root(root_name)
        state.gallery_root = root_name

        # Reset to root path
        current_path = ""
        state.gallery_current_path = current_path

        # Get folders and images at new root
        folders, _ = state.gallery_browser.get_items_in_path(current_path)

        # Build dropdown choices
        choices = ["-- Select folder --"]  # Neutral first choice
        for folder in folders:
            choices.append(f"📁 {folder}")

        # Scan for images at new root
        images = state.gallery_browser.scan_images(current_path)
        state.gallery_images = images

        logger.info(
            f"Switched to {root_name} root: {len(images)} images, {len(folders)} folders"
        )
        return gr.update(choices=choices, value="-- Select folder --"), current_path, images, state

    except Exception as e:
        logger.error(f"Error switching gallery root: {e}", exc_info=True)
        return gr.update(), state.gallery_current_path, state.gallery_images, state
