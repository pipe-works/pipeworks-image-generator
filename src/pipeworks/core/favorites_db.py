"""SQLite database for tracking favorited images."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class FavoritesDB:
    """Manage favorites database using SQLite.

    Tracks favorited images by their relative paths from project root.
    Supports adding, removing, and querying favorite status.
    """

    def __init__(self, db_path: Path):
        """Initialize the favorites database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()
        logger.info(f"Initialized favorites database at {self.db_path}")

    def _initialize_db(self) -> None:
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create favorites table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS favorites (
                    image_path TEXT PRIMARY KEY,
                    favorited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

            # Create index for faster queries sorted by date
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_favorited_at
                ON favorites(favorited_at DESC)
                """)

            conn.commit()

    def _normalize_path(self, image_path: str | Path) -> str:
        """Normalize image path to relative path from project root.

        Path normalization ensures consistent storage and comparison of paths
        across different operating systems and path formats. This method:
        1. Converts absolute paths to relative (when possible)
        2. Standardizes path separators to forward slashes
        3. Maintains cross-platform compatibility

        Args:
            image_path: Absolute or relative image path

        Returns:
            Normalized relative path as string

        Notes:
            - Relative paths are stored as-is (already relative to project root)
            - Absolute paths within project are converted to relative
            - Absolute paths outside project are stored as absolute
            - Forward slashes used for consistency across Windows/Unix
        """
        path = Path(image_path)

        # If absolute, try to make relative to current working directory
        # This allows for portable database that works across different setups
        if path.is_absolute():
            try:
                # Try to express path relative to project root (cwd)
                cwd = Path.cwd()
                path = path.relative_to(cwd)
            except ValueError:
                # Path is outside project directory (e.g., /tmp/image.png)
                # Store as absolute path in this case
                pass

        # Convert to string with forward slashes for cross-platform consistency
        # Windows paths use backslashes, but forward slashes work everywhere
        return str(path).replace("\\", "/")

    def add_favorite(self, image_path: str | Path) -> bool:
        """Add an image to favorites.

        Args:
            image_path: Path to image file

        Returns:
            True if added successfully, False if already favorited
        """
        normalized_path = self._normalize_path(image_path)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Use INSERT OR IGNORE to handle duplicate entries gracefully
                # If image_path already exists (PRIMARY KEY), the insert is ignored
                # This is more efficient than checking for existence first
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO favorites (image_path, favorited_at)
                    VALUES (?, ?)
                    """,
                    (normalized_path, datetime.now().isoformat()),
                )
                conn.commit()

                # Check if row was actually inserted (rowcount > 0)
                # or ignored because it already existed (rowcount == 0)
                was_inserted = cursor.rowcount > 0
                if was_inserted:
                    logger.info(f"Added to favorites: {normalized_path}")
                else:
                    logger.debug(f"Already in favorites: {normalized_path}")

                return was_inserted

        except sqlite3.Error as e:
            logger.error(f"Error adding favorite {normalized_path}: {e}")
            return False

    def remove_favorite(self, image_path: str | Path) -> bool:
        """Remove an image from favorites.

        Args:
            image_path: Path to image file

        Returns:
            True if removed successfully, False if not in favorites
        """
        normalized_path = self._normalize_path(image_path)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    DELETE FROM favorites WHERE image_path = ?
                    """,
                    (normalized_path,),
                )
                conn.commit()

                # Check if row was actually deleted
                was_deleted = cursor.rowcount > 0
                if was_deleted:
                    logger.info(f"Removed from favorites: {normalized_path}")
                else:
                    logger.debug(f"Not in favorites: {normalized_path}")

                return was_deleted

        except sqlite3.Error as e:
            logger.error(f"Error removing favorite {normalized_path}: {e}")
            return False

    def is_favorite(self, image_path: str | Path) -> bool:
        """Check if an image is in favorites.

        Args:
            image_path: Path to image file

        Returns:
            True if favorited, False otherwise
        """
        normalized_path = self._normalize_path(image_path)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT 1 FROM favorites WHERE image_path = ? LIMIT 1
                    """,
                    (normalized_path,),
                )
                result = cursor.fetchone()
                return result is not None

        except sqlite3.Error as e:
            logger.error(f"Error checking favorite status for {normalized_path}: {e}")
            return False

    def get_all_favorites(self) -> list[str]:
        """Get all favorited image paths.

        Returns:
            List of image paths, sorted by favorited date (newest first)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT image_path FROM favorites ORDER BY favorited_at DESC
                    """)
                results = cursor.fetchall()
                return [row[0] for row in results]

        except sqlite3.Error as e:
            logger.error(f"Error getting favorites: {e}")
            return []

    def get_favorite_count(self) -> int:
        """Get total count of favorited images.

        Returns:
            Number of favorited images
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM favorites")
                result = cursor.fetchone()
                return result[0] if result else 0

        except sqlite3.Error as e:
            logger.error(f"Error getting favorite count: {e}")
            return 0

    def clear_favorites(self) -> None:
        """Clear all favorites from database.

        This is primarily for testing purposes.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM favorites")
                conn.commit()
                logger.info("Cleared all favorites")

        except sqlite3.Error as e:
            logger.error(f"Error clearing favorites: {e}")

    def toggle_favorite(self, image_path: str | Path) -> bool:
        """Toggle favorite status of an image.

        Args:
            image_path: Path to image file

        Returns:
            True if now favorited, False if unfavorited
        """
        if self.is_favorite(image_path):
            self.remove_favorite(image_path)
            return False
        else:
            self.add_favorite(image_path)
            return True
