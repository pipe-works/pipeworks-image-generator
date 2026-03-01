"""Tests for the real frontend template shipped with the package.

These checks complement the FastAPI integration suite, which uses a minimal
temporary template fixture for speed and isolation.  The assertions here read
the repository's actual `index.html` so frontend-only regressions in lightbox
controls and script wiring are still caught by automated tests.
"""

from __future__ import annotations

from pathlib import Path


def test_index_template_includes_output_lightbox_transport_controls() -> None:
    """The shipped template should expose the new Output lightbox controls."""
    template_path = (
        Path(__file__).resolve().parents[2] / "src" / "pipeworks" / "templates" / "index.html"
    )
    html = template_path.read_text(encoding="utf-8")

    assert 'id="lb-btn-prev"' in html
    assert 'id="lb-btn-play"' in html
    assert 'id="lb-btn-pause"' in html
    assert 'id="lb-btn-stop"' in html
    assert 'id="lb-btn-next"' in html
    assert 'id="lb-nav-status"' in html
    assert 'id="lb-nav-hint"' in html
    assert 'type="module" src="/static/js/app.js"' in html
