"""Pipe-Works Image Generator — FastAPI REST API layer.

This package contains the FastAPI application, Pydantic request/response
models, and the prompt compilation logic.

Modules
-------
main
    FastAPI application with all route handlers and the ``main()`` CLI
    entry point.
models
    Pydantic models for API request and response validation.
prompt_builder
    Three-part prompt template compilation.
gallery_store
    File-backed gallery reconciliation, filtering, and pagination helpers.
"""
