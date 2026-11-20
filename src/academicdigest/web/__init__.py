"""Web UI + JSON API + Atom feed over the digest database (`academicdigest serve`)."""

from .app import create_app

__all__ = ["create_app"]
