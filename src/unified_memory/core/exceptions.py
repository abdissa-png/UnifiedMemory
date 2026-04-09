"""
Custom exception hierarchy for unified memory.
"""

from __future__ import annotations


class UnifiedMemoryError(Exception):
    """Base exception for unified memory."""


class ConfigurationError(UnifiedMemoryError):
    """Invalid or missing configuration."""


class ProviderNotFoundError(UnifiedMemoryError):
    """Requested provider or model is not registered."""


class NamespaceNotFoundError(UnifiedMemoryError):
    """Namespace does not exist or is inaccessible."""

class TenantNotFoundError(UnifiedMemoryError):
    """Tenant does not exist or is inaccessible."""

class TenantConfigNotFoundError(UnifiedMemoryError):
    """Tenant configuration does not exist or is inaccessible."""

class TenantConfigConflictError(UnifiedMemoryError):
    """Tenant configuration conflict."""


class BackendTransientError(UnifiedMemoryError):
    """Retryable failure from an external service."""


class IngestionError(UnifiedMemoryError):
    """Failure during document ingestion."""


class CASConflictError(UnifiedMemoryError):
    """CAS retry budget exhausted."""
