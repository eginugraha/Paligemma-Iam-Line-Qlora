"""Concrete InferenceEngine implementations.

Each sub-module in this package provides one engine class that satisfies the
`InferenceEngine` Protocol defined in `htr_sp2.engine`. Engines are imported lazily
inside `get_engine()` so unused engines (e.g., RunPodEngine in test runs) never
trigger import errors from missing optional dependencies.
"""
