from .core import main, query, query_service, compress

try:
    from ._version import version as __version__
    from ._version import version as __commit_id__
except ImportError:  # pragma: no cover
    __version__ = "0+unknown"
    __commit_id__ = None

__all__ = ["main", "query", "query_service", "compress", "__version__", "__commit_id__"]
