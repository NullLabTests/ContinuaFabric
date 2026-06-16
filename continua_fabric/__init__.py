from importlib.metadata import version

__version__ = version("continua_fabric")

from continua_fabric import core, nodes, meta, benchmarks  # noqa: F401
