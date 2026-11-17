"""XCAPE: XRD-Constrained Accelerated Prediction Engine.

Deliberately free of imports so that ``import xcape`` stays cheap: the CLI
entry point (``xcape.main:cli``) and the heavy dependencies it pulls in
(torch, pymatgen, pyxtal) are only loaded when a submodule is imported.

``__version__`` is the single source of truth for the package version;
pyproject.toml reads it via ``[tool.setuptools.dynamic]``.
"""

__version__ = "0.1.0"
