from importlib import import_module

from .registry import get_method, get_parser, list_methods, list_parsers, method_info, parser_info

__all__ = [
    "get_method",
    "get_parser",
    "list_methods",
    "list_parsers",
    "method_info",
    "parser_info",
]

_required_modules = ("base_pointwise",)
_optional_modules = (
    "base_pairwise",
    "swap_and_synthesize",
    "refbased_pairwise",
    "multimodel_synthesize",
    "decompose_single",
    "prometheus",
    "chateval",
    "multi_aspect_aggregation",
    "prepair",
    "protocol_consistency",
    "wildbench",
    "alpacaeval",
)

for _module in _required_modules:
    import_module(f"{__name__}.{_module}")

for _module in _optional_modules:
    try:
        import_module(f"{__name__}.{_module}")
    except ModuleNotFoundError:
        continue
