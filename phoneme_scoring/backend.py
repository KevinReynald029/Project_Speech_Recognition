from pathlib import Path

from .common import load_json


def create_backend(config):
    """Resolve shared config paths against its project root, not the caller cwd."""
    if isinstance(config, (str, Path)):
        file = Path(config).resolve()
        values = dict(load_json(file))
        root = (file.parent / values.pop("project_root", ".")).resolve()
    else:
        values = dict(config)
        root = Path(values.pop("project_root", Path.cwd())).resolve()
    for key in ("bin", "model", "lang", "ivector_extractor", "recipe", "scratch"):
        if key in values:
            value = Path(values[key]).expanduser()
            values[key] = str(value if value.is_absolute() else (root / value).resolve())
    if values.get("backend") == "native":
        from .native import NativeKaldiBackend
        return NativeKaldiBackend(values)
    from .kaldi import KaldiBackend
    return KaldiBackend(values)
