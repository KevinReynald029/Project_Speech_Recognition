from pathlib import Path

from .common import ScoringError, load_json


def create_backend(config):
    """Path relatif dihitung dari folder proyek."""
    if isinstance(config, (str, Path)):
        file = Path(config).resolve()
        values = dict(load_json(file))
        root = (file.parent / values.pop("project_root", ".")).resolve()
    else:
        values = dict(config)
        root = Path(values.pop("project_root", Path.cwd())).resolve()
    for key in ("bin", "model", "lang", "ivector_extractor", "scratch"):
        if key in values:
            value = Path(values[key]).expanduser()
            values[key] = str(value if value.is_absolute() else (root / value).resolve())
    if values.get("backend") != "native":
        raise ScoringError("backend_unavailable", "Use the native GOP extractor configuration")
    from .native import NativeKaldiBackend
    return NativeKaldiBackend(values)
