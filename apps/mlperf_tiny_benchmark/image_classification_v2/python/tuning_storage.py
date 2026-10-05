"""Validated, rollback-capable publication for persistent tuning files."""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
import hashlib


@contextmanager
def writer_lock(directory):
    """Allow one process to publish schedules in a directory at a time."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".tune.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"another tuning writer holds {lock}; inspect it before removing it") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def publish_file_set(changes, *, validate=None):
    """Stage and publish a set of bytes, restoring the old set on caught errors.

    A value of ``None`` removes a path. The optional validator receives the
    staged path map before any destination is replaced.
    """
    normalized = {Path(path): data for path, data in changes.items()}
    if not normalized:
        raise ValueError("publication file set is empty")
    if any(data is not None and not isinstance(data, bytes) for data in normalized.values()):
        raise TypeError("published file contents must be bytes or None")

    staged = {}
    staged_paths = {}
    originals = {path: path.read_bytes() if path.is_file() else None for path in normalized}
    try:
        for path, data in normalized.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if data is None:
                staged[path] = None
                staged_paths[path] = None
                continue
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".stage", dir=path.parent
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            staged[path] = temporary
            staged_paths[path] = Path(temporary)
        if validate is not None:
            validate(staged_paths)
        for path, temporary in staged.items():
            if temporary is None:
                path.unlink(missing_ok=True)
            else:
                os.replace(temporary, path)
                staged[path] = None
        return tuple(normalized)
    except BaseException:
        for path, old in originals.items():
            try:
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(old)
            except OSError:
                # Preserve the triggering error; later integrity checks reject
                # any incomplete file set instead of treating it as valid.
                pass
        raise
    finally:
        for temporary in staged.values():
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


def config_files(directory):
    directory = Path(directory)
    return directory / "config.json", directory / "config.sha256"


def validate_config_snapshot(directory, config_bytes, config_sha256, *, allow_mismatch=False):
    """Require a complete, byte-identical configuration snapshot when present."""
    config_path, checksum_path = config_files(directory)
    if not config_path.exists() and not checksum_path.exists():
        return False
    if not config_path.is_file() or not checksum_path.is_file():
        raise ValueError("tuning config snapshot/checksum set is incomplete")
    stored = config_path.read_bytes()
    checksum = checksum_path.read_text(encoding="ascii").strip()
    if hashlib.sha256(stored).hexdigest() != checksum:
        raise ValueError("tuning config snapshot checksum is corrupt")
    if not allow_mismatch and (stored != config_bytes or checksum != config_sha256):
        raise ValueError("tuning results belong to a different VTA config; use full FSIM tuning to replace them")
    return True
