"""
implementation-agnostic POSIX-like file helpers.
"""
import io
import os
import shutil
import typing


F = typing.TypeVar("F")

walk = os.walk
listdir = os.listdir
exists = os.path.exists
cp = shutil.copy2
open = io.open


def join(root, *parts):
    r"""
    Join path components, like os.path.join or posixpath.join.

    Differences:

    - The parts are always treated as relative paths
    - "/" or "\" separator will be picked based on the root, not based
      on whatever OS we are running on.
    - joining "" doesn't add an extra slash

    So e.g.

        join('a/b', '') -> 'a/b'
        join(r'C:\a\b','c') -> 'C:\a\b\c'
        join('/a/b','c') -> '/a/b/c'

    """
    isep = max(root.rfind("/"), root.rfind("\\"))
    sep = root[isep] if isep >= 0 else "/"
    rhs = sep.join(filter(None, (part.strip("/\\") for part in parts)))
    if not rhs:
        return root
    elif root.endswith(sep):
        return root + rhs
    else:
        return sep.join([root, rhs])


def basename(path):
    isep = max(path.rfind("/"), path.rfind("\\"))
    return path[isep + 1 :]


def dirname(path):
    isep = max(path.rfind("/"), path.rfind("\\"))
    if isep >= 0:
        return path[:isep]
    else:
        return ""


def read_bytes(
    path,
    start=0,
    stop=None,
) -> bytes:
    """
    Read contents of a file (or part of them) as a byte string.
    """
    with open(path, "rb") as f:
        if start:
            f.seek(start)
        if stop is None:
            return f.read()
        else:
            return f.read(stop - start)


async def aread_bytes(
    path,
    start=0,
    stop=None,
) -> bytes:
    return read_bytes(path, start, stop)
