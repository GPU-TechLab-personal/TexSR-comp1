"""Where comparison sets live on disk.

    <root>/<material>/compare/<run>/<map>/<column>.png

<run> is the producer's run stamp (the MatUp baseline uses its own run's
timestamp, so a set always says which run its columns came from). Runs are
ordered by name: the newest is sorted(...)[-1], the rule every report follows.
"""
import glob
import os

SET_PARENT = "compare"


def newest(dirs, want=None):
    """sorted(dirs)[-1], or the one whose basename is `want`; None when none match."""
    dirs = sorted(dirs)
    if want:
        dirs = [d for d in dirs if os.path.basename(d) == want]
    return dirs[-1] if dirs else None


def run_dirs(parent):
    """Every sub-directory of `parent` (a folder of run stamps), sorted."""
    return sorted(d for d in glob.glob(os.path.join(parent, "*")) if os.path.isdir(d))


def materials(root, need):
    """Material folders under `root` that contain the sub-folder `need`."""
    return sorted(d for d in glob.glob(os.path.join(root, "*"))
                  if os.path.isdir(os.path.join(d, need)))


def set_dir(mat_dir, run):
    return os.path.join(mat_dir, SET_PARENT, run)


def set_dirs(mat_dir):
    return run_dirs(os.path.join(mat_dir, SET_PARENT))


def newest_set(mat_dir, want=None):
    return newest(set_dirs(mat_dir), want)


def cell_path(set_path, map_name, column):
    return os.path.join(set_path, map_name, column + ".png")
