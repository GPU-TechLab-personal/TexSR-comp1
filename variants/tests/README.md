# variants/tests/

The static check that `variants/deeptex/paper_loop/` is still the upstream
DeepTextures trainer plus hook calls, and nothing else. (The equivalence tests
against the old code passed and are kept at tag `variants-verified`.)

## Contents

| file | what it does |
|---|---|
| `l0_skeleton.py` | diffs `paper_loop/loop.py` and `psnr_loop.py` against the submodules' trainers and checks every difference is a hook call; keeps the committed `.diff` files current |

## How it works

Two pairs are compared, upstream -> ours, with the committed diff beside them:

| upstream | ours | committed diff |
|---|---|---|
| `external/DeepTextures/train_esrgan.py` | `variants/deeptex/paper_loop/loop.py` | `variants/deeptex/paper_loop/upstream.diff` |
| `external/esrgan-tf2/train_psnr.py` | `variants/deeptex/paper_loop/psnr_loop.py` | `variants/deeptex/paper_loop/psnr_upstream.diff` |

Both files are read with whitespace normalised and blank lines dropped (a
hook's `with` block re-indents the two `apply_gradients`, which is not a
change), then diffed line by line:

* every line ours **adds** must call into the hooks (contain `hooks`);
* every line ours **removes** must occur verbatim in
  `variants/deeptex/paper_loop/hooks.py`, where the default implementation (or
  the comment beside it) keeps it;
* the committed `.diff` must equal a fresh unified diff of the pair.

So removing the hook calls from `loop.py` and substituting each hook's default
gives back `train_esrgan.py`. It reads files only: no docker, no GPU, seconds,
but the two submodules must be checked out.

Output is one line per pair, `PASS|FAIL <ours> vs <upstream>: N lines added
(all hook calls), M upstream lines moved into hooks.py`, followed by each
problem. Exit code 1 if either pair fails, else 0.

## Run

Run it after any change to `variants/deeptex/paper_loop/`, from the repo root:

```
python variants/tests/l0_skeleton.py           # check
python variants/tests/l0_skeleton.py --write   # regenerate paper_loop/upstream.diff, psnr_upstream.diff
```

Use `--write` after an intended change to `loop.py`, `psnr_loop.py` or
`hooks.py`, then commit the regenerated diffs.

## Known behaviour

* With `--write` the diffs are rewritten first and staleness is not reported
  (the docstring says it fails if they were stale; it does not). The
  hook-call and kept-in-`hooks.py` checks still run and still set the exit code.
