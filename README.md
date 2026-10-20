# PDI-TP

## Dataset

The `grupo_5` dataset (900 image/label pairs, ~84MB) is **not committed to this
repo** — it's attached as a zip to the [`dataset-v1` release](https://github.com/candeperles/PDI-TP/releases/tag/dataset-v1)
instead, so clones/pulls stay fast and git history doesn't bloat.

After cloning, fetch it with:

```bash
./scripts/download_data.sh
```

This downloads and extracts the dataset into `data/grupo_5/` (gitignored —
each teammate keeps their own local copy). Requires the [GitHub CLI](https://cli.github.com/)
(`gh`) for authenticated access, or falls back to a plain `curl` download.

If you add to or change the dataset, re-zip `data/grupo_5/` and upload a new
release (e.g. `dataset-v2`), then update the `TAG` in `scripts/download_data.sh`.