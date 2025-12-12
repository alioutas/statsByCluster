# statsByCluster

Compute per-cluster statistics (count, Delaunay volume, surface area, radius of gyration) from a localization table containing `x,y,z` coordinates and a `cluster` id.

## Setup (conda)

```bash
conda env create -f environment.yml
conda activate statsbycluster
```

## Data

By default the script reads:

- `data/Location3_DC_filtered16_final_merged_clusters.csv` (tab-separated)

The table must include:

- `x`, `y`, `z` (coordinates)
- `cluster` (cluster id)
- `method` (optional; used only for plotting)

## Run

```bash
python statsByCluster.py
```

This writes summary stats to:

- `data/cluster_stats.csv`

## Notes

- `delaunay_volume(df)` accepts either a pandas DataFrame or a polars DataFrame.
- Units: `delaunay_volume` scales `volume` by `1e9` and `surface_area` by `1e6`, which corresponds to converting from µm³→mm³ and µm²→mm² if your coordinates are in micrometers.
