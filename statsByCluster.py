
#%%
path = "./data/Location3_DC_filtered16_final_merged_clusters.csv" 
separator= "\t" #"\t" for tab and "," for comma


#%%
import pandas as pd
import numpy as np
try:
    import polars as pl
except ModuleNotFoundError:  # optional dependency
    pl = None

# get folder path for saving df with stats
import os
folder_path = os.path.dirname(path)

# %% Calculate the volumes and surfaces of the cluster output of PULSE, GT, and DBSCAN
try:
    from scipy.spatial import Delaunay, QhullError
except ModuleNotFoundError as e:  # optional at import time; required to run geometry
    Delaunay = None
    QhullError = None
    _scipy_import_error = e
else:
    _scipy_import_error = None

def perturb_data(points: np.ndarray, magnitude: float, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    return points + rng.normal(loc=0.0, scale=magnitude, size=points.shape)

def triangulate(points, initial_magnitude=1e-16, max_magnitude=1e-9):
    if Delaunay is None:
        raise ModuleNotFoundError(
            "scipy is required for Delaunay triangulation; install `scipy` to use `delaunay_volume`."
        ) from _scipy_import_error

    magnitude = initial_magnitude
    while magnitude <= max_magnitude:
        try:
            tri = Delaunay(points)
            return tri  # Triangulation successful, return the result
        except QhullError:
            # If a QhullError occurs, perturb the data and retry
            print(f"QhullError encountered. Retrying with magnitude {magnitude}...")
            points = perturb_data(points, magnitude)
            magnitude *= 10  # Increase the magnitude for the next iteration

    # If this point is reached, triangulation has failed even after perturbations
    raise QhullError("Unable to perform Delaunay triangulation with the given points.")

def _df_points_to_numpy(df, cols: list[str]) -> np.ndarray:
    if hasattr(df, "select"):
        return df.select(cols).to_numpy()
    return df[cols].to_numpy()

def delaunay_volume(df, only_2d: bool = False):
    # use_gpu = pulse.use_gpu
    if only_2d:
        points = _df_points_to_numpy(df, ['x', 'y'])
    else:
        points = _df_points_to_numpy(df, ['x', 'y', 'z'])

    # if use_gpu and isinstance(points, cp.ndarray):
    #     points = points.get()

    if QhullError is None:
        tri = triangulate(points)
    else:
        try:
            tri = triangulate(points)
        except QhullError as e:
            print(f"Triangulation failed: {e}")
            return None, None  # Handle the error as appropriate

    simplex_points = points[tri.simplices]

    v1 = simplex_points[:, 1, :] - simplex_points[:, 0, :]
    v2 = simplex_points[:, 2, :] - simplex_points[:, 0, :]

    if only_2d:
        # 2D cross product: v1 × v2 = v1_x * v2_y - v1_y * v2_x
        cross_product = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        tri_areas = 0.5 * np.abs(cross_product)
        return None, tri_areas.sum() / 1e6

    v3 = simplex_points[:, 3, :] - simplex_points[:, 0, :]

    tetra_volumes = np.abs(np.einsum('ij,ij->i', v1, np.cross(v2, v3))) / 6.0

    boundary_faces = tri.neighbors == -1
    boundary_simplex_indices, boundary_face_indices = np.where(boundary_faces)
    boundary_tetra = simplex_points[boundary_simplex_indices]  # (k, 4, 3)

    all_vertex_indices = np.tile(np.arange(4), (len(boundary_face_indices), 1))
    face_mask = all_vertex_indices != boundary_face_indices[:, None]
    boundary_triangles = boundary_tetra[face_mask].reshape(-1, 3, 3)

    boundary_areas = (
        np.linalg.norm(
            np.cross(
                boundary_triangles[:, 1, :] - boundary_triangles[:, 0, :],
                boundary_triangles[:, 2, :] - boundary_triangles[:, 0, :],
            ),
            axis=1,
        )
        / 2.0
    )

    surface_area = boundary_areas.sum()

    return tetra_volumes.sum()/ 1e9 , surface_area  / 1e6


if __name__ == "__main__":
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"Missing plotting dependency: {e.name}. Install it, or import `delaunay_volume` without running this script."
        ) from e

    #%%
    # read in the data
    df = pd.read_csv(path, sep=separator)

    #%%
    # Calculates the number of localizations, volume, surface area and radius of gyration for each cluster
    cluster_stats = []
    for cluster_id in df["cluster"].unique():
        df_cluster = df[df["cluster"] == cluster_id]
        num_localizations = len(df_cluster)
        volume_mm3, surface_area_mm2 = delaunay_volume(df_cluster)
        if volume_mm3 is None or surface_area_mm2 is None:
            continue  # Skip clusters where volume/surface area calculation failed
        centroid = df_cluster[["x", "y", "z"]].mean().to_numpy()
        rg = np.sqrt(np.mean(np.sum((df_cluster[["x", "y", "z"]].to_numpy() - centroid) ** 2, axis=1)))

        cluster_stats.append(
            {
                "cluster": cluster_id,
                "num_localizations": num_localizations,
                "volume_mm3": volume_mm3,
                "surface_area_mm2": surface_area_mm2,
                "radius_of_gyration_nm": rg,
            }
        )

    if pl is not None:
        df_cluster_stats = pl.DataFrame(cluster_stats).to_pandas()
    else:
        df_cluster_stats = pd.DataFrame(cluster_stats)

    df_cluster_stats.to_csv(os.path.join(folder_path, "cluster_stats.csv"), index=False)
# %%
