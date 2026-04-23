"""
Usage
python muspan_individual.py --sample-id "14w_15658_D_10_region_001" \
    --subregion-id "subregion_1" \
    --rotation-angle 205 
"""

import geopandas as gpd
from shapely.geometry import shape
from rasterio import features
import numpy as np

import muspan as ms
import spatialdata as sd

from muspan.query import query

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import sys
from pathlib import Path

import argparse

print("Imports Complete")

parser = argparse.ArgumentParser()
parser.add_argument("--sample-id", required=True)
parser.add_argument("--subregion-id", required=True)
parser.add_argument("--rotation-angle", required=True, type=float, default=0.5)

args = parser.parse_args()

SAMPLE_ID = args.sample_id
SUBREGION_ID = args.subregion_id

ROTATION_ANGLE = args.rotation_angle
ROTATION_ANGLE = (ROTATION_ANGLE+180-360)

SAMPLE_ID_LABEL = f"{SAMPLE_ID}_{SUBREGION_ID}"

# Output paths
base_dir = Path(f"../../../data2/processed/{SAMPLE_ID_LABEL}/output_v3_uberon_stratified")
sdata_path = Path(f"{base_dir}/{SAMPLE_ID_LABEL}_uberon_stratified.zarr")

output_dir = Path(f"../../../data2/notebooks/spatial/figures/{SAMPLE_ID_LABEL}")
output_dir.mkdir(exist_ok=True)

sdata = sd.read_zarr(sdata_path)

adata = sdata.tables['table_intensities']

# Get coordinates - adjust column names based on your data
# Common patterns:
# - adata.obs['centroid_x'], adata.obs['centroid_y']
# - adata.obsm['spatial']
# - adata.obs['x'], adata.obs['y']

# Example if coordinates are in obs:
coords = np.column_stack([
    adata.obs['x'].values,  # adjust column name
    adata.obs['y'].values   # adjust column name
])


# Create MuSpAn domain directly
domain = ms.domain('my_domain')

# Add points
domain.add_points(coords, collection_name='cells')

# Add cell type labels
domain.add_labels(
    'leiden_cell_type',
    adata.obs['leiden_cell_type'].values,
    add_labels_to='cells'
)

# Get the labels array
labels_array = sdata.labels['tissue_regions_manual'].values

# Convert raster labels to vector polygons
shapes_list = []
values_list = []
for geom, value in features.shapes(labels_array.astype('int32')):
    if value != 0:  # Skip background
        shapes_list.append(shape(geom))
        values_list.append(int(value))

# Create GeoDataFrame
tissue_shapes_gdf = gpd.GeoDataFrame({
    'geometry': shapes_list,
    'region_id': values_list
})

# Convert Shapely geometries to list of numpy arrays for MuSpAn
# MuSpAn expects: list of (n x 2) arrays of coordinates
muspan_shapes = []
for geom in tissue_shapes_gdf.geometry:
    if geom.geom_type == 'Polygon':
        # Get exterior coordinates as numpy array
        coords = np.array(geom.exterior.coords)
        muspan_shapes.append(coords)
    elif geom.geom_type == 'MultiPolygon':
        # Handle MultiPolygon by adding each polygon separately
        for poly in geom.geoms:
            coords = np.array(poly.exterior.coords)
            muspan_shapes.append(coords)

# Add the shapes to your MuSpAn domain
domain.add_shapes(
    muspan_shapes,
    collection_name='tissue_regions'
)

print("Shapes Added, Rotating Domain...")

"""
Rotate Domain
"""

def rotate_coords(coords, angle_degrees, center):
    """Rotate coordinates around a center point."""
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Translate to origin, rotate, translate back
    centered = coords - center
    rotated = np.column_stack([
        centered[:, 0] * cos_a - centered[:, 1] * sin_a,
        centered[:, 0] * sin_a + centered[:, 1] * cos_a
    ])
    return rotated + center

# Get the center of rotation (use the center of your data)
all_coords = np.column_stack([
    adata.obs['x'].values,
    adata.obs['y'].values
])
center = np.array([all_coords[:, 0].mean(), all_coords[:, 1].mean()])

# Rotate cell coordinates
rotated_cell_coords = rotate_coords(all_coords, ROTATION_ANGLE, center)

# Create a new domain with rotated coordinates
domain_rotated = ms.domain('my_domain_rotated')

# Add rotated points
domain_rotated.add_points(rotated_cell_coords, collection_name='cells')

# Add cell type labels
domain_rotated.add_labels(
    'leiden_cell_type',
    adata.obs['leiden_cell_type'].values,
    add_labels_to='cells'
)

# Rotate tissue region shapes
muspan_shapes_rotated = []
for shape_coords in muspan_shapes:
    rotated = rotate_coords(shape_coords, ROTATION_ANGLE, center)
    muspan_shapes_rotated.append(rotated)

# Add rotated shapes
domain_rotated.add_shapes(
    muspan_shapes_rotated,
    collection_name='tissue_regions'
)

print(domain_rotated)

"""
Set colours
"""

cell_type_colors = {
    'PI16 Fibroblast': '#0000FF',      # blue
    'Vascular Endothelium': '#FF0000',     # red
    'PRG4 Fibroblast': '#008000',     # forest green
    'Unlabeled': '#808080',     # standard gray
}

domain_rotated.update_colors(cell_type_colors, colors_to_update='labels', label_name='leiden_cell_type')

"""
Plot cell centroids
"""
fig,ax=plt.subplots(figsize=(3,6))
ms.visualise.visualise(domain_rotated,
                       color_by='leiden_cell_type',
                       objects_to_plot=('collection', 'cells'),
                       ax=ax,
                       marker_size=0.01,
                       add_cbar=False,
                       scatter_kwargs={'alpha': 0.75})  # plot the points
ax.set_title(f'{SAMPLE_ID_LABEL} \n Cell Centroid Plot. PI16 Fibroblasts (Blue) \n Vascular Endothelium (Red) PRG4 \n Firboblast (Green)')
plt.savefig(Path(f'{output_dir}/{SAMPLE_ID_LABEL}_01_centroids_plots.pdf', bbox_inches='tight'))

print("Centroids plotted...")

"""
Vascular KDE
"""

from shapely import affinity

CELL_TYPE_PLOT = 'Vascular Endothelium'
COLOUR_PLOT = 'Reds'
GEOMETRY_LINE_COLOUR = 'white'
FIGSIZE = (4, 6)

# Rotate the GeoDataFrame for plotting overlay
center_point = (center[0], center[1])
tissue_shapes_rotated_gdf = tissue_shapes_gdf.copy()
tissue_shapes_rotated_gdf['geometry'] = tissue_shapes_rotated_gdf['geometry'].apply(
    lambda geom: affinity.rotate(geom, ROTATION_ANGLE, origin=center_point)
)

# Create figure and axes first
fig, ax = plt.subplots(figsize=FIGSIZE)

# KDE with boundary overlay - pass ax via visualise_heatmap_kwargs
kde = ms.distribution.kernel_density_estimation(
    domain_rotated,
    population=('leiden_cell_type', CELL_TYPE_PLOT),
    visualise_output=True,
    visualise_heatmap_kwargs={
        "heatmap_cmap": COLOUR_PLOT,
        "ax": ax
    }
)

tissue_shapes_rotated_gdf.boundary.plot(ax=ax, color=GEOMETRY_LINE_COLOUR, linewidth=1.5)
ax.set_title(f'{SAMPLE_ID_LABEL} \n {CELL_TYPE_PLOT} Kernel Density \n Estimate with annotations \n overlaid ({GEOMETRY_LINE_COLOUR})')
plt.savefig(Path(f'{output_dir}/{SAMPLE_ID_LABEL}_02_vasc_KDE.pdf', bbox_inches='tight'))
plt.show()
print("Vasc KDE plotted...")

"""
PI16 KDE
"""
CELL_TYPE_PLOT = 'PI16 Fibroblast'
COLOUR_PLOT = 'Blues'
GEOMETRY_LINE_COLOUR = 'white'

FIGSIZE = (4, 6)

# Rotate the GeoDataFrame for plotting overlay
center_point = (center[0], center[1])
tissue_shapes_rotated_gdf = tissue_shapes_gdf.copy()
tissue_shapes_rotated_gdf['geometry'] = tissue_shapes_rotated_gdf['geometry'].apply(
    lambda geom: affinity.rotate(geom, ROTATION_ANGLE, origin=center_point)
)

# Create figure and axes first
fig, ax = plt.subplots(figsize=FIGSIZE)

# KDE with boundary overlay
kde = ms.distribution.kernel_density_estimation(
    domain_rotated,
    population=('leiden_cell_type', CELL_TYPE_PLOT),
    visualise_output=True,
    visualise_heatmap_kwargs={
        "heatmap_cmap": COLOUR_PLOT,
        "ax": ax
    }
)

ax = plt.gca()
tissue_shapes_rotated_gdf.boundary.plot(ax=ax, color=GEOMETRY_LINE_COLOUR, linewidth=1.5)
ax.set_title(f'{SAMPLE_ID_LABEL} \n {CELL_TYPE_PLOT} Kernel Density \n Estimate with annotations \n overlaid ({GEOMETRY_LINE_COLOUR})')
plt.savefig(Path(f'{output_dir}/{SAMPLE_ID_LABEL}_03_PI16_KDE.pdf', bbox_inches='tight'))
plt.show()
print("PI16 KDE plotted...")

"""
PI16 Hexgrid
"""
HEX_SIDE_LENGTH = (461.55)
HEX_SIDE_LENGTH_PLOT = round(HEX_SIDE_LENGTH * 0.325, 1)

ms.region_based.generate_hexgrid(domain_rotated, side_length=HEX_SIDE_LENGTH, regions_collection_name=f'Hexgrids {HEX_SIDE_LENGTH}')

CELL_TYPE_PLOT = 'PI16 Fibroblast'
CELL_TO_EXCLUDE = 'Unlabeled'

filtered_cells = query(domain_rotated, ('label', 'leiden_cell_type'), 'is', CELL_TYPE_PLOT)

fig,ax=plt.subplots(figsize=(5,6))
ms.visualise.visualise(domain_rotated,color_by=f'region counts: {CELL_TYPE_PLOT}',objects_to_plot=('collection',f'Hexgrids {HEX_SIDE_LENGTH}'),ax=ax)  # plot the tiles
ms.visualise.visualise(domain_rotated,
                       color_by='leiden_cell_type',
                       objects_to_plot=filtered_cells,
                       ax=ax,
                       marker_size=0.15,
                       add_cbar=False,
                       scatter_kwargs={'alpha': 0.75})  # plot the points
tissue_shapes_rotated_gdf.boundary.plot(ax=ax, color=GEOMETRY_LINE_COLOUR, linewidth=0.5)
ax.set_title(f'{SAMPLE_ID_LABEL} \n {CELL_TYPE_PLOT} Centroid \n Density Hexbins plotted (side={HEX_SIDE_LENGTH_PLOT}um) \n Annotations overlaid ({GEOMETRY_LINE_COLOUR})')
plt.savefig(Path(f'{output_dir}/{SAMPLE_ID_LABEL}_04_PI16_hex.pdf', bbox_inches='tight'))
print("PI16 Hexgrid plotted...")

"""
Vascular Hexgrid
"""
ms.region_based.generate_hexgrid(domain_rotated, side_length=HEX_SIDE_LENGTH, regions_collection_name=f'Hexgrids {HEX_SIDE_LENGTH}')

CELL_TYPE_PLOT = 'Vascular Endothelium'
CELL_TO_EXCLUDE = 'Unlabeled'

filtered_cells = query(domain_rotated, ('label', 'leiden_cell_type'), 'is', CELL_TYPE_PLOT)

fig,ax=plt.subplots(figsize=(5,6))
ms.visualise.visualise(domain_rotated,color_by=f'region counts: {CELL_TYPE_PLOT}',objects_to_plot=('collection',f'Hexgrids {HEX_SIDE_LENGTH}'),ax=ax)  # plot the tiles
ms.visualise.visualise(domain_rotated,
                       color_by='leiden_cell_type',
                       objects_to_plot=filtered_cells,
                       ax=ax,
                       marker_size=0.15,
                       add_cbar=False,
                       scatter_kwargs={'alpha': 0.75})  # plot the points
tissue_shapes_rotated_gdf.boundary.plot(ax=ax, color=GEOMETRY_LINE_COLOUR, linewidth=0.5)
ax.set_title(f'{SAMPLE_ID_LABEL} \n {CELL_TYPE_PLOT} Centroid \n Density Hexbins plotted (side={HEX_SIDE_LENGTH_PLOT}um) \n Annotations overlaid ({GEOMETRY_LINE_COLOUR})')
plt.savefig(Path(f'{output_dir}/{SAMPLE_ID_LABEL}_05_Vasc_hex.pdf', bbox_inches='tight'))
print("Vasc Hexgrid plotted, starting TCM...")

"""
TCM
"""
GEOMETRY_LINE_COLOUR = 'black'

CELL_TYPE_A = "PI16 Fibroblast"
CELL_TYPE_B = "Vascular Endothelium"

TCM_RADIUS = 50
TCM_KERNEL_RADIUS = 150
TCM_KERNEL_SIGMA = 50
TCM_MESH_STEP = 10

# Define the two populations
population_A = query(domain_rotated, ('label', 'leiden_cell_type'), 'is', CELL_TYPE_A)
population_B = query(domain_rotated, ('label', 'leiden_cell_type'), 'is', CELL_TYPE_B)

# Compute the TCM without auto-visualisation
tcm = ms.spatial_statistics.topographical_correlation_map(
    domain_rotated,
    population_A=population_A,
    population_B=population_B,
    radius_of_interest=(TCM_RADIUS*3.077),
    kernel_radius=(TCM_KERNEL_RADIUS*3.077),
    kernel_sigma=(TCM_KERNEL_SIGMA*3.077),
    mesh_step=(TCM_MESH_STEP*3.077),
    visualise_output=False
)

# Plot manually so we have access to ax
fig, ax = plt.subplots(figsize=(5, 6))
ms.visualise.visualise_topographical_correlation_map(domain_rotated, tcm, ax=ax)
tissue_shapes_rotated_gdf.boundary.plot(ax=ax, color=GEOMETRY_LINE_COLOUR, linewidth=0.5)
ax.set_title(f'{SAMPLE_ID_LABEL} \n Topological Correlation Map from \n {CELL_TYPE_A} to {CELL_TYPE_B} (r={TCM_RADIUS}um) \n Annotations overlaid ({GEOMETRY_LINE_COLOUR})')
plt.savefig(Path(f'{output_dir}/{SAMPLE_ID_LABEL}_06_TCM.pdf', bbox_inches='tight'))
print("TCM plotted...")

"""
Combined plotting
"""
fig, axes = plt.subplots(1, 6, figsize=(20, 6))
axes = axes.flatten()  # Makes indexing easier: axes[0], axes[1], etc.

# Plot 1: Cell centroid plot
ms.visualise.visualise(domain_rotated,
                       color_by='leiden_cell_type',
                       objects_to_plot=('collection', 'cells'),
                       ax=axes[0],
                       marker_size=0.01,
                       add_cbar=False,
                       scatter_kwargs={'alpha': 0.75})
axes[0].set_title('Cell Centroids \n')

# Plot 2: KDE Vascular Endothelium
kde = ms.distribution.kernel_density_estimation(
    domain_rotated,
    population=('leiden_cell_type', 'Vascular Endothelium'),
    visualise_output=True,
    visualise_heatmap_kwargs={"heatmap_cmap": "Reds", "ax": axes[1]}
)
tissue_shapes_rotated_gdf.boundary.plot(ax=axes[1], color='white', linewidth=1.5)
axes[1].set_title('KDE: Vascular Endothelium \n')

# Plot 3: Hexbin Vascular Endothelium
filtered_vasc = query(domain_rotated, ('label', 'leiden_cell_type'), 'is', 'Vascular Endothelium')
ms.visualise.visualise(domain_rotated, color_by=f'region counts: Vascular Endothelium',
                       objects_to_plot=('collection', f'Hexgrids {HEX_SIDE_LENGTH}'), ax=axes[2])
ms.visualise.visualise(domain_rotated, color_by='leiden_cell_type',
                       objects_to_plot=filtered_vasc, ax=axes[2],
                       marker_size=0.15, add_cbar=False, scatter_kwargs={'alpha': 0.5})
tissue_shapes_rotated_gdf.boundary.plot(ax=axes[2], color='white', linewidth=0.5)
axes[2].set_title('Hexbin: Vascular Endothelium \n')

# Plot 4: KDE PI16 Fibroblast
kde = ms.distribution.kernel_density_estimation(
    domain_rotated,
    population=('leiden_cell_type', 'PI16 Fibroblast'),
    visualise_output=True,
    visualise_heatmap_kwargs={"heatmap_cmap": "Blues", "ax": axes[3]}
)
tissue_shapes_rotated_gdf.boundary.plot(ax=axes[3], color='white', linewidth=1.5)
axes[3].set_title('KDE: PI16 Fibroblast \n')

# Plot 5: Hexbin PI16 Fibroblast
filtered_pi16 = query(domain_rotated, ('label', 'leiden_cell_type'), 'is', 'PI16 Fibroblast')
ms.visualise.visualise(domain_rotated, color_by=f'region counts: PI16 Fibroblast',
                       objects_to_plot=('collection', f'Hexgrids {HEX_SIDE_LENGTH}'), ax=axes[4])
ms.visualise.visualise(domain_rotated, color_by='leiden_cell_type',
                       objects_to_plot=filtered_pi16, ax=axes[4],
                       marker_size=0.15, add_cbar=False, scatter_kwargs={'alpha': 0.5})
tissue_shapes_rotated_gdf.boundary.plot(ax=axes[4], color='white', linewidth=0.5)
axes[4].set_title('Hexbin: PI16 Fibroblast \n')

# Plot 6: TCM
ms.visualise.visualise_topographical_correlation_map(domain_rotated, tcm, ax=axes[5])
tissue_shapes_rotated_gdf.boundary.plot(ax=axes[5], color='black', linewidth=0.5)
axes[5].set_title(f'Topographical Correlation Map (r={TCM_RADIUS}um) \n')

plt.tight_layout()
plt.savefig(Path(f'{output_dir}/{SAMPLE_ID_LABEL}_00_combined_plots.pdf', bbox_inches='tight'))
plt.show()

print(f"Sample {SAMPLE_ID_LABEL} completed")