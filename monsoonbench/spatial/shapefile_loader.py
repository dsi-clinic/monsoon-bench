import os
import geopandas as gpd
import geodatasets
import zipfile
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_DIR, exist_ok=True)

state_dir = os.path.join(DATA_DIR, "india_states_shapefile")
os.makedirs(state_dir, exist_ok=True)

country_dir = os.path.join(DATA_DIR, "india_country_shapefile")
os.makedirs(country_dir, exist_ok=True)

## Country Boundaries comes with GeoPandas
## Country Outline for India

# Load global country boundaries from Natural Earth
world = gpd.read_file(
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
)

# Filter for India
india_country = world[world["ADMIN"] == "India"]
print(f"Loaded India outline with {len(india_country)} feature(s)")

# Visualize 
india_country.plot(edgecolor="black", facecolor="lightblue")

# Load state-level boundaries
states = gpd.read_file(
    "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_1_states_provinces.zip"
)

india_states = states[states["admin"] == "India"]
print(f"Loaded {len(india_states)} state polygons for India")

india_states.plot(edgecolor="gray", facecolor="lightgreen")

## Saving to shapefile components
india_country.to_file(os.path.join(country_dir, "india_country.shp"))
india_states.to_file(os.path.join(state_dir, "india_states.shp"))

# Zip them together for country boundary
country_zip = os.path.join(DATA_DIR, "india_country_shapefile.zip")
with zipfile.ZipFile(country_zip, "w") as zf:
    for ext in [".shp", ".shx", ".dbf", ".prj"]:
        zf.write(
            os.path.join(country_dir, f"india_country{ext}"),
            arcname=f"india_country{ext}"
        )

print(f"Saved zipped shapefile to: {country_zip}")

# Zip them together for state boundary
state_zip = os.path.join(DATA_DIR, "india_states_shapefile.zip")
with zipfile.ZipFile(state_zip, "w") as zf:
    for ext in [".shp", ".shx", ".dbf", ".prj"]:
        zf.write(os.path.join(state_dir, f"india_states{ext}"), arcname=f"india_states{ext}")

print(f"Saved zipped shapefile to: {state_zip}")

""" 
def load_india_shapefile(path=state_zip):
    ### Load India shapefile (from local zip).
    gdf = gpd.read_file(f"zip://{path}")
    print(f"Loaded shapefile with {len(gdf)} features")
    return gdf

# Test
india = load_india_shapefile()
india.plot(edgecolor="black", facecolor="lightgreen")

"""