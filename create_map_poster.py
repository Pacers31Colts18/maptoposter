#!/usr/bin/env python3
"""
City Map Poster Generator

This module generates beautiful, minimalist map posters for any city in the world.
It fetches OpenStreetMap data using OSMnx, applies customizable themes, and creates
high-quality poster-ready images with roads, water features, and parks.
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import io
import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from lat_lon_parser import parse
from matplotlib.font_manager import FontProperties
from networkx import MultiDiGraph
from shapely.geometry import Point
from tqdm import tqdm

from font_management import load_fonts


class CacheError(Exception):
    """Raised when a cache operation fails."""


CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

FILE_ENCODING = "utf-8"

FONTS = load_fonts()


def _cache_path(key: str) -> str:
    """
    Generate a safe cache file path from a cache key.

    Args:
        key: Cache key identifier

    Returns:
        Path to cache file with .pkl extension
    """
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    """
    Retrieve a cached object by key.

    Args:
        key: Cache key identifier

    Returns:
        Cached object if found, None otherwise

    Raises:
        CacheError: If cache read operation fails
    """
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}") from e


def cache_set(key: str, value):
    """
    Store an object in the cache.

    Args:
        key: Cache key identifier
        value: Object to cache (must be picklable)

    Raises:
        CacheError: If cache write operation fails
    """
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}") from e


# Font loading now handled by font_management.py module


def is_latin_script(text):
    """
    Check if text is primarily Latin script.
    Used to determine if letter-spacing should be applied to city names.

    :param text: Text to analyze
    :return: True if text is primarily Latin script, False otherwise
    """
    if not text:
        return True

    latin_count = 0
    total_alpha = 0

    for char in text:
        if char.isalpha():
            total_alpha += 1
            # Latin Unicode ranges:
            # - Basic Latin: U+0000 to U+007F
            # - Latin-1 Supplement: U+0080 to U+00FF
            # - Latin Extended-A: U+0100 to U+017F
            # - Latin Extended-B: U+0180 to U+024F
            if ord(char) < 0x250:
                latin_count += 1

    # If no alphabetic characters, default to Latin (numbers, symbols, etc.)
    if total_alpha == 0:
        return True

    # Consider it Latin if >80% of alphabetic characters are Latin
    return (latin_count / total_alpha) > 0.8


def generate_output_filename(city, theme_name, output_format):
    """
    Generate unique output filename with city, theme, and datetime.
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    """
    Scans the themes directory and returns a list of available theme names.
    """
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []

    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith(".json"):
            theme_name = file[:-5]  # Remove .json extension
            themes.append(theme_name)
    return themes


def load_theme(theme_name="terracotta"):
    """
    Load theme from JSON file in themes directory.
    """
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")

    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default terracotta theme.")
        # Fallback to embedded terracotta theme
        return {
            "name": "Terracotta",
            "description": "Mediterranean warmth - burnt orange and clay tones on cream",
            "bg": "#F5EDE4",
            "text": "#8B4513",
            "gradient_color": "#F5EDE4",
            "water": "#A8C4C4",
            "parks": "#E8E0D0",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
        }

    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)
        print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
        if "description" in theme:
            print(f"  {theme['description']}")
        return theme


# Load theme (can be changed via command line or input)
THEME = dict[str, str]()  # Will be loaded later


def fetch_park_boundary(park_name):
    """
    Fetch the park boundary polygon directly from OSM by name using
    ox.geocode_to_gdf, which returns the actual admin/boundary relation
    as a proper polygon — not the piecemeal ways that features_from_point returns.
    Returns a GeoDataFrame in WGS84, or None on failure.
    """
    cache_key = f"boundary_{park_name.lower().replace(' ', '_')}"
    cached = cache_get(cache_key)
    if cached is not None:
        print("✓ Using cached park boundary")
        return cached

    queries = [
        park_name,
        f"{park_name} National Park",
        f"{park_name} State Park",
    ]
    for query in queries:
        try:
            gdf = ox.geocode_to_gdf(query)
            if gdf is not None and not gdf.empty:
                print(f"✓ Park boundary fetched for: {query}")
                try:
                    cache_set(cache_key, gdf)
                except CacheError:
                    pass
                return gdf
        except Exception:
            continue

    print("⚠ No park boundary found in OSM for this location")
    return None


def render_park_boundary(ax, boundary_gdf, graph_crs, outline_color, exterior_color):
    """
    Render the park boundary:
      - A semi-transparent dark overlay outside the park boundary
      - A dashed outline on the boundary edge
    """
    from shapely.geometry import box
    from shapely.ops import unary_union
    import geopandas as gpd

    # Always reproject to the same CRS as the graph — never re-derive a new UTM zone
    try:
        boundary_proj = boundary_gdf.to_crs(graph_crs)
    except Exception:
        boundary_proj = ox.projection.project_gdf(boundary_gdf)

    park_shape = unary_union(boundary_proj.geometry)

    # Snapshot limits before plotting — gdf.plot() auto-expands them
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    padding = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.5
    big_box = box(
        xlim[0] - padding, ylim[0] - padding,
        xlim[1] + padding, ylim[1] + padding,
    )

    # Exterior = big box minus the park shape
    exterior = big_box.difference(park_shape)
    exterior_gdf = gpd.GeoDataFrame(geometry=[exterior], crs=boundary_proj.crs)
    exterior_gdf.plot(
        ax=ax,
        facecolor=exterior_color,
        edgecolor='none',
        alpha=0.35,
        zorder=2.0,
    )

    # Boundary outline
    boundary_proj.plot(
        ax=ax,
        facecolor='none',
        edgecolor=outline_color,
        linewidth=1.5,
        linestyle='--',
        zorder=2.1,
    )

    # Restore limits — prevent the large exterior polygon from zooming out
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def fetch_dem(point, dist):
    """
    Fetch a Digital Elevation Model from the USGS 3DEP service (US only).
    Returns (dem_array, transform, crs) or (None, None, None) on failure.
    Uses the same cache system as other features.
    """
    try:
        import rasterio
        import rasterio.transform
    except ImportError:
        print("⚠ rasterio not installed — skipping topography. Run: pip install rasterio")
        return None, None, None

    lat, lon = point
    cache_key = f"dem_{lat:.4f}_{lon:.4f}_{dist}"
    cached = cache_get(cache_key)
    if cached is not None:
        print("✓ Using cached elevation data")
        return cached

    # Build WGS84 bounding box from center + distance (in meters)
    lat_delta = dist / 111000
    lon_delta = dist / (111000 * math.cos(math.radians(lat)))
    west, east = lon - lon_delta, lon + lon_delta
    south, north = lat - lat_delta, lat + lat_delta

    url = (
        "https://elevation.nationalmap.gov/arcgis/rest/services/"
        "3DEPElevation/ImageServer/exportImage"
    )
    params = {
        "bbox": f"{west},{south},{east},{north}",
        "bboxSR": "4326",
        "size": "512,512",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "f": "image",
    }

    try:
        import requests as _requests
        print("Downloading elevation data from USGS 3DEP...")
        resp = _requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        with rasterio.open(io.BytesIO(resp.content)) as src:
            dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
        result = (dem, transform, crs)
        try:
            cache_set(cache_key, result)
        except CacheError:
            pass
        print("✓ Elevation data downloaded")
        return result
    except Exception as e:
        print(f"⚠ Elevation fetch failed: {e}")
        return None, None, None


def compute_hillshade(dem, azimuth=315, altitude=45):
    """
    Compute a hillshade array from a DEM using numpy.
    azimuth: sun direction in degrees (315 = NW, classic cartographic default)
    altitude: sun angle above horizon in degrees
    """
    dem = np.where(dem < -1000, np.nan, dem.astype(np.float64))
    dem_filled = np.where(np.isnan(dem), 0.0, dem)
    dy, dx = np.gradient(dem_filled)
    azimuth_rad = np.radians(360.0 - azimuth)
    altitude_rad = np.radians(altitude)
    slope = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
    aspect = np.arctan2(-dy, dx)
    hs = (
        np.sin(altitude_rad) * np.cos(slope)
        + np.cos(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect)
    )
    return np.clip(hs, 0.0, 1.0)


def render_topo(ax, dem, dem_transform, dem_crs, graph_crs, contour_interval=200, hillshade_alpha=0.55):
    """
    Reproject DEM to the graph's projected CRS, then render:
      - hillshade as a semi-transparent grayscale layer (z=0.2)
      - contour lines at the requested interval in meters (z=0.3)
    """
    try:
        import rasterio.warp
        from rasterio.crs import CRS as RasterioCRS
    except ImportError:
        return

    h, w = dem.shape

    # Build source bounds from affine transform
    left = dem_transform.c
    top = dem_transform.f
    right = left + dem_transform.a * w
    bottom = top + dem_transform.e * h  # e is negative for north-up rasters

    target_crs = RasterioCRS.from_user_input(graph_crs)

    dst_transform, dst_w, dst_h = rasterio.warp.calculate_default_transform(
        dem_crs, target_crs, w, h,
        left=left, bottom=bottom, right=right, top=top,
    )

    dem_proj = np.zeros((dst_h, dst_w), dtype=np.float32)
    rasterio.warp.reproject(
        source=dem,
        destination=dem_proj,
        src_transform=dem_transform,
        src_crs=dem_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=rasterio.warp.Resampling.bilinear,
    )

    # Projected extent (x_left, x_right, y_bottom, y_top)
    proj_left = dst_transform.c
    proj_top = dst_transform.f
    proj_right = proj_left + dst_transform.a * dst_w
    proj_bottom = proj_top + dst_transform.e * dst_h
    extent = [proj_left, proj_right, proj_bottom, proj_top]

    # --- Hillshade ---
    hs = compute_hillshade(dem_proj)
    ax.imshow(
        hs,
        extent=extent,
        cmap="gray",
        alpha=hillshade_alpha,
        zorder=0.2,
        origin="upper",
        aspect="auto",
        interpolation="bilinear",
    )

    # --- Contour lines ---
    valid = dem_proj[dem_proj > -1000]
    if valid.size == 0:
        return
    min_ele = np.floor(valid.min() / contour_interval) * contour_interval
    max_ele = np.ceil(valid.max() / contour_interval) * contour_interval
    levels = np.arange(min_ele, max_ele + contour_interval, contour_interval)
    if len(levels) < 2:
        return

    xs = np.linspace(proj_left, proj_right, dst_w)
    ys = np.linspace(proj_top, proj_bottom, dst_h)
    X, Y = np.meshgrid(xs, ys)
    dem_masked = np.where(dem_proj < -1000, np.nan, dem_proj)

    ax.contour(
        X, Y, dem_masked,
        levels=levels,
        colors=THEME.get("contours", "#8B7355"),
        linewidths=0.5,
        zorder=0.9,
        alpha=0.7,
    )


def create_gradient_fade(ax, color, location="bottom", zorder=10):
    """
    Creates a fade effect at the top or bottom of the map.
    """
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]

    if location == "bottom":
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def filter_major_pois(gdf, poi_type, max_peaks=15):
    """
    Filter a POI GeoDataFrame to only 'major' features to reduce clutter.

    - peaks: requires a name, sorts by elevation (ele tag), keeps top max_peaks
    - all others: requires a name
    """
    if gdf is None or gdf.empty:
        return gdf

    points = gdf[gdf.geometry.type == "Point"].copy()
    if points.empty:
        return points

    # All types: require a name
    if 'name' in points.columns:
        points = points[points['name'].apply(lambda n: isinstance(n, str) and bool(n.strip()))]

    if poi_type == 'peaks' and not points.empty:
        if 'ele' in points.columns:
            def _to_float(v):
                try:
                    return float(str(v).split(';')[0].strip())
                except (ValueError, TypeError):
                    return None
            points = points.copy()
            points['_ele_num'] = points['ele'].apply(_to_float)
            # Prefer peaks with elevation data; fall back to any named peak
            with_ele = points[points['_ele_num'].notna()].nlargest(max_peaks, '_ele_num')
            without_ele = points[points['_ele_num'].isna()].head(max(0, max_peaks - len(with_ele)))
            import pandas as pd
            points = pd.concat([with_ele, without_ele]).drop(columns=['_ele_num'])
        else:
            points = points.head(max_peaks)

    return points


def render_poi_markers(ax, gdf, graph_crs, color, marker, size, zorder, label=False, font_size=5):
    """
    Render point-of-interest markers on the projected map axes.
    Projects the GDF to the graph CRS before extracting coordinates.
    If label=True, annotates each point with its OSM name where available.
    """
    if gdf is None or gdf.empty:
        return
    points = gdf[gdf.geometry.type == "Point"].copy()
    if points.empty:
        return
    try:
        points = ox.projection.project_gdf(points)
    except Exception:
        points = points.to_crs(graph_crs)
    xs = [geom.x for geom in points.geometry]
    ys = [geom.y for geom in points.geometry]
    ax.scatter(xs, ys, c=color, marker=marker, s=size, zorder=zorder, linewidths=0)
    if label and 'name' in points.columns:
        for x, y, name in zip(xs, ys, points['name']):
            if isinstance(name, str) and name.strip():
                ax.annotate(
                    name,
                    xy=(x, y),
                    xytext=(4, 4),
                    textcoords='offset points',
                    color=THEME.get('text', '#2D3B2E'),
                    fontsize=font_size,
                    zorder=zorder + 0.1,
                    ha='left',
                    va='bottom',
                )


def get_edge_colors_by_type(g):
    """
    Assigns colors to edges based on road type hierarchy.
    Returns a list of colors corresponding to each edge in the graph.
    """
    edge_colors = []

    for _u, _v, data in g.edges(data=True):
        # Get the highway type (can be a list or string)
        highway = data.get('highway', 'unclassified')

        # Handle list of highway types (take the first one)
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        # Assign color based on road type
        if highway in ["motorway", "motorway_link"]:
            color = THEME["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            color = THEME["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            color = THEME["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            color = THEME["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            color = THEME["road_residential"]
        else:
            color = THEME['road_default']

        edge_colors.append(color)

    return edge_colors


def get_edge_widths_by_type(g):
    """
    Assigns line widths to edges based on road type.
    Major roads get thicker lines.
    """
    edge_widths = []

    for _u, _v, data in g.edges(data=True):
        highway = data.get('highway', 'unclassified')

        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        # Assign width based on road importance
        if highway in ["motorway", "motorway_link"]:
            width = 1.2
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            width = 1.0
        elif highway in ["secondary", "secondary_link"]:
            width = 0.8
        elif highway in ["tertiary", "tertiary_link"]:
            width = 0.6
        else:
            width = 0.4

        edge_widths.append(width)

    return edge_widths


def get_coordinates(city, country):
    """
    Fetches coordinates for a given city and country using geopy.
    Includes rate limiting to be respectful to the geocoding service.
    """
    coords = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords)
    if cached:
        print(f"✓ Using cached coordinates for {city}, {country}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)

    # Add a small delay to respect Nominatim's usage policy
    time.sleep(1)

    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}") from e

    # If geocode returned a coroutine in some environments, run it to get the result.
    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError as exc:
            # If an event loop is already running, try using it to complete the coroutine.
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Running event loop in the same thread; raise a clear error.
                raise RuntimeError(
                    "Geocoder returned a coroutine while an event loop is already running. "
                    "Run this script in a synchronous environment."
                ) from exc
            location = loop.run_until_complete(location)

    if location:
        # Use getattr to safely access address (helps static analyzers)
        addr = getattr(location, "address", None)
        if addr:
            print(f"✓ Found: {addr}")
        else:
            print("✓ Found location (address not available)")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)

    raise ValueError(f"Could not find coordinates for {city}, {country}")


def get_park_info(park_name):
    """
    Geocode a national or state park by name using Nominatim.
    Returns (lat, lon, distance_meters, display_name) where distance is
    derived from the OSM bounding box so the whole park fits in the frame.
    Caps at 60000m to keep data fetching manageable.
    """
    cache_key = f"park_{park_name.lower().replace(' ', '_')}"
    cached = cache_get(cache_key)
    if cached is not None:
        print(f"✓ Using cached park info for {park_name}")
        return cached

    print(f"Looking up park: {park_name}...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    time.sleep(1)

    # Try progressively broader queries until we get a boundary result
    queries = [
        park_name,
        f"{park_name} National Park",
        f"{park_name} State Park",
    ]
    location = None
    for query in queries:
        try:
            location = geolocator.geocode(query, exactly_one=True)
            if location:
                break
        except Exception as e:
            raise ValueError(f"Geocoding failed for '{park_name}': {e}") from e

    if location is None:
        raise ValueError(
            f"Could not find '{park_name}'. Try adding 'National Park' or 'State Park' to the name."
        )

    print(f"✓ Found: {location.address}")

    # Derive distance and true center from the Nominatim bounding box.
    # raw['boundingbox'] = [south, north, west, east] as strings.
    # Nominatim's lat/lon is the OSM relation centroid, which is often off-center
    # for irregular park shapes — use the bbox midpoint instead.
    bbox = location.raw.get("boundingbox")
    if bbox:
        south, north, west, east = [float(v) for v in bbox]
        lat = (south + north) / 2
        lon = (west + east) / 2
        lat_span_m = (north - south) * 111000
        lon_span_m = (east - west) * 111000 * math.cos(math.radians(lat))
        # Use the larger dimension as the radius, add 15% padding, cap at 60km
        dist = int(min(max(lat_span_m, lon_span_m) / 2 * 1.15, 60000))
    else:
        lat, lon = location.latitude, location.longitude
        dist = 20000  # fallback

    print(f"✓ Center: {lat:.4f}, {lon:.4f}")
    print(f"✓ Auto-distance: {dist}m")

    # Clean up display name: strip country/state suffix, keep the park name
    display_name = location.address.split(",")[0].strip()

    result = (lat, lon, dist, display_name)
    try:
        cache_set(cache_key, result)
    except CacheError:
        pass
    return result


def get_crop_limits(g_proj, center_lat_lon, fig, dist):
    """
    Crop inward to preserve aspect ratio while guaranteeing
    full coverage of the requested radius.
    """
    lat, lon = center_lat_lon

    # Project center point into graph CRS
    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=g_proj.graph["crs"]
        )[0]
    )
    center_x, center_y = center.x, center.y

    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height

    # Start from the *requested* radius
    half_x = dist
    half_y = dist

    # Cut inward to match aspect
    if aspect > 1:  # landscape → reduce height
        half_y = half_x / aspect
    else:  # portrait → reduce width
        half_x = half_y * aspect

    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


def fetch_graph(point, dist) -> MultiDiGraph | None:
    """
    Fetch street network graph from OpenStreetMap.

    Uses caching to avoid redundant downloads. Fetches all network types
    within the specified distance from the center point.

    Args:
        point: (latitude, longitude) tuple for center point
        dist: Distance in meters from center point

    Returns:
        MultiDiGraph of street network, or None if fetch fails
    """
    lat, lon = point
    graph = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph)
    if cached is not None:
        print("✓ Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        g = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all', truncate_by_edge=True)
        # Rate limit between requests
        time.sleep(0.5)
        try:
            cache_set(graph, g)
        except CacheError as e:
            print(e)
        return g
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None


def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    """
    Fetch geographic features (water, parks, etc.) from OpenStreetMap.

    Uses caching to avoid redundant downloads. Fetches features matching
    the specified OSM tags within distance from center point.

    Args:
        point: (latitude, longitude) tuple for center point
        dist: Distance in meters from center point
        tags: Dictionary of OSM tags to filter features
        name: Name for this feature type (for caching and logging)

    Returns:
        GeoDataFrame of features, or None if fetch fails
    """
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features)
    if cached is not None:
        print(f"✓ Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        # Rate limit between requests
        time.sleep(0.3)
        try:
            cache_set(features, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def create_poster(
    city,
    country,
    point,
    dist,
    output_file,
    output_format,
    width=12,
    height=16,
    country_label=None,
    name_label=None,
    display_city=None,
    display_country=None,
    fonts=None,
    park_mode=False,
    park_labels=False,
    max_peaks=15,
    topo_mode=False,
    contour_interval=200,
    show_boundary=False,
    park_name=None,
):
    """
    Generate a complete map poster with roads, water, parks, and typography.

    Creates a high-quality poster by fetching OSM data, rendering map layers,
    applying the current theme, and adding text labels with coordinates.

    Args:
        city: City name for display on poster
        country: Country name for display on poster
        point: (latitude, longitude) tuple for map center
        dist: Map radius in meters
        output_file: Path where poster will be saved
        output_format: File format ('png', 'svg', or 'pdf')
        width: Poster width in inches (default: 12)
        height: Poster height in inches (default: 16)
        country_label: Optional override for country text on poster
        _name_label: Optional override for city name (unused, reserved for future use)

    Raises:
        RuntimeError: If street network data cannot be retrieved
    """
    # Handle display names for i18n support
    # Priority: display_city/display_country > name_label/country_label > city/country
    display_city = display_city or name_label or city
    display_country = display_country or country_label or country

    print(f"\nGenerating map for {city}, {country}...")

    # Progress bar for data fetching
    trails = peaks = viewpoints = campgrounds = trailheads = None
    dem = dem_transform = dem_crs = None
    park_boundary = None
    total_steps = (8 if park_mode else 3) + (1 if topo_mode else 0) + (1 if show_boundary else 0)
    with tqdm(
        total=total_steps,
        desc="Fetching map data",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ) as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        # fetch_dist: square bbox radius used for all data downloads.
        # Must cover the full visible crop area (half_y = dist for portrait,
        # half_x = dist for landscape) plus 15% buffer for edge roads.
        fetch_dist = int(dist * 1.15)
        g = fetch_graph(point, fetch_dist)
        if g is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)

        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        water = fetch_features(
            point,
            fetch_dist,
            tags={"natural": ["water", "bay", "strait"], "waterway": "riverbank"},
            name="water",
        )
        pbar.update(1)

        # 3. Fetch Green Spaces (expanded tags in park mode)
        pbar.set_description("Downloading parks/green spaces")
        if park_mode:
            # Exclude leisure=nature_reserve and boundary=national_park — these
            # return a single polygon covering the entire park including lakes,
            # which buries water features. Stick to landuse/natural sub-polygons only.
            green_tags = {
                "landuse": ["forest", "meadow", "grass", "recreation_ground"],
                "natural": ["wood", "scrub", "heath", "grassland"],
            }
        else:
            green_tags = {"leisure": "park", "landuse": "grass"}
        parks = fetch_features(point, fetch_dist, tags=green_tags, name="parks")
        pbar.update(1)

        if park_mode:
            # 4. Fetch Trails
            pbar.set_description("Downloading trails")
            trails = fetch_features(
                point,
                fetch_dist,
                tags={"highway": ["path", "track", "bridleway"]},
                name="trails",
            )
            pbar.update(1)

            # 5. Fetch Peaks / Summits
            pbar.set_description("Downloading peaks")
            peaks = fetch_features(
                point,
                fetch_dist,
                tags={"natural": ["peak", "saddle"]},
                name="peaks",
            )
            pbar.update(1)

            # 6. Fetch Viewpoints
            pbar.set_description("Downloading viewpoints")
            viewpoints = fetch_features(
                point,
                fetch_dist,
                tags={"tourism": "viewpoint"},
                name="viewpoints",
            )
            pbar.update(1)

            # 7. Fetch Campgrounds
            pbar.set_description("Downloading campgrounds")
            campgrounds = fetch_features(
                point,
                fetch_dist,
                tags={"tourism": "camp_site"},
                name="campgrounds",
            )
            pbar.update(1)

            # 8. Fetch Trailheads
            pbar.set_description("Downloading trailheads")
            trailheads = fetch_features(
                point,
                fetch_dist,
                tags={"highway": "trailhead"},
                name="trailheads",
            )
            pbar.update(1)

        if topo_mode:
            pbar.set_description("Downloading elevation data")
            dem, dem_transform, dem_crs = fetch_dem(point, fetch_dist)
            pbar.update(1)

        if show_boundary:
            pbar.set_description("Downloading park boundary")
            park_boundary = fetch_park_boundary(park_name or city)
            pbar.update(1)

    print("✓ All data retrieved successfully!")

    # 2. Setup Plot
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    # Project graph to a metric CRS so distances and aspect are linear (meters)
    g_proj = ox.project_graph(g)

    # 3. Plot Layers
    # Layer: Hillshade + Contours (topo mode only, rendered first so everything sits on top)
    if topo_mode and dem is not None:
        render_topo(
            ax, dem, dem_transform, dem_crs,
            g_proj.graph['crs'],
            contour_interval=contour_interval,
        )
    # Layer 1: Polygons (filter to only plot polygon/multipolygon geometries, not points)
    if water is not None and not water.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        water_polys = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not water_polys.empty:
            # Project water features in the same CRS as the graph
            try:
                water_polys = ox.projection.project_gdf(water_polys)
            except Exception:
                water_polys = water_polys.to_crs(g_proj.graph['crs'])
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1.2)

    if parks is not None and not parks.empty:
        # Filter to only polygon/multipolygon geometries to avoid point features showing as dots
        parks_polys = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not parks_polys.empty:
            # Project park features in the same CRS as the graph
            try:
                parks_polys = ox.projection.project_gdf(parks_polys)
            except Exception:
                parks_polys = parks_polys.to_crs(g_proj.graph['crs'])
            parks_alpha = 0.45 if topo_mode else 1.0
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=0.8, alpha=parks_alpha)

    # Layer: Trails (park mode only)
    if park_mode and trails is not None and not trails.empty:
        trail_lines = trails[trails.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        if not trail_lines.empty:
            try:
                trail_lines = ox.projection.project_gdf(trail_lines)
            except Exception:
                trail_lines = trail_lines.to_crs(g_proj.graph['crs'])
            trail_lines.plot(
                ax=ax,
                color=THEME.get('trails', '#8B5E3C'),
                linewidth=0.35,
                zorder=0.6,
            )

    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(g_proj)
    edge_widths = get_edge_widths_by_type(g_proj)

    # Determine cropping limits to maintain the poster aspect ratio
    crop_xlim, crop_ylim = get_crop_limits(g_proj, point, fig, dist)
    # Plot the projected graph and then apply the cropped limits
    ox.plot_graph(
        g_proj, ax=ax, bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False,
        close=False,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)

    # Layer: Park boundary outline + exterior shading
    if show_boundary and park_boundary is not None:
        render_park_boundary(
            ax, park_boundary, g_proj.graph['crs'],
            outline_color=THEME.get('boundary', '#2D3B2E'),
            exterior_color=THEME.get('bg', '#EDE8DC'),
        )
    elif show_boundary:
        print("⚠ No park boundary found in OSM for this location")

    # Layer 3: Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

    # Layer: POI Markers (park mode only)
    if park_mode:
        graph_crs = g_proj.graph['crs']
        render_poi_markers(ax, filter_major_pois(peaks, 'peaks', max_peaks), graph_crs, THEME.get('peaks', '#4A3728'), '^', 30, 1.5, label=park_labels, font_size=5)
        render_poi_markers(ax, filter_major_pois(viewpoints, 'viewpoints'), graph_crs, THEME.get('viewpoints', '#B8860B'), 'o', 20, 1.5, label=park_labels, font_size=5)
        render_poi_markers(ax, filter_major_pois(campgrounds, 'campgrounds'), graph_crs, THEME.get('campgrounds', '#8B2020'), 'v', 20, 1.5, label=park_labels, font_size=5)
        render_poi_markers(ax, filter_major_pois(trailheads, 'trailheads'), graph_crs, THEME.get('trailheads', '#5C3D1E'), 'D', 18, 1.5, label=park_labels, font_size=5)

    # Calculate scale factor based on smaller dimension (reference 12 inches)
    # This ensures text scales properly for both portrait and landscape orientations
    scale_factor = min(height, width) / 12.0

    # Base font sizes (at 12 inches width)
    base_main = 60
    base_sub = 22
    base_coords = 14
    base_attr = 8

    # 4. Typography - use custom fonts if provided, otherwise use default FONTS
    active_fonts = fonts or FONTS
    if active_fonts:
        # font_main is calculated dynamically later based on length
        font_sub = FontProperties(
            fname=active_fonts["light"], size=base_sub * scale_factor
        )
        font_coords = FontProperties(
            fname=active_fonts["regular"], size=base_coords * scale_factor
        )
        font_attr = FontProperties(
            fname=active_fonts["light"], size=base_attr * scale_factor
        )
    else:
        # Fallback to system fonts
        font_sub = FontProperties(
            family="monospace", weight="normal", size=base_sub * scale_factor
        )
        font_coords = FontProperties(
            family="monospace", size=base_coords * scale_factor
        )
        font_attr = FontProperties(family="monospace", size=base_attr * scale_factor)

    # Format city name based on script type
    # Latin scripts: apply uppercase and letter spacing for aesthetic
    # Non-Latin scripts (CJK, Thai, Arabic, etc.): no spacing, preserve case structure
    if is_latin_script(display_city):
        # Latin script: uppercase with letter spacing (e.g., "P  A  R  I  S")
        spaced_city = "  ".join(list(display_city.upper()))
    else:
        # Non-Latin script: no spacing, no forced uppercase
        # For scripts like Arabic, Thai, Japanese, etc.
        spaced_city = display_city

    # Dynamically adjust font size based on city name length to prevent truncation
    # We use the already scaled "main" font size as the starting point.
    base_adjusted_main = base_main * scale_factor
    city_char_count = len(display_city)

    # Heuristic: If length is > 10, start reducing.
    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if active_fonts:
        font_main_adjusted = FontProperties(
            fname=active_fonts["bold"], size=adjusted_font_size
        )
    else:
        font_main_adjusted = FontProperties(
            family="monospace", weight="bold", size=adjusted_font_size
        )

    # --- BOTTOM TEXT ---
    ax.text(
        0.5,
        0.14,
        spaced_city,
        transform=ax.transAxes,
        color=THEME["text"],
        ha="center",
        fontproperties=font_main_adjusted,
        zorder=11,
    )

    ax.text(
        0.5,
        0.10,
        display_country.upper(),
        transform=ax.transAxes,
        color=THEME["text"],
        ha="center",
        fontproperties=font_sub,
        zorder=11,
    )

    lat, lon = point
    coords = (
        f"{lat:.4f}° N / {lon:.4f}° E"
        if lat >= 0
        else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    )
    if lon < 0:
        coords = coords.replace("E", "W")

    ax.text(
        0.5,
        0.07,
        coords,
        transform=ax.transAxes,
        color=THEME["text"],
        alpha=0.7,
        ha="center",
        fontproperties=font_coords,
        zorder=11,
    )

    ax.plot(
        [0.4, 0.6],
        [0.125, 0.125],
        transform=ax.transAxes,
        color=THEME["text"],
        linewidth=1 * scale_factor,
        zorder=11,
    )

    # --- ATTRIBUTION (bottom right) ---
    if FONTS:
        font_attr = FontProperties(fname=FONTS["light"], size=8)
    else:
        font_attr = FontProperties(family="monospace", size=8)

    ax.text(
        0.98,
        0.02,
        "© OpenStreetMap contributors",
        transform=ax.transAxes,
        color=THEME["text"],
        alpha=0.5,
        ha="right",
        va="bottom",
        fontproperties=font_attr,
        zorder=11,
    )

    # 5. Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(
        facecolor=THEME["bg"],
        bbox_inches="tight",
        pad_inches=0.05,
    )

    # DPI matters mainly for raster formats
    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)

    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid

  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline

  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads

  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout

  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula

  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split

  # List themes
  python create_map_poster.py --list-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --country-label   Override country text displayed on poster
  --theme, -t       Theme name (default: terracotta)
  --all-themes      Generate posters for all themes
  --distance, -d    Map radius in meters (default: 18000)
  --list-themes     List all available themes

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")


def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return

    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r", encoding=FILE_ENCODING) as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except (OSError, json.JSONDecodeError):
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city "New York" --country "USA" -l 40.776676 -73.971321 --theme neon_cyberpunk
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
        """,
    )

    parser.add_argument(
        "--park", "-p", type=str,
        help='National or state park name (e.g. "Yosemite", "Rocky Mountain"). '
             'Auto-sets distance, park-mode, park-labels, topo-mode, and show-boundary.',
    )
    parser.add_argument("--city", "-c", type=str, help="City name")
    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument(
        "--latitude",
        "-lat",
        dest="latitude",
        type=str,
        help="Override latitude center point",
    )
    parser.add_argument(
        "--longitude",
        "-long",
        dest="longitude",
        type=str,
        help="Override longitude center point",
    )
    parser.add_argument(
        "--country-label",
        dest="country_label",
        type=str,
        help="Override country text displayed on poster",
    )
    parser.add_argument(
        "--theme",
        "-t",
        type=str,
        default="terracotta",
        help="Theme name (default: terracotta)",
    )
    parser.add_argument(
        "--all-themes",
        "--All-themes",
        dest="all_themes",
        action="store_true",
        help="Generate posters for all themes",
    )
    parser.add_argument(
        "--distance",
        "-d",
        type=int,
        default=None,
        help="Map radius in meters (default: 18000, or auto-calculated with --park)",
    )
    parser.add_argument(
        "--width",
        "-W",
        type=float,
        default=12,
        help="Image width in inches (default: 12, max: 20 )",
    )
    parser.add_argument(
        "--height",
        "-H",
        type=float,
        default=16,
        help="Image height in inches (default: 16, max: 20)",
    )
    parser.add_argument(
        "--list-themes", action="store_true", help="List all available themes"
    )
    parser.add_argument(
        "--display-city",
        "-dc",
        type=str,
        help="Custom display name for city (for i18n support)",
    )
    parser.add_argument(
        "--display-country",
        "-dC",
        type=str,
        help="Custom display name for country (for i18n support)",
    )
    parser.add_argument(
        "--font-family",
        type=str,
        help='Google Fonts family name (e.g., "Noto Sans JP", "Open Sans"). If not specified, uses local Roboto fonts.',
    )
    parser.add_argument(
        "--format",
        "-f",
        default="png",
        choices=["png", "svg", "pdf"],
        help="Output format for the poster (default: png)",
    )
    parser.add_argument(
        "--park-mode",
        dest="park_mode",
        action="store_true",
        help="Enable national/state park mode: fetches trails, peaks, viewpoints, campgrounds, trailheads, and expanded green spaces",
    )
    parser.add_argument(
        "--park-labels",
        dest="park_labels",
        action="store_true",
        help="Label peaks, viewpoints, campgrounds, and trailheads with their names (requires --park-mode)",
    )
    parser.add_argument(
        "--max-peaks",
        dest="max_peaks",
        type=int,
        default=15,
        help="Maximum number of peaks to show, ranked by elevation (default: 15, requires --park-mode)",
    )
    parser.add_argument(
        "--topo-mode",
        dest="topo_mode",
        action="store_true",
        help="Add hillshade and contour lines from USGS 3DEP elevation data (US parks only, requires rasterio)",
    )
    parser.add_argument(
        "--contour-interval",
        dest="contour_interval",
        type=int,
        default=200,
        help="Contour line interval in meters (default: 200, requires --topo-mode)",
    )
    parser.add_argument(
        "--show-boundary",
        dest="show_boundary",
        action="store_true",
        help="Outline the park boundary and shade the surrounding area to make the park stand out",
    )

    args = parser.parse_args()

    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)

    # Validate required arguments
    if not args.park and (not args.city or not args.country):
        print("Error: either --park or both --city and --country are required.\n")
        print_examples()
        sys.exit(1)

    # Enforce maximum dimensions
    if args.width > 20:
        print(
            f"⚠ Width {args.width} exceeds the maximum allowed limit of 20. It's enforced as max limit 20."
        )
        args.width = 20.0
    if args.height > 20:
        print(
            f"⚠ Height {args.height} exceeds the maximum allowed limit of 20. It's enforced as max limit 20."
        )
        args.height = 20.0

    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            sys.exit(1)
        themes_to_generate = [args.theme]

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    # Load custom fonts if specified
    custom_fonts = None
    if args.font_family:
        custom_fonts = load_fonts(args.font_family)
        if not custom_fonts:
            print(f"⚠ Failed to load '{args.font_family}', falling back to Roboto")

    # Get coordinates and generate poster
    try:
        if args.park:
            park_lat, park_lon, auto_dist, park_display_name = get_park_info(args.park)
            coords = (park_lat, park_lon)
            # Use auto-calculated distance unless user explicitly passed --distance
            if args.distance is None:
                args.distance = auto_dist
            args.park_mode = True
            args.park_labels = True
            args.topo_mode = True
            args.show_boundary = True
            # Default display name from OSM unless user provided one
            if not args.display_city:
                args.display_city = park_display_name
            if not args.display_country:
                args.display_country = "NATIONAL PARK"
            if not args.city:
                args.city = args.park
            if not args.country:
                args.country = "USA"
            # Default to national_park theme if user hasn't chosen one
            if args.theme == "terracotta":
                args.theme = "national_park"
                themes_to_generate = ["national_park"]
        elif args.latitude and args.longitude:
            lat = parse(args.latitude)
            lon = parse(args.longitude)
            coords = (lat, lon)
            print(f"✓ Coordinates: {', '.join([str(i) for i in coords])}")
        else:
            coords = get_coordinates(args.city, args.country)

        if args.distance is None:
            args.distance = 18000

        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)
            create_poster(
                args.city,
                args.country,
                coords,
                args.distance,
                output_file,
                args.format,
                args.width,
                args.height,
                country_label=args.country_label,
                display_city=args.display_city,
                display_country=args.display_country,
                fonts=custom_fonts,
                park_mode=args.park_mode,
                park_labels=args.park_labels,
                max_peaks=args.max_peaks,
                topo_mode=args.topo_mode,
                contour_interval=args.contour_interval,
                show_boundary=args.show_boundary,
                park_name=args.park,
            )

        print("\n" + "=" * 50)
        print("✓ Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
