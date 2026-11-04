"""Export a real Mumbai feature sample from Google Earth Engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ee
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from configs.config import (  # noqa: E402
	EQUITY_COLUMNS,
	FEATURE_COLUMNS,
	MUMBAI_BOUNDS,
	PROCESSED_DATA_DIR,
	TARGET_COLUMN,
)

EE_PROJECT = "proud-lamp-453406-n1"
DEFAULT_START = "2024-01-01"
DEFAULT_END = "2024-02-01"
DEFAULT_SCALE = 30
DEFAULT_NUM_PIXELS = 10000


def initialize_earth_engine(project: str = EE_PROJECT) -> None:
	"""Initialize Earth Engine using the configured Cloud project."""
	try:
		ee.Initialize(project=project)
	except Exception as error:
		raise RuntimeError(
			"Earth Engine is not initialized. Run "
			"'.venv\\Scripts\\earthengine.exe authenticate --auth_mode=localhost' "
			f"and configure project '{project}'."
		) from error


def mumbai_roi() -> ee.Geometry:
	return ee.Geometry.Rectangle(
		[
			MUMBAI_BOUNDS["min_lon"],
			MUMBAI_BOUNDS["min_lat"],
			MUMBAI_BOUNDS["max_lon"],
			MUMBAI_BOUNDS["max_lat"],
		],
		geodesic=False,
	)


def _mask_landsat(image: ee.Image) -> ee.Image:
	qa = image.select("QA_PIXEL")
	clear = (
		qa.bitwiseAnd(1 << 1).eq(0)
		.And(qa.bitwiseAnd(1 << 3).eq(0))
		.And(qa.bitwiseAnd(1 << 4).eq(0))
		.And(qa.bitwiseAnd(1 << 5).eq(0))
	)
	return image.updateMask(clear).updateMask(image.select("ST_B10").gt(0))


def _mask_sentinel(image: ee.Image) -> ee.Image:
	scl = image.select("SCL")
	clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
	return image.updateMask(clear)


def _mask_modis_night(image: ee.Image) -> ee.Image:
	quality = image.select("QC_Night").bitwiseAnd(3).lte(1)
	return image.updateMask(quality).updateMask(image.select("LST_Night_1km").gt(0))


def _required_collection(collection: ee.ImageCollection, name: str) -> ee.ImageCollection:
	if collection.size().getInfo() == 0:
		raise RuntimeError(f"No usable Earth Engine images found for {name}.")
	return collection


def build_feature_image(start: str = DEFAULT_START, end: str = DEFAULT_END) -> ee.Image:
	"""Build a sampled, co-registered feature image for the requested period."""
	roi = mumbai_roi()
	landsat = _required_collection(
		ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
		.merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
		.filterBounds(roi)
		.filterDate(start, end)
		.filter(ee.Filter.eq("PROCESSING_LEVEL", "L2SP"))
		.map(_mask_landsat),
		"Landsat 8/9 L2SP",
	).median()
	sentinel = _required_collection(
		ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
		.filterBounds(roi)
		.filterDate(start, end)
		.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
		.map(_mask_sentinel),
		"Sentinel-2 SR",
	).median()

	s2 = sentinel.select(["B2", "B3", "B4", "B8", "B11", "B12"]).multiply(0.0001)
	ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")
	ndbi = s2.normalizedDifference(["B11", "B8"]).rename("ndbi")
	ndwi = s2.normalizedDifference(["B3", "B8"]).rename("ndwi")
	mndwi = s2.normalizedDifference(["B3", "B11"]).rename("mndwi")

	worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
	water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gt(50)
	era5 = _required_collection(
		ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
		.filterBounds(roi)
		.filterDate(start, end),
		"ERA5-Land",
	).median()
	worldpop = _required_collection(
		ee.ImageCollection("WorldPop/GP/100m/pop")
		.filterBounds(roi)
		.filterDate("2020-01-01", "2024-12-31"),
		"WorldPop",
	).select("population").mosaic()
	buildings = _required_collection(
		ee.ImageCollection("GOOGLE/Research/open-buildings-temporal/v1")
		.filterBounds(roi)
		.sort("system:time_start", False),
		"Open Buildings Temporal",
	).first()
	albedo = (
		_required_collection(
			ee.ImageCollection("MODIS/061/MCD43A3")
			.filterBounds(roi)
			.filterDate(start, end),
			"MODIS albedo",
		)
		.median()
		.select("Albedo_WSA_shortwave")
		.multiply(0.001)
		.rename("albedo")
	)
	night_lst_c = (
		_required_collection(
			ee.ImageCollection("MODIS/061/MYD11A1")
			.filterBounds(roi)
			.filterDate(start, end)
			.map(_mask_modis_night),
			"MODIS nighttime LST",
		)
		.median()
		.select("LST_Night_1km")
		.multiply(0.02)
		.subtract(273.15)
		.rename("lst_night_c")
	)
	distance_to_sea = (
		water.unmask(0)
		.fastDistanceTransform(512, "pixels", "squared_euclidean")
		.sqrt()
		.multiply(30)
		.divide(1000)
		.rename("dist_to_sea_km")
	)
	lst_c = landsat.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename(TARGET_COLUMN)
	emissivity = landsat.select("ST_EMIS").multiply(0.0001).rename("emissivity")
	terrain = ee.Image("USGS/SRTMGL1_003")
	elevation = terrain.select("elevation")
	slope = ee.Terrain.slope(elevation).rename("slope")
	building_density = buildings.select("building_fractional_count").rename("building_density")
	building_height = buildings.select("building_height").rename("building_height")
	u_wind = era5.select("u_component_of_wind_10m").rename("u_wind")
	v_wind = era5.select("v_component_of_wind_10m").rename("v_wind")
	air_temp_c = era5.select("temperature_2m").subtract(273.15).rename("air_temp_c")
	solar_radiation = era5.select("surface_solar_radiation_downwards_hourly").divide(3600).rename("solar_radiation")
	vulnerability = worldpop.divide(1000).clamp(0, 1).rename("vulnerability_index")

	return (
		ee.Image.cat(
			ee.Image.pixelLonLat(),
			ndvi,
			ndbi,
			ndwi,
			mndwi,
			albedo,
			emissivity,
			elevation,
			slope,
			distance_to_sea,
			building_density,
			building_height,
			u_wind,
			v_wind,
			air_temp_c,
			solar_radiation,
			lst_c,
			night_lst_c,
			worldpop.rename("population_100m"),
			vulnerability,
			worldcover.rename("land_cover"),
		)
		.updateMask(worldcover.neq(80))
		.clip(roi)
	)


def _assign_spatial_zone(dataframe: pd.DataFrame) -> pd.Series:
	west = dataframe["lon"] < 72.88
	south = dataframe["lat"] < 19.02
	east = dataframe["lon"] >= 72.94
	north = dataframe["lat"] >= 19.16
	return (1 + west.astype(int) + 2 * south.astype(int) + 2 * east.astype(int) + 2 * north.astype(int)).clip(1, 5)


def sample_real_dataset(
	start: str = DEFAULT_START,
	end: str = DEFAULT_END,
	scale: int = DEFAULT_SCALE,
	num_pixels: int = DEFAULT_NUM_PIXELS,
	output_path: Path | None = None,
) -> pd.DataFrame:
	"""Sample Earth Engine pixels and persist the model-ready Parquet dataset."""
	image = build_feature_image(start, end)
	sample_response = image.sample(
		region=mumbai_roi(),
		scale=scale,
		numPixels=num_pixels,
		seed=42,
		dropNulls=True,
		geometries=False,
		tileScale=4,
	).getInfo() or {}
	samples = sample_response.get("features", [])
	dataframe = pd.DataFrame([feature["properties"] for feature in samples])
	if dataframe.empty:
		raise RuntimeError("Earth Engine returned no complete pixels for the requested period.")
	dataframe = dataframe.rename(columns={"longitude": "lon", "latitude": "lat"})
	dataframe["is_land"] = 1
	dataframe["spatial_zone"] = _assign_spatial_zone(dataframe)
	dataframe = dataframe.drop(columns=["land_cover"], errors="ignore")
	required = [
		"lon",
		"lat",
		"is_land",
		"spatial_zone",
		*FEATURE_COLUMNS,
		TARGET_COLUMN,
		"lst_night_c",
		*EQUITY_COLUMNS,
	]
	missing = sorted(set(required) - set(dataframe.columns))
	if missing:
		raise RuntimeError(f"Earth Engine sample is missing required columns: {missing}")
	dataframe = dataframe[required].dropna().reset_index(drop=True)
	destination = output_path or (PROCESSED_DATA_DIR / "mumbai_features.parquet")
	destination.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_parquet(destination, index=False)
	dataframe.to_csv(destination.with_suffix(".csv"), index=False)
	return dataframe


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--start", default=DEFAULT_START)
	parser.add_argument("--end", default=DEFAULT_END)
	parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
	parser.add_argument("--num-pixels", type=int, default=DEFAULT_NUM_PIXELS)
	parser.add_argument("--project", default=EE_PROJECT)
	args = parser.parse_args()
	initialize_earth_engine(args.project)
	dataframe = sample_real_dataset(args.start, args.end, args.scale, args.num_pixels)
	print(f"Exported {len(dataframe):,} real pixels to {PROCESSED_DATA_DIR / 'mumbai_features.parquet'}")
	print(dataframe[[TARGET_COLUMN, "ndvi", "ndbi"]].describe().to_string())


if __name__ == "__main__":
	main()
