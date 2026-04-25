"""
Multi-City Registry and Configuration Module
Provides centralized city metadata, bounding boxes, altitude data, and geospatial utilities.
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import math
import logging

logger = logging.getLogger(__name__)


# =====================================================
# TYPE ALIASES FOR CLARITY
# =====================================================
BoundingBox = Tuple[float, float, float, float]  # (min_lat, max_lat, min_lon, max_lon)
Coordinates = Tuple[float, float]  # (latitude, longitude)


# =====================================================
# CITIES REGISTRY: Centralized Configuration
# =====================================================
CITIES_REGISTRY: Dict[str, Dict] = {
    "delhi": {
        "display_name": "National Capital Territory of Delhi",
        "country": "India",
        "center_lat": 28.7041,
        "center_lon": 77.1025,
        # Bounding box: [min_lat, max_lat, min_lon, max_lon]
        # Delhi UCR (Urban Conglomeration Region) bounds
        "bounding_box": (28.40, 29.00, 76.80, 77.50),
        "altitude_m": 216,  # Mean altitude above sea level
        "population_density": 11320,  # People per km²
        "data_sources": ["cpcb", "insat", "merra2"],
        "timezone": "Asia/Kolkata",
        "region_type": "metropolitan",
    },
    "mumbai": {
        "display_name": "Mumbai (Bombay)",
        "country": "India",
        "center_lat": 19.0760,
        "center_lon": 72.8777,
        # Greater Mumbai bounds
        "bounding_box": (18.80, 19.40, 72.50, 73.20),
        "altitude_m": 14,  # Coastal city
        "population_density": 20961,  # People per km²
        "data_sources": ["cpcb", "insat", "merra2"],
        "timezone": "Asia/Kolkata",
        "region_type": "metropolitan",
    },
    "bangalore": {
        "display_name": "Bangalore (Bengaluru)",
        "country": "India",
        "center_lat": 12.9716,
        "center_lon": 77.5946,
        # Greater Bangalore bounds
        "bounding_box": (12.60, 13.30, 77.30, 77.90),
        "altitude_m": 920,  # High altitude city
        "population_density": 4736,  # People per km²
        "data_sources": ["cpcb", "insat", "merra2"],
        "timezone": "Asia/Kolkata",
        "region_type": "metropolitan",
    },
    "kolkata": {
        "display_name": "Kolkata (Calcutta)",
        "country": "India",
        "center_lat": 22.5726,
        "center_lon": 88.3639,
        # Greater Kolkata bounds
        "bounding_box": (22.30, 22.80, 88.10, 88.60),
        "altitude_m": 9,  # Gangetic plain, very low altitude
        "population_density": 9068,  # People per km²
        "data_sources": ["cpcb", "insat", "merra2"],
        "timezone": "Asia/Kolkata",
        "region_type": "metropolitan",
    },
    "hyderabad": {
        "display_name": "Hyderabad",
        "country": "India",
        "center_lat": 17.3850,
        "center_lon": 78.4867,
        # Greater Hyderabad bounds
        "bounding_box": (17.10, 17.70, 78.20, 78.80),
        "altitude_m": 505,  # Plateau region
        "population_density": 6355,  # People per km²
        "data_sources": ["cpcb", "insat", "merra2"],
        "timezone": "Asia/Kolkata",
        "region_type": "metropolitan",
    },
}


# =====================================================
# DATACLASS: CityConfig with Validation
# =====================================================
@dataclass
class CityConfig:
    """
    Immutable configuration for a single city.
    Supports strict validation of coordinates, bounds, and metadata.
    """

    city_key: str
    display_name: str
    center_lat: float
    center_lon: float
    bounding_box: BoundingBox
    altitude_m: float
    population_density: float
    data_sources: List[str] = field(default_factory=list)
    timezone: str = "Asia/Kolkata"
    region_type: str = "metropolitan"

    def __post_init__(self) -> None:
        """Validate all input parameters after initialization."""
        self._validate_coordinates()
        self._validate_bounding_box()
        self._validate_altitude()
        self._validate_population_density()

    def _validate_coordinates(self) -> None:
        """Ensure center coordinates are within valid latitude/longitude ranges."""
        if not -90 <= self.center_lat <= 90:
            raise ValueError(
                f"Center latitude {self.center_lat} out of range [-90, 90] for {self.city_key}"
            )
        if not -180 <= self.center_lon <= 180:
            raise ValueError(
                f"Center longitude {self.center_lon} out of range [-180, 180] for {self.city_key}"
            )

    def _validate_bounding_box(self) -> None:
        """Validate bounding box bounds and ordering."""
        min_lat, max_lat, min_lon, max_lon = self.bounding_box

        if not -90 <= min_lat <= 90 or not -90 <= max_lat <= 90:
            raise ValueError(
                f"Latitude bounds {min_lat}, {max_lat} out of range [-90, 90] for {self.city_key}"
            )
        if not -180 <= min_lon <= 180 or not -180 <= max_lon <= 180:
            raise ValueError(
                f"Longitude bounds {min_lon}, {max_lon} out of range [-180, 180] for {self.city_key}"
            )
        if min_lat >= max_lat:
            raise ValueError(
                f"Bounding box: min_lat ({min_lat}) >= max_lat ({max_lat}) for {self.city_key}"
            )
        if min_lon >= max_lon:
            raise ValueError(
                f"Bounding box: min_lon ({min_lon}) >= max_lon ({max_lon}) for {self.city_key}"
            )

        # Check if center is within bounds
        if not (min_lat <= self.center_lat <= max_lat and min_lon <= self.center_lon <= max_lon):
            logger.warning(
                f"Center ({self.center_lat}, {self.center_lon}) is outside "
                f"bounding box {self.bounding_box} for {self.city_key}"
            )

    def _validate_altitude(self) -> None:
        """Ensure altitude is within realistic range."""
        if not -500 <= self.altitude_m <= 9000:
            raise ValueError(
                f"Altitude {self.altitude_m}m out of realistic range [-500, 9000]m for {self.city_key}"
            )

    def _validate_population_density(self) -> None:
        """Ensure population density is non-negative."""
        if self.population_density < 0:
            raise ValueError(
                f"Population density cannot be negative ({self.population_density}) for {self.city_key}"
            )

    def contains_point(self, latitude: float, longitude: float) -> bool:
        """
        Check if a given coordinate point is within the city's bounding box.

        Args:
            latitude: Point latitude
            longitude: Point longitude

        Returns:
            bool: True if point is within bounds, False otherwise
        """
        min_lat, max_lat, min_lon, max_lon = self.bounding_box
        return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon

    def to_dict(self) -> Dict:
        """Convert CityConfig to dictionary representation."""
        return {
            "city_key": self.city_key,
            "display_name": self.display_name,
            "center_lat": self.center_lat,
            "center_lon": self.center_lon,
            "bounding_box": self.bounding_box,
            "altitude_m": self.altitude_m,
            "population_density": self.population_density,
            "data_sources": self.data_sources,
            "timezone": self.timezone,
            "region_type": self.region_type,
        }


# =====================================================
# CITY MANAGER: Load and validate cities from registry
# =====================================================
class CityManager:
    """
    Centralized manager for city configurations.
    Provides methods to retrieve, validate, and work with city metadata.
    """

    def __init__(self) -> None:
        """Initialize CityManager and load all cities from registry."""
        self._cities: Dict[str, CityConfig] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load and validate all cities from CITIES_REGISTRY."""
        for city_key, city_data in CITIES_REGISTRY.items():
            try:
                city_config = CityConfig(
                    city_key=city_key,
                    display_name=city_data.get("display_name", city_key),
                    center_lat=city_data["center_lat"],
                    center_lon=city_data["center_lon"],
                    bounding_box=city_data["bounding_box"],
                    altitude_m=city_data["altitude_m"],
                    population_density=city_data["population_density"],
                    data_sources=city_data.get("data_sources", []),
                    timezone=city_data.get("timezone", "Asia/Kolkata"),
                    region_type=city_data.get("region_type", "metropolitan"),
                )
                self._cities[city_key] = city_config
                logger.info(f"✓ Loaded city config: {city_key} ({city_config.display_name})")
            except ValueError as e:
                logger.error(f"✗ Failed to load city config for {city_key}: {e}")
                raise

    def get_city(self, city_key: str) -> Optional[CityConfig]:
        """
        Retrieve a city config by key.

        Args:
            city_key: City identifier (e.g., "delhi", "mumbai")

        Returns:
            CityConfig if found, None otherwise
        """
        return self._cities.get(city_key)

    def get_city_strict(self, city_key: str) -> CityConfig:
        """
        Retrieve a city config by key. Raises error if not found.

        Args:
            city_key: City identifier

        Raises:
            ValueError: If city not found

        Returns:
            CityConfig
        """
        city = self._cities.get(city_key)
        if city is None:
            available = ", ".join(self._cities.keys())
            raise ValueError(
                f"City '{city_key}' not found in registry. Available cities: {available}"
            )
        return city

    def list_cities(self) -> List[str]:
        """Return list of all available city keys."""
        return list(self._cities.keys())

    def get_all_cities(self) -> Dict[str, CityConfig]:
        """Return dictionary of all city configurations."""
        return self._cities.copy()

    def get_city_pair_distance(self, city_key_1: str, city_key_2: str) -> float:
        """
        Calculate Haversine distance between two cities' centers.

        Args:
            city_key_1: First city key
            city_key_2: Second city key

        Returns:
            Distance in kilometers

        Raises:
            ValueError: If either city not found
        """
        city1 = self.get_city_strict(city_key_1)
        city2 = self.get_city_strict(city_key_2)

        return haversine_distance(
            (city1.center_lat, city1.center_lon), (city2.center_lat, city2.center_lon)
        )


# =====================================================
# GEOSPATIAL UTILITIES: Haversine Distance
# =====================================================
def haversine_distance(coord1: Coordinates, coord2: Coordinates) -> float:
    """
    Calculate the great-circle distance between two points on Earth using Haversine formula.

    Args:
        coord1: Tuple of (latitude, longitude) in degrees
        coord2: Tuple of (latitude, longitude) in degrees

    Returns:
        Distance in kilometers

    Formula:
        a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
        c = 2 * atan2(√a, √(1−a))
        d = R * c

        where φ is latitude, λ is longitude, R is earth's radius (6371 km)
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    return distance


def bearing_between_cities(
    coord1: Coordinates, coord2: Coordinates
) -> float:
    """
    Calculate bearing (initial direction) from coord1 to coord2.

    Args:
        coord1: Starting coordinates (latitude, longitude)
        coord2: Ending coordinates (latitude, longitude)

    Returns:
        Bearing in degrees (0-360), where 0 is North, 90 is East, 180 is South, 270 is West
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lon = lon2_rad - lon1_rad

    x = math.sin(delta_lon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(
        lat2_rad
    ) * math.cos(delta_lon)

    bearing_rad = math.atan2(x, y)
    bearing_deg = (math.degrees(bearing_rad) + 360) % 360

    return bearing_deg


# =====================================================
# INITIALIZATION
# =====================================================
# Create a global CityManager instance for convenience
_city_manager = CityManager()


def get_city(city_key: str) -> Optional[CityConfig]:
    """Convenience function to get city config."""
    return _city_manager.get_city(city_key)


def list_available_cities() -> List[str]:
    """Convenience function to list all available cities."""
    return _city_manager.list_cities()
