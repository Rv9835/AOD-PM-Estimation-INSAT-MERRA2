from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from airpollution.sources.cpcb import CPCBConnector
from airpollution.sources.insat import INSATConnector
from airpollution.sources.merra2 import MERRA2Connector

logger = logging.getLogger(__name__)


class SpatiotemporalJoinError(Exception):
    """Raised when spatiotemporal join operations fail."""


class SpatiotemporalJoiner:
    """Orchestrates fetching and joining CPCB, INSAT, and MERRA-2 data."""

    def __init__(
        self,
        cpcb_connector: CPCBConnector | None = None,
        insat_connector: INSATConnector | None = None,
        merra2_connector: MERRA2Connector | None = None,
    ):
        self.cpcb = cpcb_connector or CPCBConnector()
        self.insat = insat_connector or INSATConnector()
        self.merra2 = merra2_connector or MERRA2Connector()

    def fetch_and_join(
        self,
        start_date: datetime,
        end_date: datetime,
        station_ids: list[str] | None = None,
        insat_radius_km: float = 10.0,
        merra2_radius_km: float = 50.0,
    ) -> pd.DataFrame:
        """
        Fetch all data sources and spatiotemporally join them.

        Args:
            start_date: Start of period.
            end_date: End of period.
            station_ids: Optional list of CPCB station IDs.
            insat_radius_km: Radius for nearest-neighbor search to INSAT grid.
            merra2_radius_km: Radius for nearest-neighbor search to MERRA-2 grid.

        Returns:
            Unified DataFrame with columns:
            [time, lat, lon, station_id, pm25, pm10,
             aod, cloud_fraction,
             temperature, humidity, wind_speed, boundary_layer_height]
        """
        logger.info(
            "Fetching data from %s to %s",
            start_date.isoformat(),
            end_date.isoformat(),
        )

        # Fetch all sources
        cpcb_data = self.cpcb.fetch(start_date, end_date, station_ids)
        insat_data = self.insat.fetch(start_date, end_date)
        merra2_data = self.merra2.fetch(start_date, end_date)

        # Join INSAT to CPCB
        joined = self._join_insat_to_cpcb(cpcb_data, insat_data, insat_radius_km)
        logger.info("After INSAT join: %s rows", len(joined))

        # Join MERRA-2 to result
        joined = self._join_merra2_to_cpcb(joined, merra2_data, merra2_radius_km)
        logger.info("After MERRA-2 join: %s rows", len(joined))

        return joined

    @staticmethod
    def _join_insat_to_cpcb(
        cpcb: pd.DataFrame,
        insat: pd.DataFrame,
        radius_km: float,
    ) -> pd.DataFrame:
        """Join INSAT AOD to CPCB stations using nearest-neighbor search."""
        if cpcb.empty or insat.empty:
            raise SpatiotemporalJoinError("CPCB or INSAT data is empty")

        import numpy as np

        # Convert to radians for haversine metric
        insat_coords = np.radians(insat[["lat", "lon"]].values)
        cpcb_coords = np.radians(cpcb[["lat", "lon"]].values)

        # Build spatial index for INSAT
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric="haversine")
        nbrs.fit(insat_coords)

        # Find nearest INSAT point for each CPCB station
        distances, indices = nbrs.kneighbors(cpcb_coords)

        # Convert distance from radians to km (Earth radius ≈ 6371 km)
        distances_km = distances.flatten() * 6371.0

        # Filter pairs within radius
        valid_mask = distances_km <= radius_km
        cpcb_indices = (cpcb.reset_index(drop=True).index.values)[valid_mask]
        insat_indices = indices.flatten()[valid_mask]

        if len(cpcb_indices) == 0:
            raise SpatiotemporalJoinError(f"No INSAT points within {radius_km} km of CPCB stations")

        # Create join pairs
        cpcb_keys = cpcb.iloc[cpcb_indices].reset_index(drop=True)
        cpcb_keys["_cpcb_idx"] = cpcb_indices
        cpcb_keys["_join_key"] = range(len(cpcb_keys))

        insat_keys = insat.iloc[insat_indices].reset_index(drop=True)
        insat_keys["_insat_idx"] = insat_indices
        insat_keys["_join_key"] = range(len(insat_keys))

        # Temporal join: match by nearest time
        merged = pd.merge_asof(
            cpcb_keys.sort_values("time"),
            insat_keys[["time", "aod", "cloud_fraction", "_insat_idx"]].sort_values("time"),
            on="time",
            tolerance=pd.Timedelta(hours=6),
            direction="nearest",
        )

        if merged.empty:
            raise SpatiotemporalJoinError("No temporal overlap between CPCB and INSAT")

        return merged

    @staticmethod
    def _join_merra2_to_cpcb(
        joined: pd.DataFrame,
        merra2: pd.DataFrame,
        radius_km: float,
    ) -> pd.DataFrame:
        """Join MERRA-2 meteorology to CPCB+INSAT joined data."""
        if joined.empty or merra2.empty:
            raise SpatiotemporalJoinError("Joined or MERRA-2 data is empty")

        import numpy as np

        # Convert to radians for haversine metric
        merra2_coords = np.radians(merra2[["lat", "lon"]].values)
        joined_coords = np.radians(joined[["lat", "lon"]].values)

        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", metric="haversine")
        nbrs.fit(merra2_coords)

        distances, indices = nbrs.kneighbors(joined_coords)
        distances_km = distances.flatten() * 6371.0

        valid_mask = distances_km <= radius_km
        joined_indices = (joined.reset_index(drop=True).index.values)[valid_mask]
        merra2_indices = indices.flatten()[valid_mask]

        if len(joined_indices) == 0:
            raise SpatiotemporalJoinError(f"No MERRA-2 points within {radius_km} km")

        joined_keys = joined.iloc[joined_indices].reset_index(drop=True)
        merra2_keys = merra2.iloc[merra2_indices].reset_index(drop=True)

        # Temporal join on already-filtered indices
        final = pd.merge_asof(
            joined_keys.sort_values("time"),
            merra2_keys[
                ["time", "temperature", "humidity", "wind_speed", "boundary_layer_height"]
            ].sort_values("time"),
            on="time",
            tolerance=pd.Timedelta(hours=12),
            direction="nearest",
        )

        if final.empty:
            raise SpatiotemporalJoinError("No temporal overlap between CPCB+INSAT and MERRA-2")

        return final
