"""
base_helper_functions.py - Deterministic data retrieval and weather enrichment functions.
Supports dynamic column projection to conserve LLM token budget.
"""

import os
from typing import Any, Dict, List, Optional
from pymongo import ASCENDING, MongoClient
import requests

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "ahs_ces_data"

_client: Optional[MongoClient] = None


def _get_db():
    """Singleton MongoDB client to avoid socket exhaustion and connection overhead."""
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME]


def _build_projection(
    fields: Optional[List[str]] = None, extra_keys: Optional[List[str]] = None
) -> Dict[str, int]:
    """Constructs a MongoDB projection dictionary, excluding '_id' and enforcing temporal keys."""
    projection: Dict[str, int] = {"_id": 0}
    if fields:
        mandatory = ["uid", "day"] + (extra_keys or [])
        for key in mandatory:
            projection[key] = 1
        for field in fields:
            projection[field] = 1
    return projection


def get_daily_summary(
    uid: str,
    start_date: str,
    end_date: str,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves user daily aggregate records. Use the 'fields' parameter to
    fetch only the required columns to avoid overflowing the LLM context.

    Args:
        uid: User identifier (e.g., 'u001').
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        fields: Optional list of specific columns to retrieve.
            - Mobility / Activity: 'step', 'loc_dist', 'act_on_foot', 'act_still', 'act_on_bike', 'act_in_vehicle'
            - Semantic Locations (duration in seconds): 'loc_home_dur', 'loc_study_dur', 'loc_workout_dur',
              'loc_food_dur', 'loc_leisure_dur', 'loc_self_dorm_dur', 'loc_other_dorm_dur',
              'loc_social_dur', 'loc_health_dur', 'loc_worship_dur'
            - Screen / Conversations: 'unlock_duration', 'unlock_num', 'audio_convo_duration'
            - Phone Calls: 'call_in_duration', 'call_in_num', 'call_out_duration', 'call_out_num', 'call_miss_num'
            - Sleep: 'sleep_dur'

    Returns:
        List of daily records sorted chronologically.
    """
    collection = _get_db()["daily_data"]
    query = {"uid": uid, "day": {"$gte": start_date, "$lte": end_date}}
    projection = _build_projection(fields)
    return list(collection.find(query, projection).sort("day", ASCENDING))


def get_hourly_slice(
    uid: str,
    start_date: str,
    end_date: str,
    start_hour: Optional[int] = None,
    end_hour: Optional[int] = None,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves hourly time-series data (tall format). Suitable for analyzing
    specific time slots (e.g., nighttime behavior, study hours).

    Args:
        uid: User identifier (e.g., 'u001').
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        start_hour: Optional starting hour (0-23).
        end_hour: Optional ending hour (0-23).
        fields: Optional list of metrics.
            Available columns: 'step', 'loc_dist', 'act_on_foot', 'act_still',
            'act_on_bike', 'act_in_vehicle', 'unlock_duration', 'unlock_num',
            'audio_convo_duration', 'other_playing_duration'

    Returns:
        List of hourly records sorted by day and hour.
    """
    collection = _get_db()["hourly_data"]
    query: Dict[str, Any] = {"uid": uid, "day": {"$gte": start_date, "$lte": end_date}}

    if start_hour is not None and end_hour is not None:
        query["hour"] = {"$gte": int(start_hour), "$lte": int(end_hour)}
    elif start_hour is not None:
        query["hour"] = {"$gte": int(start_hour)}
    elif end_hour is not None:
        query["hour"] = {"$lte": int(end_hour)}

    projection = _build_projection(fields, extra_keys=["hour"])
    return list(collection.find(query, projection).sort([("day", ASCENDING), ("hour", ASCENDING)]))


def get_ema_surveys(
    uid: str,
    start_date: str,
    end_date: str,
    fields: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves subjective Ecological Momentary Assessment (EMA) survey records.

    Args:
        uid: User identifier.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        fields: Specific psychological scales to retrieve.
            - Mood / Affect: 'pam'
            - Anxiety / Depression: 'phq4-1', 'phq4-2', 'phq4-3', 'phq4-4', 'phq4_score'
            - Stress / Self-Esteem: 'sse3-1', 'sse3-2', 'sse3-3', 'sse3-4', 'stress'
            - Social Level: 'social_level'

    Returns:
        List of valid EMA responses submitted during the period.
    """
    collection = _get_db()["ema_data"]
    query = {"uid": uid, "day": {"$gte": start_date, "$lte": end_date}}
    projection = _build_projection(fields)
    return list(collection.find(query, projection).sort("day", ASCENDING))


def get_historical_weather(
    start_date: str,
    end_date: str,
    hourly: bool = False,
) -> List[Dict[str, Any]]:
    """
    Queries historical weather observations for Hanover, NH (Dartmouth College, 43.7022 N, -72.2896 W)[cite: 4].
    Useful for checking if drops in steps or mobility were caused by rain, snow, or extreme cold[cite: 4].

    Args:
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        hourly: False for aggregated daily values, True for hourly sampling.

    Returns:
        List of weather observations (temperature, precipitation, snowfall, wind speed).
    """
    endpoint = "https://archive-api.open-meteo.com/v1/archive"
    params: Dict[str, Any] = {
        "latitude": 43.7022,
        "longitude": -72.2896,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "America/New_York",
    }

    if hourly:
        params["hourly"] = [
            "temperature_2m",
            "precipitation",
            "snowfall",
            "weather_code",
            "wind_speed_10m",
        ]
    else:
        params["daily"] = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "snowfall_sum",
            "weather_code",
            "wind_speed_10m_max",
        ]

    try:
        response = requests.get(endpoint, params=params, timeout=10.0)
        response.raise_for_status()
        payload = response.json()

        key = "hourly" if hourly else "daily"
        if key not in payload or "time" not in payload[key]:
            return []

        data_block = payload[key]
        times = data_block["time"]
        metrics = [k for k in data_block.keys() if k != "time"]

        records = []
        for i, timestamp in enumerate(times):
            entry = {"timestamp": timestamp}
            for m in metrics:
                entry[m] = data_block[m][i]
            records.append(entry)
        return records
    except requests.exceptions.RequestException as e:
        print(f"[get_historical_weather] Open-Meteo error: {e}")
        return []


# Explicit list of exposed tools for PydanticAI agent registration
BASE_TOOLS = [
    get_daily_summary,
    get_hourly_slice,
    get_ema_surveys,
    get_historical_weather,
]