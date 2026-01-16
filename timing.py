"""
SongGeneration Studio - Timing History
Track generation times for smart estimates.

Algorithm:
- Each model has a base default time (Base: 3:00, Base Full: 4:00, Large: 6:00)
- Initial estimate = base + complexity_factor (based on lyrics chars and sections)
- After each generation, the estimate is gradually updated using exponential moving average
- More lyrics and sections = higher complexity = higher estimate
- Records are grouped by "complexity bucket" for better matching
"""

import json
from config import TIMING_FILE, MAX_TIMING_RECORDS

# ============================================================================
# Timing History for Smart Estimates
# ============================================================================

def load_timing_history() -> list:
    """Load timing history from file"""
    try:
        if TIMING_FILE.exists():
            with open(TIMING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[TIMING] Error loading timing history: {e}")
    return []


def save_timing_history(history: list):
    """Save timing history to file"""
    try:
        with open(TIMING_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[TIMING] Error saving timing history: {e}")


def get_complexity_bucket(num_sections: int, lyrics_length: int) -> str:
    """
    Categorize generation into complexity buckets for better matching.
    Returns a string key like "low", "medium", "high", "very_high"
    """
    # Calculate complexity score
    # Sections contribute: 0-3 = low, 4-6 = medium, 7+ = high
    # Lyrics contribute: 0 = none, 1-500 = low, 501-1500 = medium, 1500+ = high

    section_score = 0 if num_sections <= 3 else (1 if num_sections <= 6 else 2)

    if lyrics_length == 0:
        lyrics_score = 0
    elif lyrics_length <= 500:
        lyrics_score = 1
    elif lyrics_length <= 1500:
        lyrics_score = 2
    else:
        lyrics_score = 3

    total = section_score + lyrics_score

    if total <= 1:
        return "low"
    elif total <= 3:
        return "medium"
    elif total <= 4:
        return "high"
    else:
        return "very_high"


def save_timing_record(metadata: dict):
    """Save a timing record from a completed generation"""
    if metadata.get("generation_time_seconds", 0) <= 0:
        return  # Don't save if no valid timing

    num_sections = metadata.get("num_sections", 0)
    lyrics_length = metadata.get("total_lyrics_length", 0)

    record = {
        "model": metadata.get("model"),
        "num_sections": num_sections,
        "total_lyrics_length": lyrics_length,
        "complexity_bucket": get_complexity_bucket(num_sections, lyrics_length),
        "has_lyrics": metadata.get("has_lyrics", False),
        "output_mode": metadata.get("output_mode", "mixed"),
        "has_reference": bool(metadata.get("reference_audio_id")),
        "generation_time_seconds": metadata.get("generation_time_seconds"),
        "completed_at": metadata.get("completed_at"),
    }

    history = load_timing_history()
    history.append(record)

    # Keep only last N records
    if len(history) > MAX_TIMING_RECORDS:
        history = history[-MAX_TIMING_RECORDS:]

    save_timing_history(history)
    print(f"[TIMING] Saved record: {record['model']}, {record['num_sections']} sections, "
          f"{lyrics_length} chars, bucket={record['complexity_bucket']}, {record['generation_time_seconds']}s")


def get_timing_stats() -> dict:
    """
    Calculate timing statistics from history for smart estimates.
    Returns per-model stats with complexity bucket averages for gradual learning.
    """
    history = load_timing_history()

    if not history:
        return {"has_history": False, "models": {}}

    # Group by model
    model_stats = {}
    for record in history:
        model = record.get("model", "unknown")
        if model not in model_stats:
            model_stats[model] = {
                "all_times": [],
                "by_bucket": {"low": [], "medium": [], "high": [], "very_high": []},
                "by_lyrics": {"with": [], "without": []},
                "by_reference": {"with": [], "without": []},
                # Store raw records for weighted averaging
                "records": [],
            }

        time_sec = record.get("generation_time_seconds", 0)
        if time_sec <= 0:
            continue

        model_stats[model]["all_times"].append(time_sec)
        model_stats[model]["records"].append(record)

        # Group by complexity bucket
        bucket = record.get("complexity_bucket", get_complexity_bucket(
            record.get("num_sections", 0),
            record.get("total_lyrics_length", 0)
        ))
        if bucket in model_stats[model]["by_bucket"]:
            model_stats[model]["by_bucket"][bucket].append(time_sec)

        # Track by lyrics presence
        lyrics_key = "with" if record.get("has_lyrics") else "without"
        model_stats[model]["by_lyrics"][lyrics_key].append(time_sec)

        # Track by reference presence
        ref_key = "with" if record.get("has_reference") else "without"
        model_stats[model]["by_reference"][ref_key].append(time_sec)

    # Calculate statistics with exponential weighting (recent records matter more)
    result = {"has_history": True, "models": {}}

    # Only use last N records for EMA to be more responsive to recent performance
    RECENT_RECORDS_FOR_EMA = 20

    for model, stats in model_stats.items():
        if not stats["all_times"]:
            continue

        # Calculate weighted average using only recent records
        def weighted_avg(times_list, alpha=0.5):
            """Exponential moving average - only uses recent records for responsiveness"""
            if not times_list:
                return None

            # Only use last N records for more responsive estimates
            recent = times_list[-RECENT_RECORDS_FOR_EMA:] if len(times_list) > RECENT_RECORDS_FOR_EMA else times_list

            if len(recent) == 1:
                return int(recent[0])

            # Start with oldest of recent, apply EMA
            ema = recent[0]
            for t in recent[1:]:
                ema = alpha * t + (1 - alpha) * ema
            return int(ema)

        result["models"][model] = {
            "count": len(stats["all_times"]),
            "avg_time": weighted_avg(stats["all_times"]),
            "min_time": min(stats["all_times"]),
            "max_time": max(stats["all_times"]),
            # Complexity bucket averages (EMA weighted)
            "by_bucket": {
                bucket: weighted_avg(times)
                for bucket, times in stats["by_bucket"].items()
                if times
            },
            # Lyrics impact
            "avg_with_lyrics": weighted_avg(stats["by_lyrics"]["with"]),
            "avg_without_lyrics": weighted_avg(stats["by_lyrics"]["without"]),
            # Reference impact
            "avg_with_reference": weighted_avg(stats["by_reference"]["with"]),
            "avg_without_reference": weighted_avg(stats["by_reference"]["without"]),
        }

    return result
