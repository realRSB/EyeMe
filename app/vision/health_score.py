import time

def compute_strain_level(blink_log, current_time, window=300):
    """
    Computes an eye strain level based on the blink count in the past `window` seconds.
    
    Parameters:
      blink_log (deque): Timestamps of detected blinks.
      current_time (float): Current time (in seconds).
      window (int): Time window in seconds to consider (default: 300 seconds).
    
    Returns:
      str: A strain level label: "Low", "Moderate", or "High".
    """
    recent_blinks = [t for t in blink_log if current_time - t <= window]
    blink_count = len(recent_blinks)
    if blink_count >= 40:
        return "Low"
    elif 20 <= blink_count < 40:
        return "Moderate"
    else:
        return "High"

def compute_eye_health_score(blink_rate, redness, blink_log, elapsed_time, baseline_blink_rate=8):
    """
    Compute an overall eye health score (0-100) based on:
      - Blink rate: Expecting around 8 blinks per minute as baseline.
      - Redness: Ratio of red-dominant pixels from the sclera; values above 0.05 trigger a penalty.
      - Strain: Derived from the number of recent blinks as a proxy for eye strain.
      
    If the elapsed time is less than 60 seconds, the score returns a default neutral value
    (50) and strain as "Insufficient Data", because measurements are not yet reliable.
    
    Parameters:
      blink_rate (float): Current blink rate (blinks per minute).
      redness (float): Measured redness ratio.
      blink_log (deque): Timestamps of detected blinks.
      elapsed_time (float): Time in seconds since the software started.
      baseline_blink_rate (float): Expected healthy blink rate (default 8 blinks/min).
      
    Returns:
      tuple: (score, strain) where score is between 0 and 100 and strain is a string label.
    """
    # Check for warm-up period
    if elapsed_time < 60:
        return 50, "Insufficient Data"

    score = 100  # Start with a perfect score.

    # Penalize for low blink rate if below the baseline.
    if blink_rate < baseline_blink_rate:
        blink_penalty = 25 * (baseline_blink_rate - blink_rate) / baseline_blink_rate
        score -= blink_penalty

    # Penalize for redness if it exceeds the threshold (0.05).
    if redness > 0.05:
        redness_penalty = 25 * ((redness - 0.05) / 0.05)
        score -= redness_penalty

    # Compute strain level from the blink log over the past 5 minutes (300 sec).
    current_time = time.time()
    strain = compute_strain_level(blink_log, current_time, window=300)
    if strain == "Moderate":
        score -= 10
    elif strain == "High":
        score -= 20

    # Clamp score between 0 and 100.
    score = max(0, min(100, score))
    return score, strain
