#!/usr/bin/env python3
"""
Test the temporal smoothing algorithm on existing results
"""

import json

# Load existing results
with open('dj_set_93ZGx5wjRdo_boundary_shazam_results.json', 'r') as f:
    data = json.load(f)
    results = data['results']

def smooth_recognition_results(results, min_track_duration=120):
    """
    Smooth recognition results using temporal filtering.
    If track A plays, then track B for <min_track_duration seconds, then track A again,
    replace B with A (it was likely a false positive during a transition).
    """
    if len(results) < 3:
        return results

    smoothed = [results[0]]
    corrections = []

    for i in range(1, len(results) - 1):
        prev_track = f"{results[i-1]['artist']} - {results[i-1]['title']}"
        curr_track = f"{results[i]['artist']} - {results[i]['title']}"
        next_track = f"{results[i+1]['artist']} - {results[i+1]['title']}"

        # Calculate how long current track plays
        curr_duration = results[i+1]['timestamp'] - results[i]['timestamp']

        # If surrounded by same track and plays <2 min, it's likely false positive
        if (prev_track.lower() == next_track.lower() and
            prev_track.lower() != curr_track.lower() and
            curr_duration < min_track_duration):
            # Replace with surrounding track
            smoothed_result = results[i].copy()
            smoothed_result['artist'] = results[i-1]['artist']
            smoothed_result['title'] = results[i-1]['title']

            corrections.append({
                'timestamp': results[i]['timestamp_formatted'],
                'original': curr_track,
                'corrected_to': prev_track,
                'duration': curr_duration
            })

            smoothed.append(smoothed_result)
        else:
            smoothed.append(results[i])

    # Add last result
    smoothed.append(results[-1])

    return smoothed, corrections

# Apply smoothing
smoothed_results, corrections = smooth_recognition_results(results)

print("=" * 80)
print("TEMPORAL SMOOTHING ANALYSIS")
print("=" * 80)
print()
print(f"Original detections: {len(results)}")
print(f"Corrections made: {len(corrections)}")
print()

if corrections:
    print("Corrections:")
    for c in corrections:
        print(f"  {c['timestamp']} - {c['original']}")
        print(f"    → Corrected to: {c['corrected_to']}")
        print(f"    → Duration was only: {c['duration']}s")
        print()
else:
    print("No corrections needed!")

# Remove consecutive duplicates
unique_original = []
for result in results:
    if not unique_original or \
       result['artist'].lower() != unique_original[-1]['artist'].lower() or \
       result['title'].lower() != unique_original[-1]['title'].lower():
        unique_original.append(result)

unique_smoothed = []
for result in smoothed_results:
    if not unique_smoothed or \
       result['artist'].lower() != unique_smoothed[-1]['artist'].lower() or \
       result['title'].lower() != unique_smoothed[-1]['title'].lower():
        unique_smoothed.append(result)

print("=" * 80)
print("TRACKLIST COMPARISON")
print("=" * 80)
print()
print(f"Before smoothing: {len(unique_original)} unique tracks")
print(f"After smoothing: {len(unique_smoothed)} unique tracks")
print(f"Removed: {len(unique_original) - len(unique_smoothed)} false positive track changes")
