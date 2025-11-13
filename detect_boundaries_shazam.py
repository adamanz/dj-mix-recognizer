#!/usr/bin/env python3
"""
Detect track boundaries using onset detection, then recognize with Shazam using longer chunks
"""

import sys
import asyncio
import json
import librosa
import numpy as np
from shazamio import Shazam
from pydub import AudioSegment
from datetime import timedelta
import time

def format_timestamp(seconds):
    """Convert seconds to MM:SS or HH:MM:SS format"""
    td = timedelta(seconds=int(seconds))
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def detect_boundaries(audio_file, min_separation=30):
    """
    Detect potential track boundaries using onset detection

    Args:
        audio_file: Path to audio file
        min_separation: Minimum seconds between boundaries (default: 30)

    Returns:
        List of boundary timestamps in seconds
    """
    print(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file, sr=None, mono=True)

    print("Detecting onsets and energy changes...")

    # Detect onsets (sudden changes in energy)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        units='time',
        backtrack=True
    )

    # Also detect tempo changes
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Compute spectral flux (energy changes)
    spec = librosa.stft(y)
    spec_flux = np.sqrt(np.sum(np.diff(np.abs(spec), axis=1)**2, axis=0))
    spec_flux_times = librosa.frames_to_time(np.arange(len(spec_flux)), sr=sr)

    # Find peaks in spectral flux
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(spec_flux, prominence=np.percentile(spec_flux, 90))
    flux_peak_times = spec_flux_times[peaks]

    # Combine all potential boundaries
    all_boundaries = sorted(set(list(onsets) + list(flux_peak_times)))

    # Filter to keep only boundaries with minimum separation
    filtered_boundaries = []
    last_boundary = -min_separation

    for boundary in all_boundaries:
        if boundary - last_boundary >= min_separation:
            filtered_boundaries.append(boundary)
            last_boundary = boundary

    return filtered_boundaries, librosa.get_duration(y=y, sr=sr)

async def recognize_chunk_shazam(audio_file, start_time, chunk_length=60):
    """
    Recognize a chunk of audio using Shazam

    Args:
        audio_file: Path to audio file
        start_time: Start time in seconds
        chunk_length: Length of chunk in seconds (default: 60)

    Returns:
        Dictionary with recognition result
    """
    # Extract chunk using ffmpeg
    chunk_file = "/tmp/shazam_chunk.mp3"

    import subprocess
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-ss', str(start_time),
        '-t', str(chunk_length),
        '-i', audio_file,
        '-acodec', 'libmp3lame',
        '-ar', '44100',
        chunk_file
    ]
    subprocess.run(cmd, check=True)

    # Recognize with Shazam
    shazam = Shazam()
    try:
        result = await shazam.recognize(chunk_file)

        if 'track' in result:
            track = result['track']
            return {
                'artist': track.get('subtitle', 'Unknown'),
                'title': track.get('title', 'Unknown'),
                'confidence': 'high'
            }
        else:
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None

async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_boundaries_shazam.py <audio_file> [chunk_length]")
        print("Example: python3 detect_boundaries_shazam.py mix.mp3 60")
        sys.exit(1)

    audio_file = sys.argv[1]
    chunk_length = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    print("=" * 80)
    print("BOUNDARY DETECTION + SHAZAM RECOGNITION")
    print("=" * 80)
    print(f"Chunk length: {chunk_length} seconds")
    print()

    # Step 1: Detect boundaries
    boundaries, duration = detect_boundaries(audio_file, min_separation=30)

    print()
    print(f"Audio duration: {format_timestamp(duration)} ({int(duration)} seconds)")
    print(f"Detected {len(boundaries)} potential track boundaries")
    print()
    print("Detected boundaries:")
    for i, boundary in enumerate(boundaries[:20], 1):  # Show first 20
        print(f"  {i}. {format_timestamp(boundary)}")
    if len(boundaries) > 20:
        print(f"  ... and {len(boundaries) - 20} more")
    print()

    # Step 2: Recognize at each boundary with longer chunks
    print("=" * 80)
    print(f"RECOGNIZING WITH SHAZAM ({chunk_length}s chunks at boundaries)")
    print("=" * 80)
    print()

    results = []
    for i, boundary in enumerate(boundaries, 1):
        # Start recognition 5 seconds after boundary to avoid transition
        start_time = boundary + 5

        # Skip if too close to end
        if start_time + chunk_length > duration:
            continue

        timestamp_str = format_timestamp(start_time)
        print(f"[{timestamp_str}] Boundary {i}/{len(boundaries)}... ", end='', flush=True)

        result = await recognize_chunk_shazam(audio_file, start_time, chunk_length)

        if result:
            print(f"✓ {result['artist']} - {result['title']}")
            results.append({
                'timestamp': int(start_time),
                'timestamp_formatted': timestamp_str,
                **result
            })
        else:
            print("✗ No match")

        # Rate limiting
        time.sleep(1)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Apply temporal smoothing using sliding window majority voting
    def smooth_recognition_results(results, window_size=5, min_track_duration=120):
        """
        Smooth recognition results using sliding window majority voting.
        This approach is scalable and works across different DJ sets.

        Algorithm:
        1. For each detection, look at surrounding detections in a time window
        2. If current track is isolated (appears <min_track_duration) and
           majority of window disagrees, replace with majority track
        3. Uses majority voting principle from audio fingerprinting systems

        Args:
            results: List of recognition results
            window_size: Number of detections to consider (default: 5)
            min_track_duration: Minimum duration in seconds for valid track (default: 120)
        """
        if len(results) < 3:
            return results

        smoothed = []

        for i in range(len(results)):
            curr_track = f"{results[i]['artist']} - {results[i]['title']}"

            # Calculate window boundaries
            window_start = max(0, i - window_size // 2)
            window_end = min(len(results), i + window_size // 2 + 1)

            # Get tracks in window
            window_tracks = []
            for j in range(window_start, window_end):
                if j != i:  # Exclude current
                    track = f"{results[j]['artist']} - {results[j]['title']}"
                    window_tracks.append(track.lower())

            # Calculate current track duration
            if i < len(results) - 1:
                curr_duration = results[i+1]['timestamp'] - results[i]['timestamp']
            else:
                curr_duration = 300  # Assume last track is long enough

            # Count votes in window
            from collections import Counter
            if window_tracks:
                vote_counts = Counter(window_tracks)
                majority_track, majority_count = vote_counts.most_common(1)[0]

                # If current track is brief AND doesn't match majority, smooth it
                if (curr_duration < min_track_duration and
                    curr_track.lower() != majority_track and
                    majority_count >= len(window_tracks) * 0.6):  # 60% majority

                    # Find a result with the majority track to copy metadata
                    for j in range(window_start, window_end):
                        if j != i:
                            check_track = f"{results[j]['artist']} - {results[j]['title']}"
                            if check_track.lower() == majority_track:
                                smoothed_result = results[i].copy()
                                smoothed_result['artist'] = results[j]['artist']
                                smoothed_result['title'] = results[j]['title']
                                smoothed_result['note'] = f"(smoothed from {curr_track})"
                                smoothed.append(smoothed_result)
                                break
                    else:
                        smoothed.append(results[i])
                else:
                    smoothed.append(results[i])
            else:
                smoothed.append(results[i])

        return smoothed

    # Apply smoothing with configurable parameters
    # window_size=5 means looking at 2 before + current + 2 after
    # min_track_duration=120 means tracks <2min are candidates for smoothing
    smoothed_results = smooth_recognition_results(results, window_size=5, min_track_duration=120)

    # Remove consecutive duplicates
    unique_results = []
    for result in smoothed_results:
        if not unique_results or \
           result['artist'].lower() != unique_results[-1]['artist'].lower() or \
           result['title'].lower() != unique_results[-1]['title'].lower():
            unique_results.append(result)

    # Create tracklist
    tracklist = f"🎵 TRACKLIST (Boundary Detection + Shazam {chunk_length}s) 🎵\n\n"

    for i, track in enumerate(unique_results, 1):
        timestamp = track['timestamp_formatted']
        artist = track['artist']
        title = track['title']
        tracklist += f"{timestamp} {artist} - {title}\n"

    tracklist += f"\n✨ Generated using boundary detection + Shazam ({chunk_length}s chunks)\n"
    tracklist += f"📊 Recognition Rate: {len(results)}/{len(boundaries)} boundaries ({len(results)*100/len(boundaries):.1f}%)\n"
    tracklist += f"🎵 Unique Tracks: {len(unique_results)}\n"

    print(tracklist)

    # Save results
    output_base = audio_file.rsplit('.', 1)[0]

    # Save raw JSON
    json_file = f"{output_base}_boundary_shazam_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'boundaries': [float(b) for b in boundaries],
            'chunk_length': chunk_length,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved to: {json_file}")

    # Save tracklist
    tracklist_file = f"{output_base}_boundary_shazam_tracklist.txt"
    with open(tracklist_file, 'w', encoding='utf-8') as f:
        f.write(tracklist)
    print(f"Tracklist saved to: {tracklist_file}")

    print()
    print("=" * 80)

if __name__ == '__main__':
    asyncio.run(main())
