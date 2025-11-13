# DJ Mix Track Recognition with Intelligent Smoothing

Automatically detect and recognize tracks in DJ mixes using boundary detection + Shazam with production-grade temporal smoothing.

## Features

- 🎵 **Boundary Detection**: Uses librosa for onset detection, beat tracking, and spectral flux analysis
- 🔍 **Shazam Recognition**: Configurable chunk lengths (default: 60s) for accurate track identification
- 🧠 **Intelligent Smoothing**: Sliding window majority voting algorithm to filter false positives
- 📊 **High Accuracy**: Successfully detects tracks that other services miss
- ⚙️ **Scalable**: Works across different DJ sets and mixing styles

## Algorithm Overview

### 1. Boundary Detection
Detects potential track transitions using three methods:
- **Onset Detection**: Identifies sudden energy changes
- **Beat Tracking**: Analyzes tempo and beat patterns
- **Spectral Flux**: Detects frequency spectrum changes

Boundaries are filtered to maintain minimum 30-second separation.

### 2. Track Recognition
For each detected boundary:
- Extracts 60-second audio chunk (configurable)
- Sends to Shazam API for recognition
- Records artist, title, timestamp, and confidence

### 3. Temporal Smoothing
Production-grade algorithm inspired by audio fingerprinting systems (NVIDIA NeMo, Dejavu, audfprint):

**Sliding Window Majority Voting**:
- Examines 5 detections: 2 before, current, 2 after
- Uses Python's `collections.Counter` for vote counting
- Replaces brief interruptions (<120s) with majority track
- Requires 60% majority threshold for correction

This eliminates false positives during long continuous tracks (e.g., a 7-minute track interrupted by brief misidentifications).

## Installation

```bash
# Install dependencies
pip install librosa numpy scipy shazamio pydub

# Install ffmpeg (required for audio processing)
brew install ffmpeg  # macOS
# or
sudo apt-get install ffmpeg  # Linux
```

## Usage

```bash
# Basic usage (60-second chunks)
python3 detect_boundaries_shazam.py your_mix.mp3

# Custom chunk length (e.g., 30 seconds)
python3 detect_boundaries_shazam.py your_mix.mp3 30

# Longer chunks for difficult mixes (90 seconds)
python3 detect_boundaries_shazam.py your_mix.mp3 90
```

## Output Files

The script generates three files:

1. **`*_boundary_shazam_results.json`**: Raw detection data with all boundaries and timestamps
2. **`*_boundary_shazam_tracklist.txt`**: Human-readable tracklist with formatted timestamps
3. Console output with recognition progress and statistics

## Example Output

```
🎵 TRACKLIST (Boundary Detection + Shazam 60s) 🎵

04:39 Rex the Dog - Hold It / Control It
13:10 Rex the Dog - Vortex
21:12 Beckers & D'nox - Tiramisu
23:12 Neutron - Poly
27:14 Rex the Dog - Korgasmatron
35:16 Rex the Dog - Change This Pain For Ecstasy
43:18 Rex the Dog - Teufelsberg
50:23 Rex the Dog - Laika
57:59 Rex the Dog - Do You Feel What I Feel (feat. Jamie McDermott)
01:10:37 Mijo - Próximo Berlín (Rex the Dog Remix)
01:19:13 Robyn - Dancing On My Own (Fred Falke Remix Edit)

✨ Generated using boundary detection + Shazam (60s chunks)
📊 Recognition Rate: 121/171 boundaries (70.8%)
🎵 Unique Tracks: 33
```

## Algorithm Parameters

Configurable in the `smooth_recognition_results()` function:

- **`window_size`** (default: 5): Number of detections to examine for majority voting
- **`min_track_duration`** (default: 120): Minimum seconds for a valid track
- **`majority_threshold`** (default: 0.6): Percentage agreement required (60%)

## Why This Approach Works

### Advantages over fixed-interval scanning:
1. **Adaptive Detection**: Finds actual track boundaries instead of arbitrary time points
2. **Longer Chunks**: 60-second recognition windows provide better accuracy
3. **Intelligent Filtering**: Majority voting eliminates false positives during transitions
4. **Scalable**: Works across different genres, BPMs, and mixing styles

### Comparison with other services:
- **ACRCloud 20s**: Missed "Rex the Dog - Korgasmatron" entirely
- **Shazam 30s**: Detected Korgasmatron but had many false positives
- **Boundary + Shazam 60s**: Successfully detected all tracks with intelligent smoothing

## Testing

Run the test script to see smoothing effectiveness:

```bash
python3 test_smoothing.py
```

Example results:
- Original detections: 55
- Corrections made: 11
- Final unique tracks: 33
- False positives removed: Martin Garrix - Animals, brief interruptions, etc.

## Technical Details

### Dependencies:
- **librosa**: Audio analysis and feature extraction
- **shazamio**: Async Shazam API wrapper
- **scipy**: Signal processing for peak detection
- **numpy**: Numerical operations
- **pydub**: Audio format conversion

### Recognition Process:
1. Load full audio file with librosa
2. Compute onset envelope and spectral flux
3. Detect peaks using scipy.signal.find_peaks
4. Extract chunks with ffmpeg
5. Recognize with Shazam asynchronously
6. Apply temporal smoothing
7. Remove consecutive duplicates

## Limitations

- Requires Shazam to have tracks in their database
- Rate limited to avoid API throttling (1 second delay between requests)
- May struggle with very short tracks (<30 seconds)
- Best results with electronic music and clear transitions

## Contributing

This algorithm is based on research into production audio fingerprinting systems. Contributions welcome for:
- Additional smoothing algorithms
- Alternative recognition services
- Performance optimizations
- Multi-threaded processing

## License

MIT License - See LICENSE file

## Credits

Algorithm inspired by:
- NVIDIA NeMo (speaker diarization)
- Dejavu audio fingerprinting
- audfprint robust audio matching
- Librosa audio analysis library
