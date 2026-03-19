"""
Script to load Kerry Emanuel TC tracks and compute wind fields using CLIMADA
Processes a single synthetic event set - designed for parallel execution
"""

from pathlib import Path
from climada.hazard import Centroids, TropCyclone, TCTracks
from climada.util.constants import SYSTEM_DIR
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Define Florida bounding box
FL_BOUNDS = {
    'min_lat': 24.0,
    'max_lat': 31.5,
    'min_lon': -90.0,
    'max_lon': -79.0
}

def create_centroids():
    """Create centroids grid for Florida at 120 arcsec resolution"""
    LOGGER.info("Creating centroids grid...")
    cent = Centroids.from_pnt_bounds(
        (FL_BOUNDS['min_lon'], FL_BOUNDS['min_lat'], 
         FL_BOUNDS['max_lon'], FL_BOUNDS['max_lat']), 
        res=120/3600  # 120 arcsec = 1/30 degree ≈ 3.7 km resolution
    )
    LOGGER.info(f"Created {cent.size} centroids")
    return cent

def get_output_filename(track_file_path):
    """
    Extract output filename from track file path
    
    Example:
    Simona_FLA_AL_canesm_20thcal.mat -> FL_canesm_20thcal.hdf5
    Simona_FLA_AL_era5_reanalcal.mat -> FL_era5_reanalcal.hdf5
    """
    # Get filename without extension
    name = track_file_path.name
    if name.endswith('.mat'):
        name = name[:-4]  # Remove .mat extension
    
    # Remove "Simona_FLA_AL_" prefix and add "FL_" prefix
    if name.startswith("Simona_FLA_AL_"):
        output_name = "FL_" + name.replace("Simona_FLA_AL_", "")
    else:
        output_name = "FL_" + name
    return output_name + ".hdf5"

def process_track_file(track_file, centroids, output_dir):
    """
    Process a single track file: load tracks, compute windfields, save hazard
    
    Parameters
    ----------
    track_file : Path
        Path to the track file (zip)
    centroids : Centroids
        CLIMADA Centroids object
    output_dir : Path
        Directory to save output hazard files
    """
    LOGGER.info("="*80)
    LOGGER.info(f"Processing: {track_file.name}")
    LOGGER.info("="*80)
    
    # Load tracks from Kerry Emanuel simulations
    LOGGER.info(f"Loading tracks from {track_file}...")
    tc_tracks = TCTracks.from_simulations_emanuel(
        str(track_file),
        hemisphere='N'  # Northern hemisphere (Atlantic)
    )
    LOGGER.info(f"Loaded {tc_tracks.size} tracks")
    
    if tc_tracks.size == 0:
        LOGGER.warning(f"No tracks found in {track_file.name}, skipping...")
        return
    
    # Equal timestep (0.5 hour)
    LOGGER.info("Resampling tracks to equal timestep...")
    tc_tracks.equal_timestep(time_step_h=0.5)
    
    # Generate wind fields
    LOGGER.info("Computing wind fields...")
    tc_haz = TropCyclone.from_tracks(tc_tracks, centroids=centroids)
    
    # Set event names
    LOGGER.info("Setting event names...")
    output_basename = get_output_filename(track_file).replace('.hdf5', '')
    tc_haz.event_name = [f"{output_basename}_{i:05d}" for i in range(tc_haz.event_id.size)]
    
    # Save output
    output_file = output_dir / get_output_filename(track_file)
    LOGGER.info(f"Saving wind fields to {output_file}")
    tc_haz.write_hdf5(output_file)
    
    # Print summary
    print("\n" + "-"*60)
    print(f"COMPLETED: {track_file.name}")
    print("-"*60)
    print(f"  Tracks: {tc_tracks.size}")
    print(f"  Events in hazard: {tc_haz.event_id.size}")
    print(f"  Max wind speed (m/s): {tc_haz.intensity.max():.2f}")
    print(f"  Output: {output_file.name}")
    print("-"*60 + "\n")

# ============== MAIN SCRIPT ==============

if __name__ == "__main__":
    
    # Get track filename from command line argument
    if len(sys.argv) < 2:
        LOGGER.error("Usage: python compute_windfields_emanuel.py <track_filename>")
        LOGGER.error("Example: python compute_windfields_emanuel.py Simona_FLA_AL_canesm_20thcal.mat")
        sys.exit(1)
    
    track_filename = sys.argv[1]
    
    # Set paths
    tracks_dir = Path(SYSTEM_DIR) / "tracks" / "Kerry" / "Florida"
    output_dir = Path(SYSTEM_DIR) / "hazard" / "Florida"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    track_file = tracks_dir / track_filename
    
    # Check if track file exists
    if not track_file.exists():
        LOGGER.error(f"Track file not found: {track_file}")
        sys.exit(1)
    
    LOGGER.info(f"Track file: {track_file}")
    LOGGER.info(f"Output directory: {output_dir}")
    
    # Create centroids
    centroids = create_centroids()
    
    # Process the track file
    print("\n" + "="*80)
    print(f"PROCESSING: {track_filename}")
    print("="*80 + "\n")
    
    try:
        process_track_file(track_file, centroids, output_dir)
        print("\n" + "="*80)
        print("SUCCESS")
        print("="*80)
    except Exception as e:
        LOGGER.error(f"Failed to process {track_filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
