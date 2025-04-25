import numpy as np

def calculate_rh_metrics(waveform, ground_elevation, bin_size=0.15, percentiles=[0, 25, 50, 75, 98, 100]):
    """
    Calculate Relative Height (RH) metrics from a GEDI waveform
    
    Parameters:
    -----------
    waveform : numpy.ndarray
        Normalized waveform array of length 512, with ground already found and standardized.
        Values represent energy returns at each bin.
    ground_elevation : float
        Elevation of the ground point in meters
    bin_size : float, optional
        Size of each bin in meters (default is 0.15m for GEDI)
    percentiles : list, optional
        RH percentiles to calculate (default: [0, 25, 50, 75, 98, 100])
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'rh_values': dict of RH heights in meters above ground
        - 'rh_elevations': dict of RH absolute elevations in meters
        - 'cumulative_energy': array of cumulative energy values
    """
    # Create height array relative to the ground position
    # Assuming ground is at index 0 after standardization
    heights_above_ground = np.arange(len(waveform)) * bin_size
    
    # Calculate cumulative energy (from ground up)
    total_energy = np.sum(waveform)
    if total_energy == 0:
        return None  # Handle empty waveform case
        
    cumulative_energy = np.cumsum(waveform) / total_energy * 100
    
    # Calculate RH metrics for requested percentiles
    rh_values = {}
    rh_elevations = {}
    
    for p in percentiles:
        if p == 0:
            # RH0 is always at ground level
            rh_values[f'rh{p}'] = 0
            rh_elevations[f'rh{p}'] = ground_elevation
        elif p == 100:
            # RH100 is the maximum height with energy return
            # Find the last bin with non-zero energy
            non_zero_indices = np.where(waveform > 0)[0]
            if len(non_zero_indices) > 0:
                max_idx = non_zero_indices[-1]
                rh_values[f'rh{p}'] = heights_above_ground[max_idx]
                rh_elevations[f'rh{p}'] = ground_elevation + heights_above_ground[max_idx]
            else:
                rh_values[f'rh{p}'] = 0
                rh_elevations[f'rh{p}'] = ground_elevation
        else:
            # Find height at which cumulative energy equals p%
            # Use linear interpolation for more precise estimation
            idx = np.searchsorted(cumulative_energy, p)
            if idx == 0:
                height = heights_above_ground[0]
            elif idx >= len(waveform):
                height = heights_above_ground[-1]
            else:
                # Linear interpolation between adjacent bins
                prev_cum = cumulative_energy[idx-1]
                next_cum = cumulative_energy[idx]
                prev_height = heights_above_ground[idx-1]
                next_height = heights_above_ground[idx]
                
                # Interpolate height
                weight = (p - prev_cum) / (next_cum - prev_cum) if next_cum > prev_cum else 0
                height = prev_height + weight * (next_height - prev_height)
                
            rh_values[f'rh{p}'] = height
            rh_elevations[f'rh{p}'] = ground_elevation + height
    
    return {
        'rh_values': rh_values,  # Heights above ground
        'rh_elevations': rh_elevations,  # Absolute elevations
        'cumulative_energy': cumulative_energy  # For visualization/debugging
    }

# Example usage:
if __name__ == "__main__":
    # Create a sample waveform (gaussian shape centered at bin 100)
    x = np.arange(512)
    sample_waveform = np.exp(-0.5 * ((x - 100) / 30)**2)
    ground_elevation = 500.0  # meters above sea level
    
    results = calculate_rh_metrics(sample_waveform, ground_elevation)
    
    # Print results
    print("RH metrics (meters above ground):")
    for k, v in results['rh_values'].items():
        print(f"{k}: {v:.2f}m")
    
    print("\nRH elevations (meters above sea level):")
    for k, v in results['rh_elevations'].items():
        print(f"{k}: {v:.2f}m")