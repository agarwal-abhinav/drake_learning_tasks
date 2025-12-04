import numpy as np 

def is_stationary_and_close_2D(array, reference_point, stopping_threshold=1e-8, distance_threshold=8e-3,chunk_size=2):
    """
    Check if there is a chunk of points in the array where:
    1. All points in the chunk are close to each other (stationary).
    2. All points in the chunk are close to the reference point.

    Parameters:
        array (np.ndarray): Numpy array of shape (N, 2).
        reference_point (np.ndarray): Reference point of shape (2,).
        distance_threshold (float): Maximum distance to consider "close".
        chunk_size (int): Size of the chunk to check (default is 4).

    Returns:
        int: Start index of the valid chunk if found, -1 otherwise.
    """
    N = array.shape[0]
    
    # Iterate through chunks of size `chunk_size`
    for i in range(N - chunk_size + 1):
        chunk = array[i:i + chunk_size]  # Extract the chunk of points
        
        # Check if all points in the chunk are close to each other
        pairwise_distances = np.linalg.norm(chunk[:, None, :] - chunk[None, :, :], axis=-1)
        if np.all(pairwise_distances <= stopping_threshold):
            # Check if all points in the chunk are close to the reference point
            distances_to_reference = np.linalg.norm(chunk - reference_point, axis=1)
            if np.all(distances_to_reference <= distance_threshold):
                if i < 10: 
                    pass 
                else: 
                    return i  # Return the start index of the valid chunk
            
    return -1

def clip_start_end_idle(traj, eps=1e-5, keep_idle=2): 
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)

    moving = diffs > eps 

    if not moving.any(): 
        print("Warning: trajectory has no movement.")
        return traj[:min(keep_idle, len(traj))]
    
    moving_idx = np.flatnonzero(moving)

    first_move_diff_idx = moving_idx[0]
    last_move_diff_idx = moving_idx[-1]

    if first_move_diff_idx - keep_idle < 0: 
        first_move_sampling = 0
    else: 
        first_move_sampling = first_move_diff_idx - keep_idle

    last_move_sampling = last_move_diff_idx + 1 + keep_idle

    return first_move_sampling, last_move_sampling