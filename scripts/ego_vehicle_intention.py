import pandas as pd
import numpy as np

def compute_turn_or_straight(ego_df, threshold=15):
    """
    Classifies the intention of the ego vehicle as 'Turn' or 'Straight'
    based on the change in heading (yaw) between consecutive timestamps.

    Args:
        ego_df (pd.DataFrame): DataFrame containing ego vehicle information
        threshold (float): The threshold (in degrees) to classify a turn. 
                            If the yaw change is greater than this, it's considered a "Turn."

    Returns:
        pd.Series: A series containing 'Turn' or 'Straight' for each timestamp
    """
    ego_df = ego_df.sort_values('timestamp_ns')  # Sort the DataFrame by timestamp
    yaw_diff = np.diff(ego_df['yaw'])  # Calculate the change in yaw (angle)

    # Convert yaw differences to degrees (assuming yaw is in radians)
    yaw_diff_deg = np.degrees(yaw_diff)

    # Classify as "Turn" if the yaw difference is greater than the threshold, else "Straight"
    intention = ['Straight']  # The first row doesn't have a previous value, so it's straight
    for diff in yaw_diff_deg:
        if abs(diff) > threshold:
            intention.append('Turn')
        else:
            intention.append('Straight')
    
    return pd.Series(intention)

# Load your ego vehicle data
ego_df = pd.read_csv('data/processed/ego_vehicle_yaw.csv')

# Calculate turn vs. straight classification
ego_df['intention'] = compute_turn_or_straight(ego_df)

# Save the updated dataframe
ego_df.to_csv('data/processed/ego_vehicle_with_intention.csv', index=False)

print("✅ Intention classification added to the dataset!")
