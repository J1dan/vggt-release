from typing import List, Dict, Any

def classify_raw_steps_from_trajectory(
    trajectory: List[Dict[str, Any]], 
    yaw_thresh_deg: float = 2.0, 
    forward_thresh_m: float = 0.012
) -> List[str]:
    """
    Classify a sequence of trajectory steps into discrete motion labels.
    
    Args:
        trajectory: List of dicts containing 'yaw_deg' and 'forward_m'
        yaw_thresh_deg: Threshold for yaw to be considered a turn
        forward_thresh_m: Threshold for forward motion to be considered moving forward
        
    Returns:
        List of string labels ('TL', 'TR', 'F', 'S')
    """
    labels = []
    for step in trajectory:
        yaw = step.get('yaw_deg', 0.0)
        forward = step.get('forward_m', 0.0)
        
        if abs(yaw) > yaw_thresh_deg:
            labels.append("TL" if yaw > 0 else "TR")
        elif forward > forward_thresh_m:
            labels.append("F")
        else:
            labels.append("S")
            
    return labels
