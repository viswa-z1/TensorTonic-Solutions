import math

def roi_pool(feature_map, rois, output_size):
    """
    Apply ROI Pooling to extract fixed-size features.
    
    Args:
        feature_map (list of list of float): 2D feature map.
        rois (list of list of int): List of ROIs, each defined as [x1, y1, x2, y2].
        output_size (int): Target output size (output_size x output_size).
    
    Returns:
        list of list of list of float: Fixed-size feature maps for each ROI.
    """
    pooled_features = []
    H, W = len(feature_map), len(feature_map[0])  # Feature map dimensions
    
    for roi in rois:
        x1, y1, x2, y2 = roi
        roi_h = y2 - y1
        roi_w = x2 - x1
        
        # Initialize the pooled feature map for this ROI
        pooled = [[0.0 for _ in range(output_size)] for _ in range(output_size)]
        
        for i in range(output_size):
            # Compute vertical bin boundaries
            h_start = y1 + math.floor(i * roi_h / output_size)
            h_end = y1 + math.floor((i + 1) * roi_h / output_size)
            h_end = max(h_end, h_start + 1)  # Ensure at least one pixel per bin
            
            for j in range(output_size):
                # Compute horizontal bin boundaries
                w_start = x1 + math.floor(j * roi_w / output_size)
                w_end = x1 + math.floor((j + 1) * roi_w / output_size)
                w_end = max(w_end, w_start + 1)  # Ensure at least one pixel per bin
                
                # Perform max pooling within the bin
                max_value = float('-inf')
                for y in range(h_start, h_end):
                    for x in range(w_start, w_end):
                        if 0 <= y < H and 0 <= x < W:  # Ensure within bounds
                            max_value = max(max_value, feature_map[y][x])
                
                pooled[i][j] = max_value
        
        pooled_features.append(pooled)
    
    return pooled_features
