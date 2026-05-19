def iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) of two bounding boxes.

    Each box is represented as:
    [x1, y1, x2, y2]
    """

    # Intersection rectangle
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    # Compute intersection width and height
    inter_width = max(0.0, x_right - x_left)
    inter_height = max(0.0, y_bottom - y_top)

    # Intersection area
    intersection = inter_width * inter_height

    # Area of box_a
    area_a = max(0.0, box_a[2] - box_a[0]) * \
             max(0.0, box_a[3] - box_a[1])

    # Area of box_b
    area_b = max(0.0, box_b[2] - box_b[0]) * \
             max(0.0, box_b[3] - box_b[1])

    # Union area
    union = area_a + area_b - intersection

    # Avoid division by zero
    if union == 0:
        return 0.0

    return intersection / union