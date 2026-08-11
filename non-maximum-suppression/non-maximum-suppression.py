def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression.
    """
    if not boxes:
        return []

    # Sort original indices by score descending
    indices = sorted(
        range(len(boxes)),
        key=lambda i: scores[i],
        reverse=True
    )

    kept = []

    while indices:
        current = indices.pop(0)
        kept.append(current)

        x1, y1, x2, y2 = boxes[current]
        area_current = max(0, x2 - x1) * max(0, y2 - y1)

        remaining = []

        for idx in indices:
            bx1, by1, bx2, by2 = boxes[idx]

            # Intersection
            inter_x1 = max(x1, bx1)
            inter_y1 = max(y1, by1)
            inter_x2 = min(x2, bx2)
            inter_y2 = min(y2, by2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection = inter_w * inter_h

            # Area of other box
            area_other = max(0, bx2 - bx1) * max(0, by2 - by1)

            union = area_current + area_other - intersection

            iou = intersection / union if union > 0 else 0.0

            # Keep boxes whose IoU is below threshold
            if iou < iou_threshold:
                remaining.append(idx)

        indices = remaining

    return kept