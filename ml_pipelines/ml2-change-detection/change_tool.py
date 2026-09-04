import cv2
import numpy as np


def change_tool(image1, image2, threshold_value=30, min_area=5):
    """
    Compare two RGB/BGR images and return changed regions as GeoJSON.
    """

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Structural difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold
    _, change_mask = cv2.threshold(
        diff,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    # Find changed regions
    contours, _ = cv2.findContours(
        change_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    features = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Remove tiny noise
        if area < min_area:
            continue

        points = contour.reshape(-1, 2).tolist()

        # Close polygon
        if points[0] != points[-1]:
            points.append(points[0])

        features.append({
            "type": "Feature",
            "properties": {
                "area": float(area)
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [points]
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }