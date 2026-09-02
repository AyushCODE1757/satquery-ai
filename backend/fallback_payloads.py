"""
Pitch-ready fallback payloads for SatQuery AI.
Contains canned responses for all 4 task types:
1. single_image_vqa
2. visual_grounding
3. change_vqa
4. optical_sar_fusion
"""

FALLBACK_PAYLOADS = {
    "single_image_vqa": {
        "type": "final",
        "text": "Identified 4 solar farm arrays and 2 substation transformers in the target optical sector. Overall land utilization for clean energy production is estimated at 64%.",
        "geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [72.528, 23.032]
                    },
                    "properties": {
                        "label": "Primary Substation Transformer Location",
                        "confidence": 0.89
                    }
                }
            ]
        },
        "confidence": 0.89,
        "execution_summary": {
            "task": "single_image_vqa",
            "models_used": ["vqa_tool"],
            "params": {}
        }
    },

    "visual_grounding": {
        "type": "final",
        "text": "Located requested industrial storage facility in the northern quadrant of the satellite image. Bounding area highlighted on the map.",
        "geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [72.515, 23.035],
                            [72.525, 23.035],
                            [72.525, 23.045],
                            [72.515, 23.045],
                            [72.515, 23.035]
                        ]]
                    },
                    "properties": {
                        "label": "Target Industrial Storage Facility",
                        "confidence": 0.92
                    }
                }
            ]
        },
        "confidence": 0.92,
        "execution_summary": {
            "task": "visual_grounding",
            "models_used": ["grounding_tool"],
            "params": {}
        }
    },

    "change_vqa": {
        "type": "final",
        "text": "Bi-temporal change analysis detects 14.2% expansion in urban infrastructure between T1 and T2 imagery. Significant land clearing and structural development identified in the south-western quadrant.",
        "geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [72.505, 23.015],
                            [72.520, 23.015],
                            [72.520, 23.028],
                            [72.505, 23.028],
                            [72.505, 23.015]
                        ]]
                    },
                    "properties": {
                        "label": "New Urban Construction Zone (Expansion T1 -> T2)",
                        "confidence": 0.86
                    }
                }
            ]
        },
        "confidence": 0.86,
        "execution_summary": {
            "task": "change_vqa",
            "models_used": ["change_tool"],
            "params": {}
        }
    },

    "optical_sar_fusion": {
        "type": "final",
        "text": "Optical-SAR fusion analysis: Sentinel-2 optical imagery shows dense cloud cover obscuring 35% of the coastal zone. Sentinel-1 SAR radar backscatter pierces cloud cover, confirming active maritime vessel presence and structural shoreline anomaly.",
        "geojson": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [72.520, 23.020],
                            [72.540, 23.020],
                            [72.540, 23.040],
                            [72.520, 23.040],
                            [72.520, 23.020]
                        ]]
                    },
                    "properties": {
                        "source": "optical",
                        "label": "Optical (Sentinel-2): Cloud Obscured Sector",
                        "confidence": 0.75
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [72.522, 23.022],
                            [72.538, 23.022],
                            [72.538, 23.038],
                            [72.522, 23.038],
                            [72.522, 23.022]
                        ]]
                    },
                    "properties": {
                        "source": "sar",
                        "label": "SAR (Sentinel-1 VV/VH): Pierced Radar Vessel Footprint",
                        "confidence": 0.94
                    }
                }
            ]
        },
        "confidence": 0.91,
        "execution_summary": {
            "task": "optical_sar_fusion",
            "models_used": ["fusion_tool"],
            "params": {}
        }
    }
}
