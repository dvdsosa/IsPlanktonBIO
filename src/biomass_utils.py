"""
Utility module for biomass and morphometric estimations.
"""

from typing import Any, Dict, Optional


def compute_biomass(
    species_label: str,
    area_pixels: float,
    biomass_params: Dict[str, Any],
    pixel_to_mm2_factor: float,
) -> Optional[float]:
    """
    Compute biomass using Lehette & Hernández-León (2009) formula.

    Biomass = a * (Area_mm2) ** b

    Args:
        species_label (str): The predicted taxonomy label of the species.
        area_pixels (float): The detected area of the specimen in pixels.
        biomass_params (dict): Parsed JSON dictionary containing species-specific parameters ('a' and 'b').
        pixel_to_mm2_factor (float): Hardware conversion factor from pixels to squared millimeters.

    Returns:
        Optional[float]: The computed biomass value in mg, or None if the species is not found in the parameters.
    """
    # Check whether the species is present in the biomass parameters JSON
    if species_label not in biomass_params:
        return None

    # Extract coefficients from the JSON
    a = biomass_params[species_label]["a"]
    b = biomass_params[species_label]["b"]

    # Convert area from pixels to square millimeters
    area_mm2 = area_pixels * pixel_to_mm2_factor

    # Apply the biomass formula
    biomass = a * (area_mm2**b)

    return biomass
