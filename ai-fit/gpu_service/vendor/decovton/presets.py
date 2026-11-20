"""Resolution presets and HF checkpoint subfolder conventions for DeCo-VTON."""

# (height, width)
RESOLUTION_PRESETS = {
    512: (512, 384),
    1024: (1024, 768),
}

_DATASET_DIRS = {
    "vitonhd": "VITON-HD",
    "dresscode": "DressCode",
}


def resolve_size(resolution=None, height=None, width=None):
    """Return (height, width) from either a preset or manual values.

    Exactly one of {resolution, (height and width)} may be given.
    With neither, defaults to the 512 preset.
    """
    manual = height is not None or width is not None
    if resolution is not None and manual:
        raise ValueError("Give either --resolution or --height/--width, not both.")
    if resolution is not None:
        if resolution not in RESOLUTION_PRESETS:
            raise ValueError(
                f"Unknown resolution preset {resolution}; choose from {sorted(RESOLUTION_PRESETS)}."
            )
        return RESOLUTION_PRESETS[resolution]
    if manual:
        if height is None or width is None:
            raise ValueError("Both --height and --width are required for manual size.")
        return (height, width)
    return RESOLUTION_PRESETS[512]


def resolve_subfolder(dataset_name, resolution):
    """Map (dataset, resolution) to the HF repo subfolder holding the UNet."""
    key = dataset_name.lower()
    if key not in _DATASET_DIRS:
        raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {sorted(_DATASET_DIRS)}.")
    return f"{_DATASET_DIRS[key]}-{resolution}/unet"
