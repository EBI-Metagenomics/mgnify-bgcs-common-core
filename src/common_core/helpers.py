CLASS_MAP = {
    "NRPS": "NRP",
    "PKS": "Polyketide",
    "other": "Other",
    "ribosomal": "RiPP",
    "saccharide": "Saccharide",
    "terpene": "Terpene",
    "alkaloid": "Alkaloid",
}


def normalize_bgc_class_string(label: str) -> str | None:
    """
    Return the mapped class name for an exact string match.

    Parameters
    ----------
    label : str
        Input string to map.

    Returns
    -------
    str | None
        Mapped value if found, otherwise None.
    """
    return CLASS_MAP.get(label,label)
