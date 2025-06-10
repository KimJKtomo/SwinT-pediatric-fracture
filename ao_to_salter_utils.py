# ao_to_salter_utils.py
import re

# AO subtype to Salter-Harris type
AO_TO_SALTER = {
    'E/1.1': 'Salter I',
    'E/2.1': 'Salter II',
    'E/3.1': 'Salter III',
    'E/4.1': 'Salter IV',
    'M/2.1': 'Torus',
    'M/3.1': 'Simple Complete',
    '/7': 'Styloid'
}

def extract_ao_subtypes(ao_string):
    """
    Extract valid AO subtype strings from raw AO classification input.
    e.g., "23r-E/3.1, 23u-M/2.1" → ['E/3.1', 'M/2.1']
    """
    if not isinstance(ao_string, str):
        return []

    pattern = r'(\d{2}[ru]?-([EM])/\d\.\d|/\d)'  # capture E/M with subtype, or isolated /7
    matches = re.findall(pattern, ao_string)
    ao_subs = []

    for full, region in matches:
        if '-' in full:
            _, sub = full.split('-', 1)  # "E/3.1"
            ao_subs.append(sub)
        else:
            ao_subs.append(full)  # /7

    return ao_subs

def is_growth_plate_fracture(ao_sub):
    """성장판 골절 여부"""
    return ao_sub.startswith("E/")

def ao_to_salter_label(ao_sub):
    """Salter-Harris type 이름"""
    return AO_TO_SALTER.get(ao_sub, "Unknown")
