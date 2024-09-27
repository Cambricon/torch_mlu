from typing import Dict
from .profiler_parser import ProfileData


def analyze_data(profiler_data_path: str, id2opinfo: Dict = {}):
    profile = ProfileData(profiler_data_path)
    profile.process(id2opinfo)
