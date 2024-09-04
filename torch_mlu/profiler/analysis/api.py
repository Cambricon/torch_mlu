import os
from .profiler_parser import ProfileData


def analyze_data(profiler_data_path: str):
    profile = ProfileData(profiler_data_path)
    profile.process()
