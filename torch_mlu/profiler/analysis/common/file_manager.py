import csv
import os.path

from . import utils
from .path_manager import PathManager

logger = utils.get_logger()

__all__ = []

MAX_FILE_SIZE = 1024 * 1024 * 1024 * 10
MAX_CSV_SIZE = 1024 * 1024 * 1024 * 5


class FileManager:
    @classmethod
    def create_csv_file(
        cls, output_path: str, data: list, file_name: str, headers: list = None
    ) -> None:
        if not data:
            return
        file_path = os.path.join(output_path, file_name)
        PathManager.make_dir_safety(output_path)
        PathManager.create_file_safety(file_path)
        PathManager.check_directory_path_writeable(file_path)
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            if headers:
                writer.writerow(headers)
            writer.writerows(data)
