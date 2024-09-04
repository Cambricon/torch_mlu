import os
import shutil
import warnings

__all__ = ["PathManager"]


class PathManager:
    DATA_FILE_AUTHORITY = 0o640
    DATA_DIR_AUTHORITY = 0o750

    @classmethod
    def check_path_owner_consistent(cls, path: str):
        """
        Function Description:
            check whether the path belong to process owner
        Parameter:
            path: the path to check
        Exception Description:
            when invalid path, prompt the user
        """

        if not os.path.exists(path):
            msg = f"The path does not exist: {path}"
            raise RuntimeError(msg)
        if os.stat(path).st_uid != os.getuid():
            warnings.warn(f"Warning: The {path} owner does not match the current user.")

    @classmethod
    def check_directory_path_writeable(cls, path):
        """
        Function Description:
            check whether the path is writable
        Parameter:
            path: the path to check
        Exception Description:
            when invalid data throw exception
        """
        cls.check_path_owner_consistent(path)
        if os.path.islink(path):
            msg = f"Invalid path is a soft chain: {path}"
            raise RuntimeError(msg)
        if not os.access(path, os.W_OK):
            msg = f"The path permission check failed: {path}"
            raise RuntimeError(msg)

    @classmethod
    def remove_path_safety(cls, path: str):
        msg = f"Failed to remove path: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if not os.path.exists(path):
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except Exception as err:
            raise RuntimeError(msg) from err

    @classmethod
    def make_dir_safety(cls, path: str):
        msg = f"Failed to make directory: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if os.path.exists(path):
            return
        try:
            os.makedirs(path, mode=cls.DATA_DIR_AUTHORITY, exist_ok=True)
        except Exception as err:
            raise RuntimeError(msg) from err

    @classmethod
    def create_file_safety(cls, path: str):
        msg = f"Failed to create file: {path}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if os.path.exists(path):
            return
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, cls.DATA_FILE_AUTHORITY))
        except Exception as err:
            raise RuntimeError(msg) from err
