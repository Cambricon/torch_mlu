import argparse
import hashlib
import json
import logging
import os
import platform
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# String representing the host platform (e.g. Linux, Darwin).
HOST_PLATFORM = platform.system()
HOST_PLATFORM_ARCH = platform.system() + "-" + platform.processor()

# PyTorch directory root
try:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        check=True,
    )
    PYTORCH_ROOT = result.stdout.decode("utf-8").strip()
except subprocess.CalledProcessError:
    # If git is not installed, compute repo root as 3 folders up from this file
    path_ = os.path.abspath(__file__)
    for _ in range(4):
        path_ = os.path.dirname(path_)
    PYTORCH_ROOT = path_

DRY_RUN = False


def compute_file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file and return it as a hex string."""
    # If the file doesn't exist, return an empty string.
    if not os.path.exists(path):
        return ""

    hash = hashlib.sha256()

    # Open the file in binary mode and hash it.
    with open(path, "rb") as f:
        for b in f:
            hash.update(b)

    # Return the hash as a hexadecimal string.
    return hash.hexdigest()


def report_download_progress(
    chunk_number: int, chunk_size: int, file_size: int
) -> None:
    """
    Pretty printer for file download progress.
    """
    if file_size != -1:
        percent = min(1, (chunk_number * chunk_size) / file_size)
        bar = "#" * int(64 * percent)
        sys.stdout.write(f"\r0% |{bar:<64}| {int(percent * 100)}%")


def check(binary_path: Path, reference_hash: str) -> bool:
    """Check whether the binary exists and is the right one.

    If there is hash difference, delete the actual binary.
    """
    if not binary_path.exists():
        logging.info("%s does not exist.", binary_path)
        return False

    existing_binary_hash = compute_file_sha256(str(binary_path))
    if existing_binary_hash == reference_hash:
        return True

    logging.warning(
        """\
Found binary hash does not match reference!

Found hash: %s
Reference hash: %s

Deleting %s just to be safe.
""",
        existing_binary_hash,
        reference_hash,
        binary_path,
    )
    if DRY_RUN:
        logging.critical(
            "In dry run mode, so not actually deleting the binary. But consider deleting it ASAP!"
        )
        return False

    try:
        binary_path.unlink()
    except OSError as e:
        logging.critical("Failed to delete binary: %s", e)
        logging.critical(
            "Delete this binary as soon as possible and do not execute it!"
        )

    return False


def download_file_with_retry(url, binary_path, max_retries=5, timeout=3):
    """
    Download a file from a URL with retry logic.

    :param url: The URL of the file to download (can be a list [local_url, remote_url]).
    :param binary_path: The path where the downloaded file should be saved.
    :param max_retries: Maximum number of retries if the download fails.
    :param timeout: Timeout in seconds to wait between retries.
    """
    def download_file(target_url):
        """Helper function to download a file and handle exceptions."""
        try:
            urllib.request.urlretrieve(
                target_url,
                binary_path,
                reporthook=report_download_progress if sys.stdout.isatty() else None,
            )
            return True
        except Exception as e:
            print(f"Download from {target_url} failed with error: {e}")
            return False

    if isinstance(url, list):
        local_url, remote_url = url
    else:
        local_url, remote_url = None, url

    # Try downloading from local URL first if available
    if local_url:
        if download_file(local_url):
            return

    # If download from local URL fails or not available, try remote URL with retry logic
    if remote_url:
        print("Continue to download with remote URL...")
        for retry in range(max_retries + 1):
            if download_file(remote_url):
                return
            if retry < max_retries:
                print(f"Retrying in {timeout} seconds... ({retry + 1}/{max_retries})")
                time.sleep(timeout)
            else:
                print("Maximum retry limit reached, download failed.")
                raise Exception(f"Download from {remote_url} failed after {max_retries} retries.")


def download(
    name: str,
    output_dir: str,
    url: str,
    reference_bin_hash: str,
) -> bool:
    """
    Download a platform-appropriate binary if one doesn't already exist at the expected location and verifies
    that it is the right binary by checking its SHA256 hash against the expected hash.
    """
    # First check if we need to do anything
    binary_path = Path(output_dir, name)
    if check(binary_path, reference_bin_hash):
        logging.info("Correct binary already exists at %s. Exiting.", binary_path)
        return True

    # Create the output folder
    binary_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the binary
    logging.info("Downloading %s to %s", url, binary_path)

    if DRY_RUN:
        logging.info("Exiting as there is nothing left to do in dry run mode")
        return True

    download_file_with_retry(url, binary_path)

    logging.info("Downloaded %s successfully.", name)

    # Check the downloaded binary
    if not check(binary_path, reference_bin_hash):
        logging.critical("Downloaded binary %s failed its hash check", name)
        return False

    # Ensure that executable bits are set
    mode = os.stat(binary_path).st_mode
    mode |= stat.S_IXUSR
    os.chmod(binary_path, mode)

    logging.info("Using %s located at %s", name, binary_path)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="downloads and checks binaries from s3",
    )
    parser.add_argument(
        "--config-json",
        required=True,
        help="Path to config json that describes where to find binaries and hashes",
    )
    parser.add_argument(
        "--linter",
        required=True,
        help="Which linter to initialize from the config json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="place to put the binary",
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help="name of binary",
    )
    parser.add_argument(
        "--dry-run",
        default=False,
        help="do not download, just print what would be done",
    )

    args = parser.parse_args()
    if args.dry_run == "0":
        DRY_RUN = False
    else:
        DRY_RUN = True

    logging.basicConfig(
        format="[DRY_RUN] %(levelname)s: %(message)s"
        if DRY_RUN
        else "%(levelname)s: %(message)s",
        level=logging.INFO,
        stream=sys.stderr,
    )

    config = json.load(open(args.config_json))
    config = config[args.linter]

    # Allow processor specific binaries for platform (namely Intel and M1 binaries for MacOS)
    host_platform = HOST_PLATFORM if HOST_PLATFORM in config else HOST_PLATFORM_ARCH
    # If the host platform is not in platform_to_hash, it is unsupported.
    if host_platform not in config:
        logging.error("Unsupported platform: %s/%s", HOST_PLATFORM, HOST_PLATFORM_ARCH)
        sys.exit(1)

    url = config[host_platform]["download_url"]
    hash = config[host_platform]["hash"]

    ok = download(args.output_name, args.output_dir, url, hash)
    if not ok:
        logging.critical("Unable to initialize %s", args.linter)
        sys.exit(1)
