"""Check for latest release."""

from __future__ import annotations

import typing as ty

import requests
from loguru import logger

from koyo.timer import format_datetime_ago


class DownloadDict(ty.TypedDict):
    """Download dict."""

    filename: str
    version: str
    download_url: str


LatestVersion = dict[str, DownloadDict]


def get_target() -> str | None:
    """Get target based on the platform."""
    import platform

    if platform.system() == "Windows":
        return "win_amd64"
    elif platform.system() == "Darwin":
        if platform.processor() == "arm":
            return "macosx_arm64"
        return "macosx_x86_64"
    return None


def get_latest_release(user: str = "vandeplaslab", package: str = "koyo") -> str:
    """Get latest release from GitHub."""
    data = get_latest_git(user, package)
    if "tag_name" in data:
        return data["tag_name"]
    return ""


def get_latest_git(user: str = "vandeplaslab", package: str = "koyo") -> dict[str, ty.Any]:
    """Get latest release from GitHub."""
    response = requests.get(f"https://api.github.com/repos/{user}/{package}/releases/latest")
    if response.status_code == 200:
        data = response.json()
        return data
    return {}


def format_version(data: dict) -> str:
    """Format version."""
    git_version = data.get("tag_name", "")
    if not git_version:
        return "No changelog information available."
    published_at = format_datetime_ago(data.get("published_at", ""))
    name = data.get("name", "")
    body = data.get("body", "")
    return f"# {name}\n**Published:** {published_at}\n**Change log**\n\n{body}"


def is_new_version_available(
    current_version: str, user: str = "vandeplaslab", package: str = "koyo", data: dict[str, ty.Any] | None = None
) -> tuple[bool, str]:
    """Check whether there is a new version available."""
    import requests.exceptions
    from packaging import version

    # get latest version from GitHub
    if data is None:
        data = {}
        try:
            data = get_latest_git(user, package)
        except requests.exceptions.ConnectionError:
            logger.trace("There was a connection error - could not retrieve information.")
        except requests.exceptions.RequestException:
            logger.trace("There was some problem in retrieving information.")
        except Exception:
            logger.trace("Another exception occur - not sure how to handle this!")

    git_version = data.get("tag_name", "")
    if not git_version:
        return False, "Could not retrieve latest version."
    new_available = version.parse(git_version) > version.parse(current_version)
    if new_available:
        published_at = format_datetime_ago(data.get("published_at", ""))
        return True, (
            f"New version available: <b>{git_version}</b> made available <b>{published_at}</b>."
            f" You are using version <b>{current_version}</b>"
        )
    return False, f"You are using the latest version: '{current_version}'."
