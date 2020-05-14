from datetime import datetime

__version__ = "1.0.0.rc0"

from typing import Optional


def package_version(version: str):
    """Generates a proper package version from git repository info"""

    def get_commit_hash(repository: git.Git) -> str:
        """
        Fetch current commit hash from configured git repository. The working folder should
        be part of a already git repository

        """
        return repository.log("--pretty=format:'%h'", "-n 1").replace("'", "")

    def get_first_tag_for_commit(repository: git.Git, commit_hash: str) -> str:
        """ Return tags related to current commit """

        tags = (
            repository.tag("--contains", commit_hash)
            if commit_hash
            else repository.tag("--contains")
        ).split("\n")
        return tags[0]

    def get_version_from_branch(repository: git.Git) -> Optional[str]:
        """Extract version from release branch name"""
        current_branch = repository.rev_parse("--abbrev-ref", "HEAD")

        dropped_releases = current_branch.replace("releases/", "")
        if current_branch == dropped_releases:
            return None
        return dropped_releases

    def version_matches(release_branch_version:str, configured_version:str)->bool:
        """Checks if a version matches with related release branch version"""
        minor_release = tuple(release_branch_version.split(".")[:2])
        minor_configured = tuple(configured_version.split(".")[:2])

        return minor_release == minor_configured

    try:
        repo = git.Git()
    except Exception:  # pylint: disable=broad-except
        return version

    commit = get_commit_hash(repo)
    repo_tag = get_first_tag_for_commit(repo, commit)
    release_version = get_version_from_branch(repo)
    if release_version and not repo_tag:
        assert version_matches(release_version, version)
        return version
    today = datetime.today().strftime("%Y%m%d%H%M%S")

    return repo_tag if repo_tag else f"{version}.{today}+{commit}"


try:
    import git

    __version__ = package_version(__version__)
# pylint: disable=broad-except
except Exception:
    pass
