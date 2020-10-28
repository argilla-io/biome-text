from datetime import datetime

__version__ = "1.1.0rc0"

from typing import Optional


def package_version(version: str) -> str:
    """Generates a proper package version from the git repository info

    Parameters
    ----------
    version

    Returns
    -------
    package_version
    """

    def get_commit_hash(repository: git.Git) -> str:
        """
        Fetch current commit hash from the configured git repository.
        The working directory should already be part of a git repository.

        Parameters
        ----------
        repository

        Returns
        -------
        commit_hash
        """
        return repository.log("--pretty=format:'%h'", "-n 1").replace("'", "")

    def get_first_tag_for_commit(repository: git.Git, commit_hash: str) -> str:
        """Return tags related to current commit

        Parameters
        ----------
        repository
        commit_hash

        Returns
        -------
        tag
        """
        tags = (
            repository.tag("--contains", commit_hash)
            if commit_hash
            else repository.tag("--contains")
        ).split("\n")
        return tags[0]

    def get_version_from_branch(repository: git.Git) -> Optional[str]:
        """Extract version from release branch name

        Parameters
        ----------
        repository

        Returns
        -------
        version
        """
        current_remote_branch = repository.branch("--remote", "--contains")
        if "releases/" in current_remote_branch:
            return current_remote_branch.split("/")[-1]
        return None

    def version_matches(release_branch_version: str, configured_version: str) -> bool:
        """Checks if a version matches the related release branch version

        Parameters
        ----------
        release_branch_version
        configured_version

        Returns
        -------
        bool
        """
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

    return repo_tag if repo_tag else f"{version}.dev+{commit}"


try:
    import git

    __version__ = package_version(__version__)
# pylint: disable=broad-except
except Exception:
    pass
