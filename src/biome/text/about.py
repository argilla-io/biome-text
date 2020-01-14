__version__ = "0.3.0.dev"


def get_commit_hash() -> str:
    """
    Fetch current commit hash from configured git repository. The working folder should
    be part of a already git repository

    """
    try:
        import git

        repo = git.Git()
        return repo.log("--pretty=format:'%h'", "-n 1").replace("'", "")
    except Exception as e:
        print(e)
        return ""


if ".dev" in __version__:
    __version__ += get_commit_hash()
