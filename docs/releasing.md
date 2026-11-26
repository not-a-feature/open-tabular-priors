# Releasing to PyPI

## First release

1. Put the [release workflow](../.github/workflows/release.yml) on GitHub's
   default branch.
2. In your [PyPI publishing settings](https://pypi.org/manage/account/publishing/),
   add a pending GitHub Actions Trusted Publisher with these values.

   | Field | Value |
   |---|---|
   | PyPI project name | `open-tabular-priors` |
   | GitHub owner | `not-a-feature` |
   | Repository name | `open-tabular-priors` |
   | Workflow filename | `release.yml` |
   | Environment name | `pypi` |

3. Publish a GitHub Release tagged `v0.1.0` from the commit that contains the
   workflow. The first successful upload creates the PyPI project.

PyPI serves the source distribution and wheel publicly. To make the GitHub
README, figures and linked docs available to PyPI users, provide a public copy
of the repository or those assets.

## Later releases

Update `project.version` in `pyproject.toml` and run `uv lock`. Commit and push
both files, then publish a GitHub Release with the matching tag, for example
`v0.1.1`. Use a new version for each upload.

Each published GitHub Release builds a wheel and source distribution, checks
their metadata, and uploads both through Trusted Publishing. Draft releases
and tag pushes leave publishing pending until the GitHub Release is published.

See [PyPI’s Trusted Publisher setup guide](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
