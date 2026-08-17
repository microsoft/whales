# Releasing

This project uses Semantic Versioning and annotated `vMAJOR.MINOR.PATCH` Git tags. Releases are built by GitHub Actions and attached to GitHub Releases; they are not published to PyPI.

## Prepare a release

1. Create a branch from the latest `main`.
2. Update `whales/_version.py` with the next version.
3. Run the package checks:

   ```bash
   python -m build
   python -m twine check dist/*
   python -m pytest tests/test_packaging.py
   ```

4. Open and merge the version bump pull request after CI passes. Summarize the release changes in the pull request description; GitHub uses merged pull requests to generate release notes.

## Publish a release

Create the tag on the merged commit. The tag must exactly match the version in `whales/_version.py`.

```bash
git switch main
git pull --ff-only
VERSION=0.1.0  # Replace with the version from whales/_version.py.
git tag -a "v${VERSION}" -m "Release v${VERSION}"
git push origin "v${VERSION}"
```

The release workflow validates the version, builds the wheel and source distribution, creates SHA-256 checksums, and publishes the files with generated release notes. Verify the workflow and release page after pushing the tag.

Released tags and attached files must not be replaced. Publish a new patch version to correct a release.
