# Publish Release

Tag and push to GitHub:
```shell
git clone git@github.com:fgnt/meeteval.git
cd meeteval
pip install --upgrade bump2version
bump2version --verbose --tag patch  # major, minor or patch
git push origin --tags
```

This will trigger a GitHub Action that will build and publish the package to PyPI.