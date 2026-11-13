# Pending

## Clean git history (remove the old large files from history)

The current project is one commit on top of the old history, so the old
project's large files (about 115 MB) are still stored in the repository's past
commits and are downloaded by every full `git clone`.

To start over with a single commit and no old history, the repository owner
runs this once (it permanently replaces `main`; everyone else must re-clone):

```bash
git checkout --orphan clean
```

```bash
git add -A
```

```bash
git commit -m "Lightweight single-script text recognition project"
```

```bash
git branch -M clean main
```

```bash
git push --force origin main
```

Not urgent - the project works the same either way.
