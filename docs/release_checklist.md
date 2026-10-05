# Publishing the prepared v1.0.0 release

This preparation is local only. No commit, tag, remote metadata change or release is authorized in this task. The working tree therefore remains dirty. Review the [packaging audit](release_audit.md), [artifact inventory](release_assets.md) and [release notes](release_notes_v1.0.0.md) before publishing.

## GitHub metadata

Repository: `Drivax/Dynamic-Credit-Limit-Optimization-with-RL-`.

In GitHub **About → Edit**, set the description to:

> Synthetic sequential credit-limit decision research with PPO, behavior-cloning initialization, distribution shift and off-policy evaluation.

Set these topics: `reinforcement-learning`, `ppo`, `credit-risk`, `risk-management`, `pomdp`, `simulation`, `offline-to-online-rl`, `off-policy-evaluation`, `stable-baselines3`, `python`.

Optional repository name: `dynamic-credit-limit-rl` (**Settings → General → Repository name**). Relative documentation links and package imports are independent of the repository name. If renamed, update the remote, the README clone URL and CI badge URLs, and `CITATION.cff`'s repository URL. The current links deliberately use the actual existing URL.

For **Settings → General → Social preview**, use a simple background, the project title, the subtitle “When does learned credit control earn its complexity?” and the existing incremental-value figure. No arbitrary generated image is included. On the profile page, use **Customize your pins** to pin the repository if desired.

## Release gate

- Review and commit the local changes under `Prepare v1.0.0 research release` after publication is authorized.
- Publish the commit and confirm the configured Ubuntu/Windows CI succeeds. Local results are not hosted CI evidence.
- Set `date-released` in `CITATION.cff` to the actual publication date and remove the unreleased wording from the changelog/release notes as part of the release commit.
- Verify `git status --short` is empty and inspect `git log -1 --oneline`.
- Verify the final scientific and packaging manifests correspond to the reviewed local files.
- Only then create and push the annotated tag:

```bash
git tag -a v1.0.0 -m "Final research release"
git push origin v1.0.0
```

Create a GitHub release from that tag with title `v1.0.0 — Final research release` and the prepared release notes. Do not claim unavailable model archives as assets. No remote rename or history rewrite is part of the prepared release.
