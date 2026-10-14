<!-- SPDX-License-Identifier: MPL-2.0 -->
# CI validation boundaries

The repository permits only GitHub-created or enterprise-owned Actions. The
legacy Docker setup and third-party HTML action were rejected before any job
started. Neither the security policy nor the rejected action code is bypassed.

The Build and Test workflow installs the editable project with development
requirements and runs the complete test suite without `|| true`. Full Git history
supports the published decoder-preservation assertions. Optional lint/format
commands previously swallowed every failure and were not correctness evidence;
the job is now accurately named as a regression gate, not a lint gate.

The container gate uses the runner's installed Docker command. The image contains
source, tests, documentation and fixtures, but excludes Git metadata, dotenv files,
key files and local environments. CI mounts the credential-free checkout read-only
at `/app`, runs with the checkout owner’s UID/GID, and exercises this gate on
pull requests as well as main when running the complete container suite, because some repository
regression tests inspect Git history. Checkout uses `persist-credentials: false`.
The source checkout stays read-only; separate temporary writable HOME and artifact
mounts support existing mirror/discovery tests without changing source files.
The writable artifacts directory is first seeded with the checkout’s tracked
artifact fixtures, so the overlay cannot hide historical regression evidence.
No credentials are copied into the image. A stand-alone image without the declared
checkout is not sufficient to run history-dependent repository assertions.

`scripts/check_html_links.py` is a repository-owned, offline gate. It checks
tracked HTML documents, local `href`/`src` targets, fragment identifiers, duplicate
IDs and paths escaping the checkout. Its tests exercise failures as well as valid
links. The exact FER output filenames are explicitly recorded as generated
artifacts: the source-only HTML gate reports these as deferred, while Ququart FER
Pages requires all of them to exist after generation.

This gate does **not** validate browser rendering, full HTML conformance, JavaScript
execution or external HTTP link liveness. It does not claim coverage equivalent
to every option of the previously configured third-party HTML checker. The current
four tracked source pages have no external `href`/`src` links; future external
links are counted and reported as unchecked rather than silently accepted as live.

The existing Pages publishing behavior is unchanged. Changes to its declared
paths can still cause its existing main-branch deployment. New telecom labs stay
offline demonstrations; no live telecom adapter is activated by these workflows.

The opened-only Auto Assign workflow also used a disallowed third-party action.
It now uses GitHub’s own `actions/github-script` to assign the same sole maintainer,
EmergentMonk, on the same issue/PR-open events with the same permissions. It is an
administrative workflow, not a test result or proof gate.
