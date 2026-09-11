# Visual output and documentation asset plan

Status: proposed implementation sequence.

## Objective

Use one collection of executable examples to support visual review, regression
tests, and release documentation. Keep generated outputs ignored in this library
repository; publish only a small, verified subset to the documentation website.

Existing tests inspect Matplotlib objects, Styler HTML/CSS, and hierarchy arrays,
but discard their outputs. An initial assessment found annotation collisions in
rotated interval plots despite passing tests, along with documentation claims that
do not match the recorded release's neutral error-table styling.

The dcmview media workflow provides the precedent: tracked scene definitions,
ignored review bundles, provenance verification, and selected assets copied into
the documentation repository.

## 1. Define examples and the artifact contract

- Add `validation/visuals/` with deterministic synthetic data, named example
  functions, and an explicit list of documentation examples.
- Cover interval plots, score distributions, neutral and ranked performance
  tables, error tables, and a Plotly rendering of compiled hierarchy data.
- Include difficult cases in the developer collection: long labels, many levels,
  missing categories, undefined intervals, wide tables, and rotated annotations.
- Exercise public APIs for documentation examples. Use constructed result objects
  only where necessary to isolate a difficult rendering case.
- Record each example's configuration and retain the exact generating code.

Start the documentation subset with six examples: default intervals, default
distributions, class-split distributions, neutral performance tables, ranked
performance tables, and error tables. Add the hierarchy example next.

Acceptance: every presentation surface has a representative example, and the
initial documentation subset is explicitly selected.

## 2. Build the local gallery and pytest capture

- Add a generator that writes to `artifacts/visuals/`; add that directory to
  `.gitignore`.
- Save Matplotlib images, standalone Styler HTML, interactive Plotly HTML, browser
  screenshots, and an HTML index linking outputs to their generating code.
- Add opt-in pytest capture using the same example functions. Save outputs before
  figure cleanup while retaining structural and numerical assertions.
- Keep ordinary test runs fast. Make any example-specific styling explicit; the
  gallery wrapper must not silently improve the apparent package defaults.

Acceptance: one command produces a browsable gallery, selected tests can retain
their outputs, and generated files remain outside Git.

## 3. Add reproducible rendering in CI

- Establish a dedicated rendering environment with pinned dependencies, browser,
  fonts, viewport, and image dimensions. Keep browser tooling out of library
  runtime dependencies and core imports.
- Run visual examples in a dedicated CI job and upload the gallery for review.
- Check required output completeness, successful HTML rendering, and expected
  labels and table content. Save provenance and output hashes.
- Begin with visual review and meaningful rendering checks. Add image-difference
  checks later for selected stable examples in the pinned environment, rather
  than requiring broad pixel equality across operating systems.

Acceptance: a pull request produces a downloadable gallery, and rendering
failures are visible independently of numerical test failures.

## 4. Improve presentation using the gallery

- Preserve the existing output as a baseline before changing styling.
- Address concrete defects in focused changes, beginning with rotated interval
  annotation collisions and readable labels for omitted levels.
- Review table spacing, wide-table behavior, legend placement, and readability at
  documentation widths. Preserve neutral defaults, categorical ordering, and
  statistical meaning.
- Correct existing documentation claims against its recorded stable release.
  Document later styling changes when the release containing them is synchronized.
- Add targeted regression coverage for each behavior or layout defect fixed.

Acceptance: the selected documentation examples are readable, explanations match
their outputs, and difficult cases remain available for future review.

## 5. Generate verified release assets

Extend the release workflow in this order:

1. Validate the source and build the wheel once.
2. Install that wheel in an isolated rendering environment that cannot import the
   library checkout accidentally.
3. Generate and verify the curated documentation examples.
4. Record the release tag, commit, wheel hash, example inputs, rendering
   environment, and output hashes in a manifest.
5. Publish the verified wheel and preserve the documentation bundle as a GitHub
   release asset.

Generation failures should block package publication. Artifact publication must
be retryable without rebuilding or changing the verified wheel. Local review
bundles may represent work in progress; release bundles must resolve to the exact
release source and package artifact.

Acceptance: a release produces a complete, traceable asset bundle from the same
wheel distributed to users.

## 6. Import assets into the documentation website

- Add a docs-side import and validation command that selects an explicit release
  bundle and verifies its provenance against `docs-source.json`.
- Keep generated files ignored in the library repository. Commit only the curated
  publication assets and their manifest in the documentation repository, following
  the dcmview pattern.
- Place outputs beside their generating code, with captions and meaningful
  alternative text. Retain selectable HTML for tables and static previews for
  interactive examples.
- Update affected prose, assets, and every release-provenance field together in
  the documentation synchronization pull request.
- Run the documentation repository's required validation and review the Cloudflare
  preview. Never publish unreleased checkout behavior as the stable API.

Acceptance: the site builds, asset provenance is checked automatically, and the
preview shows examples matching the documented release.

## Delivery sequence and validation

Stages 1–3 form the first implementation milestone: shared examples, an ignored
gallery, test capture, and CI artifacts. Stage 4 uses that evidence to improve
presentation. Stages 5–6 connect reviewed examples to releases and documentation.

For library changes, run the checks required by `CONTRIBUTING.md`: Ruff lint and
format checks, pytest, and a package build for packaging changes. Add regression
coverage for behavior changes. Verify that generated artifacts are ignored and
excluded from distributions, optional rendering dependencies remain outside core
imports, and caller-owned DataFrames are not mutated.

This document belongs to the library repository. Keep implementation plans out of
the published documentation repository; that repository owns release-aligned
tutorials and reference pages.
