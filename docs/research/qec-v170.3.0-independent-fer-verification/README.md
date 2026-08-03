# QEC v170.3.0 Independent FER Verification

Formal publication source and Zenodo metadata for the independent verification
record of the QEC v170.3.0 exact ququart frame-error-rate and harmonic receiver
battery.

## Verified result

- pinned commit: `dada8b7a20a75753db43acc01a6a9e723ebaa6b6`
- full-depth checkout confirmed
- full pytest suite: `19172 passed, 0 failed, 4 skipped, 4 warnings`
- focused FER battery: `18 passed, 0 failed`
- exact basis: `16^5 = 1,048,576` packed Pauli patterns
- Wilson interval coverage: `38/40`
- harmonic adversarial battery: `375` evaluations, `0` false accepts
- receiver false-trust events: `0`

## Repository publication files

- `REPORT.md` - formal mathematical report source
- `zenodo_metadata.json` - ready-to-copy Zenodo record metadata
- `CITATION.cff` - citation metadata
- `verification_receipt.json` - machine-readable audit result
- `ZENODO_UPLOAD_GUIDE.md` - publication instructions
- `ZENODO_UPLOAD_SHA256.txt` - checksum of the complete prebuilt package

The complete Zenodo upload ZIP contains the rendered PDF, LaTeX and Markdown
sources, figures, all 15 evidence artifacts, raw logs, provenance receipts,
metadata, scripts, and package-wide SHA-256 manifests.

Prebuilt package SHA-256:

```text
8f1acd545ab56364ec9389ffaad2068983216f5fc616fd0f33bf3857602f11a9
```

## Scope boundary

This record establishes exact finite-code capacity evidence and deterministic
classical harmonic-receiver evidence. It does not claim a hardware threshold,
physical fault tolerance, quantum advantage, circuit-level performance,
leakage, SPAM, pulse fidelity, or physical device behavior.

License: MPL-2.0 under the repository root `LICENSE`.
