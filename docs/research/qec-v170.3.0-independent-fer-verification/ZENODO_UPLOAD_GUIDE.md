# Zenodo upload guide

## Recommended resource type

Select **Publication -> Report**. The PDF is the primary object; the evidence and source files are supporting material.

## Recommended title

`QEC v170.3.0 Independent Verification Package: Exact Ququart Frame-Error Rate and Harmonic Receiver Evidence`

## Creator

- Slade, Trent - Independent Researcher, QSOL-IMC

## Computational contributors

- Monica AI - autonomous independent executor / data collector
- OpenAI GPT-5.6 Thinking - independent evidence reviewer and formalization

## Version and date

- Version: `1.0.0`
- Publication date: `2026-08-03`

## License

- `MPL-2.0`

## Files to upload

1. `QEC_v170.3.0_Independent_FER_Verification_Report_v1.0.0.pdf`
2. `QEC_v170.3.0_Independent_FER_Verification_v1.0.0.zip`
3. `ZENODO_UPLOAD_SHA256.txt`

The standalone PDF gives Zenodo a useful preview. The ZIP preserves the complete research object, including source, figures, evidence, provenance, and metadata.

## DOI handling

Zenodo can assign the DOI at publication. To print the DOI inside the PDF, reserve a DOI in the Zenodo draft first, add it to the Markdown metadata, rebuild the PDF, regenerate checksums, and replace the files in the draft before publishing.

## Final check

- verify the ZIP against `ZENODO_UPLOAD_SHA256.txt`;
- preview the PDF;
- confirm title, creator, date, version, and license;
- confirm the related GitHub release and commit links;
- publish only after the file set is final.
