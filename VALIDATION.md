# Release Validation

Checks completed on 2026-11-08.

- All 25 mechanism tests pass.
- The release runner replayed the first 600 frames of development sequence
  ZebraFish-03 using the original cached detector outputs. All 1,200 MOT rows
  match the saved experimental predictions exactly, including box coordinates
  and identities.
- The release evaluator processed the saved predictions for development
  sequences 03--04. TrackEval sequence combination gives HOTA
  90.29170109327089, DetA 90.15428742775046, AssA 90.43036457111742,
  IDF1 99.5070248952428, and IDSW 0.
- The point-source, birth, and decision-control modules are byte-identical
  to the frozen experimental files.
- No new inference or parameter selection used evaluation sequences 05--08.

The 600-frame replay checks packaging fidelity; it is not a new full-sequence
benchmark. The development metric check evaluates existing predictions.
Images, ground truth, cached detections, and weights remain outside this release.

## Frozen Module Hashes

SHA-256:

```text
persistent_lineage.py
c15fcc1a67781bcff17ed53d1e28cfe6b112b63b9bac363c15a1360b8418e7de
compositional_birth.py
58ef437d58b2addd8349683bcd46a9286d9ba2f16fc59d9acd89c083aefa88c3
practical_lineage_ablation.py
b249b73516f78c456269c0e7bebc0ba089641db18ccc9f28ce4b8f281ced7aa2
```
