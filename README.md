# SPSA

Source-Preserving Selective Association for Multi-Object Tracking.

SPSA stores identity-labeled points during object interactions and uses their
votes to decide local box matches. The experiments use top-view fish tracking
as the application, with one YOLO11m detector shared across methods.

Author: Yvchong Liu, University of Science and Technology of China.
Contact: lyc2002@mail.ustc.edu.cn

## Setup

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
git clone https://github.com/ymzis69/HybridSORT external/HybridSORT
git -C external/HybridSORT checkout 396f8d30db13304c0cbaf1dcf2e16ded93ce1701
git clone https://github.com/JonathonLuiten/TrackEval external/TrackEval
git -C external/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

`SPSA_HYBRIDSORT` can point to another local checkout of the pinned host.
Upstream source is not modified. The loader normalizes two scalar prediction
scores for NumPy compatibility. The point and birth modules match the frozen
experimental source.

## Run Tracking

Supply sequential images named `000001.jpg`, `000002.jpg`, and so on.
The detection CSV has six columns:

```text
frame,x1,y1,x2,y2,confidence
```

Rows have no header. Coordinates are original-image pixel coordinates, not
normalized values. Keep low-score detections down to 0.05. Input can also be an
NPZ file containing `detections` with these six columns and scalar `frames`.

```bash
python run_tracking.py --images path/to/imgT --detections detections.csv --frames 2710 --output outputs/sequence.txt
```

Use the actual length of each sequence. Output is MOT format:
`frame,id,left,top,width,height,confidence,-1,-1,-1`.
The tracking command does not read ground truth. It reads adjacent frames only
when a point pool is active.

The default configuration uses margin 0.10, missing score 0.25, pool lifetime
40 steps, at most 80 points per identity, and birth score 0.85.
Decision-only controls are available through `--control no_ab` and
`--control no_dummy`.

## Evaluate

Ground truth must use MOT whole-object boxes:
`frame,id,left,top,width,height,mark,...`. Positive-area rows with nonzero mark
are evaluated. For 3D-ZeF, use the top-view whole-fish boxes. The official stereo
3D leaderboard is a different protocol.

Create a JSON mapping of sequence names to their frame lengths. Put predictions
in one directory as `<sequence>.txt`, and ground truth under
`<sequence>/gt/gt.txt`.

```bash
python evaluate.py --gt-root path/to/ground_truth --predictions outputs --lengths lengths.json --output outputs/metrics
```

Evaluation uses HOTA at 19 thresholds from 0.05 to 0.95, and CLEAR/Identity at
IoU 0.50. Sequence results are combined with TrackEval's `combine_sequences`.
The loader restores the numeric aliases required by the pinned TrackEval
version when running with NumPy 1.24 or newer.

## Tests and Results

```bash
python -m unittest test_persistent_lineage test_practical_ablation
```

`results/baseline_comparisons.csv` contains the paper's fixed comparison values.
`results/analysis_values.json` contains the component and margin plot values.
Training sequences are 01--02, development sequences are 03--04, and the fixed
evaluation is 05--08. The development sequences were used in method development.
No new version or parameter selection is made from 05--08.
Release validation is documented in `VALIDATION.md`.

Raw images, annotations, cached detections, and model weights are not
redistributed. Obtain 3D-ZeF from https://github.com/vapaau/3D-ZeF and follow its
license. Detector configuration is in `experiment_config.json`.

## Sources

The host is Hybrid-SORT by Yang et al. (AAAI 2024). Point containment follows
the line of work represented by NetTrack (Zheng et al., CVPR 2024).
SPSA uses Shi--Tomasi corners and Lucas--Kanade optical flow.
Dependencies retain their own licenses. See `NOTICE.md`.
