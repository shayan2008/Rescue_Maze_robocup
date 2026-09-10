# Rescue Maze Robot — Team Kavosh

Everything behind our **RoboCupJunior Rescue Maze** entry: the vision model that finds hazard markers, the robot's circuit and mechanical design, the controller code, and the team description paper.

## Victim and hazard detection

`victim_detector_kits.py` is the perception pipeline. It runs a fine-tuned YOLO11 model over a live camera feed and decides — carefully — when a hazard marker is real enough to act on.

The problem it solves: a single confident frame is not enough. Reflections, motion blur and partial views all produce convincing false positives, and dropping a rescue kit in the wrong place costs points. So detection is staged:

1. **Inference** — YOLO11 (`best.pt`) at `IMGSZ = 640`, deliberately low `MODEL_PREDICT_CONF = 0.10` so nothing is discarded too early.
2. **Clustering** — overlapping boxes (`CLUSTER_IOU = 0.35`) are merged and their labels voted by summed confidence, so one marker seen three ways becomes one detection.
3. **Tracking** — a single smoothed track is maintained, tolerating short dropouts (`COUNT_DROPOUT_GRACE_SEC = 0.35`) rather than restarting on every missed frame.
4. **Dwell gate** — a marker only counts after `REQUIRED_SEE_SEC = 2.0` of continuous observation, with `LABEL_DOMINANCE_MIN = 0.80` label agreement and a per-class confidence floor (omega is held to 0.87, the class most often confused).
5. **Memory** — counted markers are remembered and cannot be recounted for `RECOUNT_AFTER_ABSENCE_SEC = 20.0`, which stops the robot scoring the same wall twice on a loop.
6. **Scoring** — classes map to kit counts (`phi` → 2, `psi` → 1, `omega` → 0) and then to points.

A live HUD shows totals, per-class counts and tracking state while the robot runs.

## Repository layout

| Path | Contents |
| --- | --- |
| `victim_detector_kits.py` | Detection, tracking and kit-counting pipeline |
| `best.pt`, `last.pt` | Trained YOLO11 weights |
| `Image processing code/` | Vision experiments and preprocessing |
| `Robot code/` | Robot controller firmware |
| `Robot Circuit/` | Schematic and wiring |
| `Robot Design/` | Mechanical design and CAD |
| `TDP & Video/` | Team description paper and competition run video |

## Running the detector

```bash
pip install ultralytics opencv-python
python victim_detector_kits.py
```

Set `CAM_INDEX` at the top of the file to select the camera. The model path defaults to `best.pt` in the repository root.

The dataset and training runs behind these weights live in [greek.v9i.yolov11](https://github.com/shayan2008/greek.v9i.yolov11).

## Team

Built with Team Kavosh (KAVOSH AI & Robotics Academy). Maintained by [Shayan Doroudiani](https://github.com/shayan2008).
