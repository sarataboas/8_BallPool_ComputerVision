"""
External dataset audit utilities for safe integration with the original 8-Ball Pool dataset.

The external dataset (Pool Ball Detection.v5i.yolov11) uses a different class schema than
the original (8-Ball Pool.v3i.yolov11). These functions detect the incompatibilities and
visualise whether integration is safe.
"""

import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Class schemas (hardcoded — verified from data.yaml of each dataset)
# ---------------------------------------------------------------------------

ORIG_CLASSES = {0: "Black", 1: "Cue", 2: "Dot", 3: "Solid", 4: "Striped"}
# Dot (class 2) = rail marker — NOT a ball, excluded from count

EXT_CLASSES  = {0: "black", 1: "blue", 2: "cue_ball", 3: "yellow"}
# No rail markers in this dataset — every annotation IS a ball

ORIG_DOT_ID = 2   # class to EXCLUDE in original dataset
EXT_CUE_ID  = 2   # class 2 in external = cue_ball (must be COUNTED, not excluded)


# ---------------------------------------------------------------------------
# Counting functions
# ---------------------------------------------------------------------------

def count_balls_orig(label_path: Path) -> int:
    """Original counting rule: all annotations except class 2 (Dot = rail marker)."""
    path = Path(label_path)
    if not path.exists():
        return 0
    return sum(
        1 for ln in path.read_text().splitlines()
        if ln.strip() and int(ln.split()[0]) != ORIG_DOT_ID
    )


def count_balls_ext(label_path: Path) -> int:
    """
    External dataset counting rule: count ALL annotation lines.
    There are no rail markers in this dataset — every line is a ball.
    Using the original rule (exclude class 2) would silently drop the cue ball.
    """
    path = Path(label_path)
    if not path.exists():
        return 0
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    return len(lines)


def _count_balls_orig_applied_to_ext(label_path: Path) -> int:
    """What our pipeline would count if we naively applied it to external labels."""
    path = Path(label_path)
    if not path.exists():
        return 0
    return sum(
        1 for ln in path.read_text().splitlines()
        if ln.strip() and not ln.startswith(f"{ORIG_DOT_ID} ")
    )


# ---------------------------------------------------------------------------
# Audit 1 — Schema comparison
# ---------------------------------------------------------------------------

def audit_schema() -> None:
    """Print a side-by-side class schema comparison and flag the class 2 collision."""
    print("=" * 62)
    print("AUDIT 1 — Label Schema")
    print("=" * 62)
    print(f"\n{'ID':<4}  {'Original (8-Ball Pool v3)':<28}  {'External (PBD v5)':<20}")
    print("-" * 62)
    for cid in range(max(len(ORIG_CLASSES), len(EXT_CLASSES))):
        orig = ORIG_CLASSES.get(cid, "—")
        ext  = EXT_CLASSES.get(cid, "—")
        flag = " ← COLLISION" if cid == ORIG_DOT_ID else ""
        print(f"  {cid}    {orig:<28}  {ext:<20}{flag}")

    print()
    print("COLLISION on class 2:")
    print(f"  Original : class 2 = Dot  (rail pocket marker, EXCLUDED from count)")
    print(f"  External : class 2 = cue_ball  (a ball, MUST BE COUNTED)")
    print()
    print("Impact: if count_balls_from_label() is applied to external labels,")
    print("  every cue_ball annotation is silently dropped → off-by-1 on every frame.")
    print()
    print("Fix: use count_balls_ext() for external images (count all lines).")


# ---------------------------------------------------------------------------
# Audit 2 — Counting convention demonstration
# ---------------------------------------------------------------------------

def audit_counting_convention(ext_labels_dir: Path, n_samples: int = 10) -> None:
    """
    Show the effect of applying the wrong counting rule to external labels.
    Compares correct count vs. naive (buggy) count on n_samples files.
    """
    ext_labels_dir = Path(ext_labels_dir)
    label_files = sorted(ext_labels_dir.glob("*.txt"))[:n_samples]

    print("=" * 62)
    print("AUDIT 2 — Counting Convention")
    print("=" * 62)
    print(f"\n{'File':<55}  {'Correct':>7}  {'Naive':>7}  {'Delta':>5}")
    print("-" * 62)

    total_delta = 0
    for lf in label_files:
        correct = count_balls_ext(lf)
        naive   = _count_balls_orig_applied_to_ext(lf)
        delta   = correct - naive
        total_delta += delta
        flag = "  ← CUE DROPPED" if delta > 0 else ""
        print(f"  {lf.name[:52]:<52}  {correct:>7}  {naive:>7}  {delta:>+5}{flag}")

    print()
    print(f"  Every frame with a visible cue ball is undercounted by {total_delta // n_samples} using")
    print(f"  the original rule (assuming 1 cue ball per frame).")


# ---------------------------------------------------------------------------
# Audit 3 — Empty / missing labels
# ---------------------------------------------------------------------------

def audit_empty_labels(ext_dir: Path) -> dict:
    """
    Report images with missing or empty label files in the external dataset.
    Returns a dict with counts and lists of problematic paths.
    """
    ext_dir = Path(ext_dir)
    images_dir = ext_dir / "images"
    labels_dir = ext_dir / "labels"

    images = sorted(images_dir.glob("*.jpg"))
    missing, empty = [], []

    for img in images:
        lf = labels_dir / (img.stem + ".txt")
        if not lf.exists():
            missing.append(img)
        elif not lf.read_text().strip():
            empty.append(img)

    print("=" * 62)
    print("AUDIT 3 — Empty / Missing Labels")
    print("=" * 62)
    print(f"\n  Total images : {len(images)}")
    print(f"  Missing labels : {len(missing)}")
    print(f"  Empty labels   : {len(empty)}  ← treated as 0 balls if included naively")
    print()
    if empty:
        print(f"  First {min(5, len(empty))} empty-label files:")
        for p in empty[:5]:
            print(f"    {p.name}")
    if len(empty) > 5:
        print(f"    ... and {len(empty) - 5} more")
    print()
    print("  Recommendation: filter out images with empty labels before training.")

    return {"total": len(images), "missing": missing, "empty": empty}


# ---------------------------------------------------------------------------
# Audit 4 — Ball count distributions
# ---------------------------------------------------------------------------

def audit_distributions(
    orig_train_dir: Path,
    ext_train_dir: Path,
    filter_empty: bool = True,
) -> None:
    """
    Plot side-by-side count distributions for original and external datasets.
    Highlights the distribution mismatch that would bias training if unaddressed.
    """
    orig_train_dir = Path(orig_train_dir)
    ext_train_dir  = Path(ext_train_dir)

    orig_labels = orig_train_dir / "labels"
    ext_labels  = ext_train_dir  / "labels"

    orig_counts = [
        count_balls_orig(orig_labels / (p.stem + ".txt"))
        for p in sorted((orig_train_dir / "images").glob("*.jpg"))
    ]

    ext_counts_raw = [
        count_balls_ext(ext_labels / (p.stem + ".txt"))
        for p in sorted((ext_train_dir / "images").glob("*.jpg"))
    ]
    ext_counts = [c for c in ext_counts_raw if (not filter_empty or c > 0)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=False)

    def _bar(ax, counts, title, color):
        c = Counter(counts)
        keys = range(0, 17)
        ax.bar(keys, [c.get(k, 0) for k in keys], color=color, width=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Ball count")
        ax.set_ylabel("# images")
        ax.set_xticks(range(0, 17))
        ax.axvline(np.mean(counts), color='black', linestyle='--', linewidth=1.2,
                   label=f"mean={np.mean(counts):.1f}")
        ax.axvline(np.median(counts), color='grey', linestyle=':', linewidth=1.2,
                   label=f"median={np.median(counts):.0f}")
        ax.legend(fontsize=8)

    _bar(axes[0], orig_counts,
         f"Original dataset  (n={len(orig_counts)}, train only)", "steelblue")
    _bar(axes[1], ext_counts,
         f"External dataset  (n={len(ext_counts)}, empty labels excluded)", "darkorange")

    plt.suptitle("Ball count distributions — original vs external", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

    print("=" * 62)
    print("AUDIT 4 — Distribution Comparison")
    print("=" * 62)
    print(f"\n  Original : n={len(orig_counts):>4}  "
          f"range={min(orig_counts)}–{max(orig_counts)}  "
          f"mean={np.mean(orig_counts):.1f}  median={int(np.median(orig_counts))}")
    print(f"  External : n={len(ext_counts):>4}  "
          f"range={min(ext_counts)}–{max(ext_counts)}  "
          f"mean={np.mean(ext_counts):.1f}  median={int(np.median(ext_counts))}")

    merged = orig_counts + ext_counts
    print(f"  Merged   : n={len(merged):>4}  "
          f"range={min(merged)}–{max(merged)}  "
          f"mean={np.mean(merged):.1f}  median={int(np.median(merged))}")

    orig_frac = sum(1 for c in orig_counts if c >= 13) / len(orig_counts)
    ext_frac  = sum(1 for c in ext_counts  if c >= 13) / len(ext_counts)
    print(f"\n  High-count images (≥13 balls): orig={orig_frac:.0%}  ext={ext_frac:.0%}")
    low_frac_ext = sum(1 for c in ext_counts if c <= 4) / len(ext_counts)
    print(f"  Low-count  images (≤4 balls) : orig=0%  ext={low_frac_ext:.0%}")
    print()
    print("  Note: the external dataset has a bimodal distribution (early-game")
    print("  low-count + late-game high-count), while the original is concentrated")
    print("  around 8–15. Merging adds both useful variety AND a low-count spike.")


# ---------------------------------------------------------------------------
# Audit 5 — Temporal duplicates (video frame clustering)
# ---------------------------------------------------------------------------

def audit_temporal_duplicates(ext_images_dir: Path) -> None:
    """
    Detect video frame clusters from filenames.
    Roboflow video exports follow the pattern: <video_name>_<frame_id>_jpg.rf.<hash>.jpg
    Frames from the same video clip cover nearly identical table states.
    """
    ext_images_dir = Path(ext_images_dir)
    images = sorted(ext_images_dir.glob("*.jpg"))

    # Extract the video stem (everything before the frame counter)
    video_re = re.compile(r"^(.+?_mp4)-(\d+)_jpg")
    clip_re  = re.compile(r"^(.+?_mp4-\d+)_jpg")

    video_groups = Counter()
    clip_groups  = Counter()

    for img in images:
        vm = video_re.match(img.stem)
        if vm:
            video_groups[vm.group(1)] += 1

        cm = clip_re.match(img.stem)
        if cm:
            clip_groups[cm.group(1)] += 1

    print("=" * 62)
    print("AUDIT 5 — Temporal Duplicates (Video Frame Clustering)")
    print("=" * 62)
    print(f"\n  Total images           : {len(images)}")
    print(f"  Unique video sources   : {len(video_groups)}")
    print(f"  Unique clips (video+time): {len(clip_groups)}")
    if video_groups:
        max_frames = max(video_groups.values())
        avg_frames = np.mean(list(video_groups.values()))
        print(f"  Max frames per video   : {max_frames}")
        print(f"  Avg frames per video   : {avg_frames:.1f}")
        print()
        print(f"  Top 5 most-represented videos:")
        for name, count in video_groups.most_common(5):
            print(f"    {name[:50]:<50}  {count:>4} frames")
    else:
        print("\n  No video frame pattern detected — images appear to be standalone.")
    print()
    print("  Note: frames from the same video share near-identical table state.")
    print("  High frame counts per video inflate the effective dataset size")
    print("  without adding diverse ball configurations.")


# ---------------------------------------------------------------------------
# Audit 6 — Sample images (visual / perspective check)
# ---------------------------------------------------------------------------

def audit_sample_images(
    orig_train_dir: Path,
    ext_train_dir: Path,
    n: int = 6,
    seed: int = 42,
) -> None:
    """
    Show n sample images from each dataset side by side with their ball counts.
    Allows visual inspection of camera perspective, table colour, and image quality.
    """
    import random
    random.seed(seed)

    orig_train_dir = Path(orig_train_dir)
    ext_train_dir  = Path(ext_train_dir)

    orig_images = random.sample(sorted((orig_train_dir / "images").glob("*.jpg")), n)
    ext_images  = random.sample(sorted((ext_train_dir  / "images").glob("*.jpg")), n)

    orig_labels = orig_train_dir / "labels"
    ext_labels  = ext_train_dir  / "labels"

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6))
    fig.suptitle("Sample images — original (top) vs external (bottom)", fontsize=12)

    for col, img_path in enumerate(orig_images):
        count = count_balls_orig(orig_labels / (img_path.stem + ".txt"))
        img   = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"orig · {count} balls", fontsize=8)
        axes[0, col].axis("off")

    for col, img_path in enumerate(ext_images):
        count = count_balls_ext(ext_labels / (img_path.stem + ".txt"))
        img   = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        axes[1, col].imshow(img)
        axes[1, col].set_title(f"ext · {count} balls", fontsize=8)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Integration — copy valid external images into training split
# ---------------------------------------------------------------------------

def integrate_external(
    ext_train_dir: Path,
    orig_train_dir: Path,
    prefix: str = "ext_",
    dry_run: bool = False,
) -> dict:
    """
    Copies valid external training images (non-empty labels) into the original
    training directory with a filename prefix.

    Each external image gets a synthetic label file with exactly N lines of
    "0 0.5 0.5 1.0 1.0" (one per ball). This makes the label readable by the
    original count_balls_from_label() without modification: class 0 != 2, so
    none are excluded, and the line count equals the true ball count.

    Bbox coordinates (0.5, 0.5, 1.0, 1.0) are dummies — only the line count
    is used during training, never the spatial position.

    Args:
        ext_train_dir: path to external dataset train/ split
        orig_train_dir: path to original dataset train/ split (destination)
        prefix: filename prefix to mark external images (default "ext_")
        dry_run: if True, report what would happen without writing anything

    Returns:
        dict with counts: added, skipped_empty, already_present
    """
    ext_train_dir  = Path(ext_train_dir)
    orig_train_dir = Path(orig_train_dir)

    ext_images_dir  = ext_train_dir  / "images"
    ext_labels_dir  = ext_train_dir  / "labels"
    orig_images_dir = orig_train_dir / "images"
    orig_labels_dir = orig_train_dir / "labels"

    added, skipped_empty, already_present = 0, 0, 0

    for src_img in sorted(ext_images_dir.glob("*.jpg")):
        src_lbl    = ext_labels_dir / (src_img.stem + ".txt")
        ball_count = count_balls_ext(src_lbl)

        if ball_count == 0:
            skipped_empty += 1
            continue

        dst_name = prefix + src_img.name
        dst_img  = orig_images_dir / dst_name
        dst_lbl  = orig_labels_dir / (dst_name[:-4] + ".txt")

        if dst_img.exists():
            already_present += 1
            continue

        if not dry_run:
            shutil.copy2(src_img, dst_img)
            dst_lbl.write_text(
                "\n".join("0 0.5 0.5 1.0 1.0" for _ in range(ball_count)) + "\n"
            )

        added += 1

    if dry_run:
        print(f"[DRY RUN] Would add {added} images, "
              f"skip {skipped_empty} empty-label, "
              f"skip {already_present} already present")
    else:
        print(f"Integration complete:")
        print(f"  Added           : {added} images + synthetic labels")
        print(f"  Skipped (empty) : {skipped_empty} images")
        print(f"  Already present : {already_present} images (no-op)")
        print(f"  Total train now : {len(list(orig_images_dir.glob('*.jpg')))} images")

    return {"added": added, "skipped_empty": skipped_empty, "already_present": already_present}


def remove_external(orig_train_dir: Path, prefix: str = "ext_") -> int:
    """
    Removes all prefixed external images (and their labels) from the training directory.
    Use this to revert integration without touching original files.
    """
    orig_train_dir = Path(orig_train_dir)
    images_dir = orig_train_dir / "images"
    labels_dir = orig_train_dir / "labels"

    removed = 0
    for img in sorted(images_dir.glob(f"{prefix}*.jpg")):
        img.unlink()
        lbl = labels_dir / (img.stem + ".txt")
        if lbl.exists():
            lbl.unlink()
        removed += 1

    print(f"Removed {removed} external image+label pairs.")
    print(f"Train images remaining: {len(list(images_dir.glob('*.jpg')))}")
    return removed
