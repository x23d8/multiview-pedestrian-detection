"""Make a WaveTrackNew copy that the five training frameworks can read.

The labelling tool's WaveTrackNew export differs from Wildtrack/MultiviewX in ways that break training:
  * many annotations carry only ``position_m`` (continuous metres) and ``positionID`` is null;
  * ``annotations_positions/`` also holds ``*.review-backup`` and ``review_log.jsonl`` files;
  * the folder is large (images), and training writes ``gt.txt`` / ``drop_annotations/`` into the dataset root,
    which is read-only on Kaggle (/kaggle/input).

This script writes a clean dataset root: calibrations copied, annotations normalised so that
``positionID = col + row * 313`` with ``col, row = round(position_m / 0.025)`` for every person (position_m is
the authority, as in the label tool), and the images symlinked (default), copied, or skipped.

    python prepare_wavetrack.py --src F:\\thesis_dataset\\WaveTrackNew --dst ./Data/WaveTrackNew
"""
import argparse
import json
import os
import re
import shutil

CELL_M = 0.025
POS_WIDTH = 313   # ROI 7.82 m / 2.5 cm
POS_HEIGHT = 241  # ROI 6.02 m / 2.5 cm
NUM_CAM = 5


def link_or_copy_images(src, dst, mode):
    if mode == 'skip':
        return 'skipped'
    os.makedirs(dst, exist_ok=True)
    used = mode
    for cam in range(1, NUM_CAM + 1):
        source, target = os.path.join(src, f'C{cam}'), os.path.join(dst, f'C{cam}')
        if os.path.lexists(target):
            continue
        if mode == 'link':
            try:
                os.symlink(source, target, target_is_directory=True)
                continue
            except OSError as error:
                print(f'cannot symlink ({error}); copying images instead')
                used = 'copy'
        shutil.copytree(source, target)
    return used


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, help='WaveTrackNew root (annotations_positions, calibrations, Image_subsets)')
    parser.add_argument('--dst', required=True, help='output dataset root (must not be inside --src)')
    parser.add_argument('--images', choices=['link', 'copy', 'skip'], default='link')
    args = parser.parse_args()

    src, dst = os.path.abspath(os.path.expanduser(args.src)), os.path.abspath(os.path.expanduser(args.dst))
    if dst == src or dst.startswith(src.rstrip(os.sep) + os.sep):
        raise SystemExit('--dst must not be inside --src')
    for needed in ('annotations_positions', 'calibrations/calibration.json', 'Image_subsets'):
        if not os.path.exists(os.path.join(src, needed)):
            raise SystemExit(f'not a WaveTrackNew root, missing: {os.path.join(src, needed)}')

    shutil.copytree(os.path.join(src, 'calibrations'), os.path.join(dst, 'calibrations'), dirs_exist_ok=True)

    out_dir = os.path.join(dst, 'annotations_positions')
    os.makedirs(out_dir, exist_ok=True)
    frame_files = sorted(name for name in os.listdir(os.path.join(src, 'annotations_positions'))
                         if re.fullmatch(r'\d+\.json', name))
    people = filled = changed = outside = max_shift = 0
    for name in frame_files:
        with open(os.path.join(src, 'annotations_positions', name), encoding='utf-8') as stream:
            persons = json.load(stream)
        kept = []
        for person in persons:
            people += 1
            x_m, y_m = person['position_m']
            if not (0 <= x_m < POS_WIDTH * CELL_M and 0 <= y_m < POS_HEIGHT * CELL_M):
                outside += 1
                continue
            # round to the nearest cell, but a person on the ROI edge stays on the last cell
            col = min(int(round(x_m / CELL_M)), POS_WIDTH - 1)
            row = min(int(round(y_m / CELL_M)), POS_HEIGHT - 1)
            position_id = col + row * POS_WIDTH
            old = person.get('positionID')
            if old is None:
                filled += 1
            elif old != position_id:
                changed += 1
                max_shift = max(max_shift, abs(old % POS_WIDTH - col), abs(old // POS_WIDTH - row))
            person['positionID'] = position_id
            kept.append(person)
        with open(os.path.join(out_dir, name), 'w', encoding='utf-8') as stream:
            json.dump(kept, stream, indent=1)

    mode = link_or_copy_images(os.path.join(src, 'Image_subsets'), os.path.join(dst, 'Image_subsets'), args.images)
    print(f'frames: {len(frame_files)}  people: {people}  positionID filled: {filled}  '
          f'overwritten: {changed} (max shift {max_shift} cells)  outside ROI dropped: {outside}')
    print(f'images: {mode} -> {os.path.join(dst, "Image_subsets")}')
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
