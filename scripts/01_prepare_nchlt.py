from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from common import parse_nchlt_xml, audit_manifest

TRAIN_FULL = 56284
TRAIN = 50655
VALIDATION = 5629
TEST = 2829
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--make-internal-split", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trn_xml = args.dataset_root / "nchlt_nso/transcriptions/nchlt_nso.trn.xml"
    tst_xml = args.dataset_root / "nchlt_nso/transcriptions/nchlt_nso.tst.xml"

    train_full = parse_nchlt_xml(trn_xml, args.dataset_root)
    test = parse_nchlt_xml(tst_xml, args.dataset_root)

    print("TRAIN PARTITION AUDIT", audit_manifest(train_full))
    print("OFFICIAL TEST AUDIT", audit_manifest(test))

    if len(train_full) != TRAIN_FULL:
        raise ValueError(f"Expected {TRAIN_FULL} training-partition utterances, got {len(train_full)}")
    if len(test) != TEST:
        raise ValueError(f"Expected {TEST} test utterances, got {len(test)}")

    train_full.to_csv(args.output_dir / "train_full.csv", index=False)
    test.to_csv(args.output_dir / "test.csv", index=False)

    if args.make_internal_split:
        print("WARNING: exact original internal validation membership was not archived.")
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(train_full))
        rng.shuffle(idx)
        val_idx = idx[:VALIDATION]
        train_idx = idx[VALIDATION:]
        train_df = train_full.iloc[train_idx].reset_index(drop=True)
        val_df = train_full.iloc[val_idx].reset_index(drop=True)
        assert len(train_df) == TRAIN and len(val_df) == VALIDATION
        train_df.to_csv(args.output_dir / "train.csv", index=False)
        val_df.to_csv(args.output_dir / "validation.csv", index=False)
        print("Train:", len(train_df), "Validation:", len(val_df))


if __name__ == "__main__":
    main()
