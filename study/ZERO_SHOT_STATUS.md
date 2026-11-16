# Zero-shot deepfake transfer

**Status: code and source checkpoints verified. Real target evaluation pending authorized media.** This stage reports no real-deepfake AUROC, average precision, balanced accuracy, or confusion counts.

FaceForensics++ requires an application and supplies the download script after acceptance [1]. The Celeb-DF v2 authors likewise release the download link after accepting an access request [2]. No authorized official held-out clips or pinned crops for either dataset were available in the current workspace. Existing metrics from other deepfake research tasks and the synthetic SR label fixtures cannot measure transfer for these models.

## Fixed source decisions

`source_run/SOURCE_DECISIONS.json` records the validation-selected compact CDCH logistic model, frozen source thresholds, and rank-fusion weights. `source_run/source_models.joblib` contains all statistical and native models, fitted transforms, rank-reference scores and thresholds. The archived RGB model and source manifest remain under `pilot/results/`. The exact source splits and models contain no deepfake examples.

Keep video-level AUROC for PNG face crops derived from FaceForensics++ c23 and Celeb-DF v2 as separate primary evaluations of the frequency and validation-selected CDCH models. Treat CDCH B, the other models, and fusion as exploratory comparisons. Report frame metrics, source/lineage-clustered intervals, fixed-threshold balanced accuracy and confusion counts. Use the dataset's official held-out splits and document sampling, tracking failures and ancestry overlap.

## Run the blind protocol

From the package root, initialize the environment as in the README. The authorized data operator should prepare `target/target_preprocessing.json` with decoder and face tracker versions and weight hashes, ROI and resize policy, fixed frame sampling rule and failure logging. Use at most 16 fixed sampled RGB128 PNG face crops per video and one policy across classes. Reuse `pilot/config/target_protocol.json` for the precommitted evaluation design. Prepare `target/blind.csv` with columns `item_id,video_id,path,sha256` and crop paths relative to `target/crops/`. The truth custodian should retain `target/labels.csv` with columns `video_id,dataset,compression,manipulation,label,cluster_id`. Real is `real,0`, fake scores are higher and each fake manipulation has a named type and label 1. Provide consistent cluster IDs for shared lineages.

```bash
export PYTHONPATH="$PWD/pilot"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P="$PWD/pilot/.venv/bin/python"

"$P" target_native.py freeze --run source_run --pairs pairs \
  --preprocessing target/target_preprocessing.json --out target/frozen
"$P" target_native.py score --frozen target/frozen --blind target/blind.csv \
  --crops target/crops --out target/scored

# Record the final extended freeze hash and score seal before retrieving labels.
"$P" target_native.py evaluate --frozen target/frozen --scored target/scored \
  --labels target/labels.csv --out target/evaluated
```

`fixture/` contains a smoke run of the freeze, blind scoring, and post-seal join for 16 DIV2K crops assigned to eight paired original/SwinIR groups and 16 pseudo-videos. It is an integrity test using known pilot inputs. Its processing record is specific to that fixture. Once real target preprocessing is pinned, create a fresh freeze with that processing record. The historical evaluator checks every frozen checksum, checks coverage and integrity of the score seal, then aggregates and evaluates scores. The new scorer writes an empty disagreement table for compatible seal checking. Its per-model scores are complete.

## References

[1] A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, “FaceForensics++: Learning to Detect Manipulated Facial Images,” ICCV, 2019. [Online]. Available: [author access instructions](https://github.com/ondyari/FaceForensics#access).

[2] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics,” CVPR, 2020. [Online]. Available: [author access instructions](https://github.com/yuezunli/celeb-deepfakeforensics#download).
