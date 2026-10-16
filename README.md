# Manga transcription and speaker tracking

AI-assisted draft based on the supplied method report and recorded experiments. The task asks for participant-written prose; review and revise this account in your own words before submission.

The system converts three manga pages into ordered story text with consistent anonymous speakers. Frozen MagiV2 supplies detection, OCR, panel-based reading order, visual embeddings, local character clusters and initial balloon links. The custom work adds transcript-based weak labels, small classifiers, constrained cross-page matching, speaker ranking, experimentation and validated JSONL export. No Magi backbone weights or LoRA adapters were trained.

The supplied method report describes the earlier tuned pipeline, which reached 0.4037 balanced joint F1. The submitted notebook extends it with a learned character-pair matcher and balloon-to-character ranker, reaching 0.4268. The report's cosine merge threshold of 0.70 describes that earlier stage, not the final learned matcher.

## Experiments and results

All results below use the same 64-training/16-validation sequence split with seed 42. Exact duplicate pages were grouped; books and chapters were not reliably separated.

| Stage | Text order | Balanced joint F1 | Joint F1 |
|---|---:|---:|---:|
| Frozen page-local identities | 0.9084 | 0.2565 | 0.4946 |
| Frozen cross-page matching | 0.9084 | 0.3993 | 0.6374 |
| Original learned filtering | 0.9085 | 0.3902 | 0.5990 |
| Earlier tuned pipeline from report | 0.9084 | 0.4037 | 0.6355 |
| Learned matcher only, best setting | 0.9084 | 0.4126 | 0.6382 |
| Learned ranker only, best setting | 0.9084 | 0.4062 | 0.6333 |
| Final matcher and ranker | 0.9084 | 0.4268 | 0.6311 |

Cross-page matching produced the largest initial gain. The 96-setting follow-up selected Magi's essential-text flag instead of the learned story filter and retained the learned narrator classifier. The later strict-anchor speaker experiment supplied only 76 character pairs and reached about 0.3525 balanced joint F1 with both components. Relaxing the anchors supplied 442 pairs and 273 ranker-training balloons. The final notebook compares 42 speaker configurations. `experiments/all_experiments.csv` preserves all four experiment stages, including the failed strict run.

The final model improves balanced joint F1 by 0.0231 over the earlier tuned pipeline, while token-weighted joint F1 decreases. Matched-token coverage is 0.9381 and matched-speaker accuracy is 0.6942. Repeated tuning on these same 16 sequences makes these optimistic selection results, not an independent estimate of test accuracy.

## Final approach and limitations

OCR-to-reference matching provides weak labels because references have no region boxes. Inclusion and narration classifiers use crop embeddings and geometric/text features. Speaker anchors depend on confident transcript matches and upstream character associations. The ranker excludes the current balloon when constructing its own target, but the remaining supervision can still be wrong.

The final inference uses the essential-text flag, a learned narration threshold of 0.85, a learned cross-page match threshold of 0.35, and ranker score/margin thresholds of 0.55/0.10. Merging cannot join two distinct local clusters from one page and requires every cross-group pair to pass. Small models are refitted on all 80 development sequences after selection. The fitted story head remains in the bundle but is inactive in the selected inference path.

Page order and region order follow the manifest and Magi's panel layout. Printed whitespace is collapsed; text is otherwise taken from OCR. Narration receives exactly `NARRATION`; character labels restart per sequence. Unassociated balloons get separate unresolved identities, fragmenting recurring off-panel speakers. There is no separate thought-speaker model, OCR correction stage or recovery of missed balloons. Essential-text filtering can retain prohibited text or miss punctuation, thoughts and vocalisations. Schema validation does not establish semantic correctness.

Next steps are a fresh book-grouped holdout, a development-only region audit, verified balloon/character labels, better narration-versus-thought decisions and off-panel speaker tracking. No test accuracy or complete manual test-page review is claimed.

## Run and inspect

1. Open `manga_ocr_learned_speakers_local.ipynb` in a CUDA-enabled Python environment. Python 3.10 was used; the recorded packages are in `results/environment.txt`.
2. Set `DATA_ROOT` in its setup cell to the original dataset directory containing `sequences.json`, `score.py`, `sample_submission.jsonl`, `development/labels.jsonl`, and the development/test image trees. Run the dependency cell if needed, then restart the kernel.
3. Run all remaining cells. They extract features, train, compare settings, refit, save weights and generate `new_run/submission.jsonl`. Caches are regenerated in `feature_cache`; images and caches are deliberately omitted from this compact package.
4. The notebook's `infer_test(bundle_path)` function is the inference code used for final export. After notebook setup, model loading and function definitions are available, `infer_test(Path('trained_pipeline.joblib'))` reloads the supplied weights. The final export cell demonstrates writing its returned predictions to JSONL.

To score the supplied held-out predictions, run from this directory:

```sh
python "PATH/TO/DATASET/score.py" --references results/validation_references.jsonl --predictions results/validation_predictions.jsonl --output validation_rescore.json
```

The supplied reference file contains only the 16 held-out sequences. Do not score test predictions against development labels.

The final notebook's data/work/cache paths and kernel-name check were made portable, and the pretrained revision was pinned. Its learning and prediction logic is unchanged. Earlier notebooks retain their original workspace paths and document the earlier stages. `speaker_learning_strict.py` preserves the failed strict-anchor training implementation.

The package contains 15 test sequences, 45 page lists and 361 entries in `submission.jsonl`. `trained_pipeline.joblib` contains the fitted heads, PCA, matcher, ranker, configuration and metadata. `results/` holds validation predictions/scores, split, metadata, environment and consolidated training counts/settings/checks. Before slimming, cached saved-model inference and a full small-model retrain reproduced test predictions byte-for-byte and reproduced validation metrics. Fresh GPU extraction was not rerun during packaging. The notebooks have no saved cell outputs; the separate result files provide the recorded evidence.

Pretrained weights: [MagiV2](https://huggingface.co/ragavsachdeva/magiv2), revision `fbc890fec52977142e8ee00bfe26e9458b65517c`. Loading uses custom model code. Its model card permits personal, research, non-commercial and not-for-profit use; other uses require an arrangement with the author. The implementation and documentation were AI-assisted.
