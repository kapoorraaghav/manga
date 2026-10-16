# Manga Transcription and Speaker Tracking

This project builds an AI-assisted pipeline that converts three-page manga sequences into ordered story text while keeping character identities consistent across pages.

The system uses a frozen **MagiV2** model for text detection, OCR, panel-based reading order, visual embeddings, character clustering, and initial speech-balloon associations. On top of this, I developed additional components for transcript-based weak supervision, lightweight classifiers, cross-page character matching, speaker assignment, and validated JSONL submission generation.

The pretrained MagiV2 backbone and its LoRA adapters were not trained or fine-tuned.

## Approach

The pipeline combines pretrained visual understanding with lightweight models trained on features extracted from the manga pages.

- **Text extraction and reading order:** MagiV2 detects text regions, extracts their contents, and determines their order using the panel layout.
- **Character identification:** Visual embeddings and page-local clusters provide the initial character identities.
- **Weak supervision:** OCR text is matched against the available transcript references to create approximate training labels. Since the references do not include bounding boxes, these labels are not always reliable.
- **Learned classifiers:** Small models help determine which text regions should be included and which are likely to represent narration.
- **Cross-page matching:** A learned character-pair matcher links character identities across pages.
- **Speaker assignment:** A learned ranker scores potential balloon-to-character associations and uses score and margin thresholds to make assignments.
- **Export and validation:** Predictions are converted into the required JSONL format and checked against the submission schema.

## Experiments and Results

All experiments use the same split of 64 training sequences and 16 validation sequences, with random seed 42. Exact duplicate pages were grouped, although books and chapters were not reliably separated.

| Pipeline stage | Text order | Balanced joint F1 | Joint F1 |
|---|---:|---:|---:|
| Frozen page-local identities | 0.9084 | 0.2565 | 0.4946 |
| Frozen cross-page matching | 0.9084 | 0.3993 | 0.6374 |
| Original learned filtering | 0.9085 | 0.3902 | 0.5990 |
| Earlier tuned pipeline | 0.9084 | 0.4037 | 0.6355 |
| Learned matcher only | 0.9084 | 0.4126 | 0.6382 |
| Learned ranker only | 0.9084 | 0.4062 | 0.6333 |
| Final matcher and ranker | 0.9084 | **0.4268** | 0.6311 |

The biggest improvement came from introducing cross-page character matching. This helped the system maintain character identities across different pages instead of treating each page independently.

I then experimented with different combinations of text filtering, character matching, and speaker assignment. A follow-up search across 96 settings showed that using Magi's essential-text flag alongside the learned narrator classifier worked better than the original learned story filter.

The final version combines the learned character matcher with the speaker ranker and achieves a balanced joint F1 of **0.4268**, compared with 0.4037 for the earlier tuned pipeline.

There were also some unsuccessful experiments. The strict-anchor speaker setup produced only 76 character pairs and achieved approximately 0.3525 balanced joint F1. Relaxing the anchor requirements increased the available training data to 442 character pairs and 273 speaker-ranking balloons. The final notebook evaluates 42 speaker configurations, and `experiments/all_experiments.csv` contains the results from all four experiment stages, including the failed strict-anchor experiment.

### Additional Metrics

- **Matched-token coverage:** 0.9381
- **Matched-speaker accuracy:** 0.6942
- **Final balanced joint F1:** 0.4268
- **Final joint F1:** 0.6311

The balanced joint F1 improved, but the token-weighted joint F1 decreased slightly compared with the earlier tuned pipeline. This suggests that improvements across the evaluation categories did not translate into an overall improvement on every metric.

These results should be treated as development results rather than a reliable estimate of generalization. Multiple configurations were selected using the same 16 validation sequences, so the final score may be optimistic.

## Final Inference Pipeline

The final inference configuration uses the following settings:

| Component | Configuration |
|---|---|
| Text inclusion | MagiV2 essential-text flag |
| Narration classifier threshold | 0.85 |
| Cross-page matching threshold | 0.35 |
| Speaker ranker score threshold | 0.55 |
| Speaker ranker margin threshold | 0.10 |

Cross-page merging is constrained: two distinct character clusters from the same page cannot be merged, and every cross-group pair must satisfy the matching condition.

After selecting the configuration, the lightweight models are refitted on all 80 development sequences. The trained story-classification head remains in the saved bundle but is not used by the selected inference path.

The notebook's `infer_test(bundle_path)` function implements the final inference process.

## Limitations

There are several limitations to keep in mind.

- **Weak training labels:** Transcript-to-OCR matching is approximate because the references do not contain text-region annotations. Errors in these matches can affect the classifiers and speaker models.
- **Speaker association errors:** Speaker anchors depend on confident transcript matches and upstream character associations. Even though the ranker excludes the current balloon when constructing its target, the remaining supervision can still be incorrect.
- **Off-panel characters:** Balloons that cannot be associated with a character receive separate unresolved identities. As a result, recurring off-panel speakers may be split into multiple identities.
- **Text filtering:** The essential-text flag can retain unwanted text or miss punctuation, thoughts, and vocalisations.
- **No dedicated thought detection:** The system does not have a separate model for distinguishing thoughts from spoken dialogue.
- **OCR limitations:** There is no dedicated OCR correction stage or mechanism for recovering balloons that MagiV2 fails to detect.
- **Evaluation limitations:** The validation set is small, and repeated tuning on the same sequences can lead to overfitting.
- **Semantic correctness:** Passing schema validation only confirms that the output follows the required structure; it does not guarantee that the transcription or speaker assignments are correct.

No independent test accuracy or complete manual review of test pages is claimed.

## Future Work

The next steps for improving the pipeline are:

1. Evaluate the final system on a fresh holdout split by book or chapter.
2. Audit text-region predictions on development pages to understand where errors originate.
3. Build a more reliable set of verified speech-balloon and character annotations.
4. Improve the distinction between narration, dialogue, thoughts, and vocalisations.
5. Develop better tracking for recurring off-panel speakers.
6. Investigate whether improvements in character matching can translate into more consistent speaker assignments.

A more reliable evaluation split and higher-quality supervision would be important before drawing stronger conclusions about the system's generalization.

## Running the Project

### Requirements

The notebook was developed using Python 3.10 in a CUDA-enabled environment. The recorded package versions are available in `results/environment.txt`.

### 1. Prepare the dataset

Open `manga_ocr_learned_speakers_local.ipynb` and update `DATA_ROOT` in the setup cell to point to the original dataset directory.

The directory should contain:

- `sequences.json`
- `score.py`
- `sample_submission.jsonl`
- `development/labels.jsonl`
- Development and test image directories

Install any missing dependencies using the notebook's dependency cell, then restart the kernel.

### 2. Run the notebook

Execute the remaining cells in order. The notebook extracts features, trains the lightweight models, compares configurations, refits the selected models, saves the trained weights, and generates `new_run/submission.jsonl`.

Feature caches are regenerated in `feature_cache`. The compact project package does not include the original images or cached features.

### 3. Load the trained pipeline

Once the notebook setup, model-loading code, and function definitions are available, the supplied trained bundle can be used directly:

```python
from pathlib import Path

predictions = infer_test(
    Path("trained_pipeline.joblib")
)
```

The final export cell demonstrates how to write the returned predictions to JSONL.

### 4. Re-score the validation predictions

To evaluate the supplied validation predictions, run the following command from the project directory:

```bash
python "PATH/TO/DATASET/score.py" \
  --references results/validation_references.jsonl \
  --predictions results/validation_predictions.jsonl \
  --output validation_rescore.json
```

Replace `PATH/TO/DATASET` with the actual dataset location.

**Important:** The supplied reference file contains only the 16 held-out validation sequences. Do not use development labels to score test predictions.

## Project Structure and Artifacts

The compact package includes the following files and directories:

- `manga_ocr_learned_speakers_local.ipynb` — final notebook containing the training, experiment-selection, and inference workflow.
- `trained_pipeline.joblib` — fitted classifiers, PCA, character matcher, speaker ranker, configuration, and metadata.
- `speaker_learning_strict.py` — implementation of the earlier strict-anchor experiment.
- `experiments/all_experiments.csv` — consolidated results from the four experiment stages.
- `results/` — validation predictions and scores, split information, metadata, environment details, training counts, selected settings, and checks.
- `new_run/submission.jsonl` — generated submission output.

The final submission contains 361 entries across 45 page lists and 15 test sequences.

Before the package was reduced, cached saved-model inference and a full retraining of the lightweight models reproduced the test predictions byte-for-byte and reproduced the validation metrics. Fresh GPU feature extraction was not rerun during packaging.

The final notebook was updated to make its data, work, and cache paths portable and to pin the pretrained model revision. Its learning and prediction logic remains unchanged. Earlier notebooks retain their original workspace paths and document previous development stages.

## Pretrained Model and License

This project uses [MagiV2](https://huggingface.co/ragavsachdeva/magiv2), with the model revision pinned to:

`fbc890fec52977142e8ee00bfe26e9458b65517c`

Loading the model requires custom model code.

According to the model card, the pretrained model is permitted for personal, research, non-commercial, and not-for-profit use. Other uses require an arrangement with the author. Review the model card and its terms before using the system for other purposes.

## Acknowledgements

This project builds on the pretrained MagiV2 model and extends its capabilities with lightweight learned components for manga transcription and speaker tracking.

The implementation and documentation were developed with AI assistance.
