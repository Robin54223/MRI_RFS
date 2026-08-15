# MRI–report–clinical survival model

This repository implements a multimodal survival model that combines breast MRI volumes, radiology reports, and structured clinical features. The model uses a 3D ResNet image branch, a frozen RadioBERT semantic encoder, a clinical-feature branch, and two-stage Transformer fusion to predict recurrence risk.

## Files

- `running.py`: model definition and training entry point
- `external_test.py`: internal and external cohort evaluation
- `dataset_external.py`: data loading, preprocessing, and dataloader construction
- `config.py` and `config_new.py`: dataset and runtime configuration
- `MRI_Model.sh`: shell wrapper for training

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required assets

Configure the dataset and tokenizer paths with environment variables:

- `DATASET_PATH`: MRI data root
- `META_PATH`: internal-cohort metadata JSON
- `EXTERNAL_JSON_PATH`: external-cohort data root
- `TOKENIZER_PATH`: Hugging Face tokenizer name or local tokenizer directory

The pretrained RadioBERT directory is passed with `--radiobert_path`. Download the updated [`checkpoint_01.pth`](https://github.com/Robin54223/MRI_RFS/releases/download/v1.0.0/checkpoint_01.pth) weights from the GitHub Release and supply the local file to the evaluation script with `--model_paths`.

## Report-modality dropout and text-free inference

The model is trained to support inference when a radiology report is unavailable. For each training sample, report-modality dropout is applied independently with probability 0.2. When selected, all report token IDs and the corresponding attention mask are set to zero; the MRI volumes and structured clinical inputs remain unchanged.

The semantic branch is not removed or bypassed. The empty-report input follows the same frozen RadioBERT encoding and trainable projection pathway as a regular report, so the multimodal architecture is identical for report-present and report-absent samples. The probability can be changed with `--report_dropout`; use `0` to disable the strategy.

For text-free external evaluation, pass `--text_free_external`. This applies the same empty-report representation by setting the external samples' report token IDs and attention masks to zero while retaining the semantic branch.

## Training

```bash
export DATASET_PATH=/path/to/data
export META_PATH=/path/to/dataset.json
export EXTERNAL_JSON_PATH=/path/to/external
export TOKENIZER_PATH=roberta-base

python3 running.py \
  --name exp001 \
  --seed_t 2026 \
  --dropout 0.4 \
  --report_dropout 0.2 \
  --lr_image 5e-4 \
  --lr_report 1e-4 \
  --lr_total 1e-4 \
  --lr_clin 1e-3 \
  --radiobert_path /path/to/radiobert \
  --output_dir ./checkpoints
```

The best validation checkpoint and its run configuration are written to `--output_dir`.

## Evaluation

Evaluation with reports available:

```bash
python3 external_test.py \
  --name exp001 \
  --seed_t 2026 \
  --hidden_dim 256 \
  --dropout 0.4 \
  --lr_image 5e-4 \
  --lr_report 1e-4 \
  --lr_total 1e-4 \
  --model_paths /path/to/checkpoint_01.pth \
  --radiobert_path /path/to/radiobert \
  --internal_json /path/to/internal_dataset.json \
  --external_json /path/to/external_dataset.json
```

To run the external cohort without report text, add:

```bash
  --text_free_external
```

Internal and external outputs are saved to `./results_internal` and `./results_external` by default. Use `--internal_output_dir` and `--external_output_dir` to change these locations.
