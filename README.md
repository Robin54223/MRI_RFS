# mri-report-clinical-survival

This repository contains the `mri-report-clinical-survival` pipeline, a multimodal survival modeling project built from MRI volumes, radiology report text, and structured clinical features. The codebase has been cleaned up for GitHub publication by removing machine-specific paths and keeping data and model assets external to the repository.

## Repository Status

- Hard-coded local and cluster paths were removed from the training workflow.
- Configuration files now support environment variable overrides.
- Generated artifacts and private assets are excluded through `.gitignore`.
- Dependency installation is documented in `requirements.txt`.
- Training and evaluation require external datasets and pretrained model files supplied by the user.

## Project Files

- `running.py`: training entrypoint for the multimodal model
- `external_test.py`: evaluation script for internal and external cohorts
- `dataset_external.py`: dataset loading, preprocessing, and dataloader construction
- `config.py` and `config_new.py`: runtime configuration with environment variable support
- `MRI_Model.sh`: minimal shell wrapper for launching training

## Requirements

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Required External Assets

Set these paths before training or evaluation:

- `DATASET_PATH`: root directory containing the MRI data
- `META_PATH`: internal dataset metadata JSON file
- `EXTERNAL_JSON_PATH`: external dataset directory
- `TOKENIZER_PATH`: Hugging Face tokenizer name or local tokenizer directory
- `--radiobert_path`: local path to the pretrained RadioBERT model

## Training

Example:

```bash
export DATASET_PATH=/path/to/data
export META_PATH=/path/to/dataset.json
export EXTERNAL_JSON_PATH=/path/to/external
export TOKENIZER_PATH=roberta-base

python3 running.py \
  --name exp001 \
  --seed_t 2026 \
  --dropout 0.4 \
  --lr_image 5e-4 \
  --lr_report 1e-4 \
  --lr_total 1e-4 \
  --lr_clin 1e-3 \
  --radiobert_path /path/to/radiobert \
  --output_dir ./checkpoints
```

Training checkpoints are written to the directory passed through `--output_dir`.

## Evaluation

Example:

```bash
python3 external_test.py \
  --name exp001 \
  --seed_t 2026 \
  --hidden_dim 256 \
  --dropout 0.4 \
  --lr_image 5e-4 \
  --lr_report 1e-4 \
  --lr_total 1e-4 \
  --model_paths ./checkpoints/exp001.pth \
  --radiobert_path /path/to/radiobert \
  --internal_json /path/to/internal_dataset.json \
  --external_json /path/to/external_dataset.json
```

Evaluation outputs are saved under the directories configured in the evaluation command.

## Notes for Public Release

- Do not commit raw medical imaging data.
- Do not commit trained `.pth` checkpoint files.
- Do not commit cluster logs or generated result files.
- Verify that your data usage agreement allows public release of the source code.
- If you plan to open-source this project, add a license file before publishing.
