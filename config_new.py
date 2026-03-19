"""
Extended project configuration for dataset loading.
"""

import os


DATASET_PATH = os.environ.get("DATASET_PATH", "./data")
TASK_ID = int(os.environ.get("TASK_ID", "1"))
IN_CHANNELS = int(os.environ.get("IN_CHANNELS", "2"))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "1"))
BACKGROUND_AS_CLASS = os.environ.get("BACKGROUND_AS_CLASS", "true").lower() == "true"

TRAIN_VAL_TEST_SPLIT = [0.4, 0.3, 0.3]
SEED = int(os.environ.get("SEED", "1203"))
TRAINING_EPOCHS = int(os.environ.get("TRAINING_EPOCHS", "100"))
TRAINING_EPOCH = TRAINING_EPOCHS
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))
VAL_BATCH_SIZE = int(os.environ.get("VAL_BATCH_SIZE", "16"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "16"))
TRAIN_CUDA = os.environ.get("TRAIN_CUDA", "true").lower() == "true"
BCE_WEIGHTS = [1, 100]

META_PATH = os.environ.get("META_PATH", "./dataset.json")
EXTERNAL_JSON_PATH = os.environ.get("EXTERNAL_JSON_PATH", "./external")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "roberta-base")
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
PIN_MEMORY = os.environ.get("PIN_MEMORY", "false").lower() == "true"
