"""
Project configuration used by training and evaluation scripts.

Set values here directly or override them with environment variables before running.
"""

import os


DATASET_PATH = os.environ.get("DATASET_PATH", "./data")
TASK_ID = int(os.environ.get("TASK_ID", "1"))
IN_CHANNELS = int(os.environ.get("IN_CHANNELS", "2"))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", "1"))
BACKGROUND_AS_CLASS = os.environ.get("BACKGROUND_AS_CLASS", "true").lower() == "true"

TRAIN_VAL_TEST_SPLIT = [0.4, 0.2, 0.4]
SEED = int(os.environ.get("SEED", "1203"))
TRAINING_EPOCH = int(os.environ.get("TRAINING_EPOCH", "100"))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", "16"))
VAL_BATCH_SIZE = int(os.environ.get("VAL_BATCH_SIZE", "16"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE", "16"))
TRAIN_CUDA = os.environ.get("TRAIN_CUDA", "true").lower() == "true"
BCE_WEIGHTS = [1, 100]
