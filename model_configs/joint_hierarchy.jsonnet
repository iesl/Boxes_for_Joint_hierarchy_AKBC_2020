local DATA_DIR = std.extVar('DATA_DIR');
local WANDB = std.extVar('WANDB');
local CUDA_DEVICE = std.extVar('CUDA_DEVICE');

{
    "dataset_reader": {
        "type": "openke-dataset",
        "all_datadir": DATA_DIR,
        "dataset_name": "AUG_NEW",
        "mode": "train"
    },
    "iterator": {
        "type": "basic",
        "batch_size": 20000,
        "cache_instances": true
    },
    "model": {
        "type": "BCE-classification-split-neg-vol-penalty-box-model",
        "box_type": "DeltaBoxTensor",
        "debug": false,
        "embedding_dim": 5,
        "init_interval_center": 0.2,
        "init_interval_delta": 1,
        "num_entities": 164228,
        "num_relations": 1,
        "number_of_negative_samples_head": 1,
        "number_of_negative_samples_tail": 1,
        "regularization_weight": 30.08e-04,
        "single_box": "true",
        "softbox_temp": 2.01
    },
    "train_data_path": "dummpy_path",
    "validation_data_path": "dummy_path",
    "trainer": {
        "type": "callback",
        "callbacks": [
            {
                "checkpointer": {
                    "num_serialized_models_to_keep": 1
                },
                "type": "checkpoint"
            },
            {
                "patience": 200,
                "type": "track_metrics",
                "validation_metric": "+fscore"
            },
            {
                "type": "validate"
            }] + (if WANDB=='true' then [
            {
                "type": "log_metrics_to_wandb"
            }
        ] else []),
        "cuda_device": std.parseInt(CUDA_DEVICE),
        "num_epochs": 5000,
        "optimizer": {
            "type": "adam",
            "lr": 0.0443
        },
        "shuffle": true
    },
    "datasets_for_vocab_creation": [],
    "validation_dataset_reader": {
        "type": "classification-validation-dataset",
        "all_datadir": DATA_DIR,
        "dataset_name": "AUG_NEW",
        "validation_file": "classification_samples_valid2id.txt"
    },
    "validation_iterator": {
        "type": "basic",
        "batch_size": 5000,
        "cache_instances": true
    }
}
