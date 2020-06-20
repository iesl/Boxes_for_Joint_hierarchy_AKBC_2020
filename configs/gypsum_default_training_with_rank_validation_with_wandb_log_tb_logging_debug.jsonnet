{
	"dataset_reader": {
            "type":
            "openke-dataset",
            "dataset_name":
            "FB15K237",
            "all_datadir":
            "/mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models/.data",
            "mode":
            "train",
            "number_negative_samples":
            1
        },
	"validation_dataset_reader":{
	    "type":
            "openke-rank-validation-dataset",
            "dataset_name":
            "FB15K237",
            "all_datadir":
            "/mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models/.data"
            },
	"train_data_path": "dummpy_path",
	"validation_data_path": "dummy_path",
	"datasets_for_vocab_creation": [],
	"iterator":{
		"type":"basic",
		"batch_size": 32
	},
	"validation_iterator":{
		"type":"single-sample-rank-validation-iterator",
		"batch_size":1
	},
        "model": {
            "type": "max-margin-conditional-box-model",
            "num_entities": 14541,
            "num_relations": 237,
            "embedding_dim": 50,
	    "debug": false
        },
	"trainer":{
		"type":"callback",
		local common_debug = true,
		local common_freq = 100,
		"callbacks":[
			{
				"type": "debug-validate",
                                "debug": common_debug,
				"log_freq": common_freq
			},
			{
				"type": "checkpoint",
				"checkpointer":{
					"num_serialized_models_to_keep":1
				}
			},
			{
				"type": "track_metrics",
				"patience": 3,
				"validation_metric": "-avg_rank"
			},
			{
				"type": "tensorboard_logging",
				"log_freq":common_freq,
				"debug": common_debug
			},
			{
				"type": "log_metrics_to_wandb",
				"debug": common_debug,
				"debug_log_freq": common_freq
			}
		],
		"optimizer": {
			"type": "sparse_adam",
			"lr":0.001
		},
		"cuda_device": 0,
		"num_epochs": 25,
		"shuffle": true
	}

}
