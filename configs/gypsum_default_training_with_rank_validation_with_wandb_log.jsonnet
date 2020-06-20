{
	"dataset_reader": {
            "type":
            "openke-dataset",
            "dataset_name":
            "FB15K237",
            "all_datadir":
            "/mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models/.data",
            "mode":
            "train"
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
		"batch_size": 32,
                "cache_instances":true
	},
	"validation_iterator":{
		"type":"single-sample-rank-validation-iterator",
		"batch_size":1,
                "cache_instances":true
	},
        "model": {
            "type": "max-margin-conditional-box-model",
            "num_entities": 14541,
            "num_relations": 237,
            "embedding_dim": 50,
            "number_of_negative_samples":10,
	    "debug": false,
            "regularization_weight": 0,
            "init_interval_center": 0.25,
            "init_interval_delta": 0.1,
            "adversarial_negative": true,
            "adv_neg_softmax_temp": 0.8
        },
	"trainer":{
		"type":"callback",
		local common_debug = false,
		local common_freq = 2,
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
				"patience": 6,
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
			"type": "adam",
			"lr":0.01
		},
		"cuda_device": 0,
		"num_epochs": 200,
		"shuffle": true
	}

}
