{
	"dataset_reader": {
            "type":
            "openke-dataset",
            "dataset_name":
            "HYPER_TR_0",
            "all_datadir":
            "/Users/dhruv/UnsyncedDocuments/IESL/kb_completion/models/.data",
            "mode":
            "train"
        },
	"validation_dataset_reader":{
	    "type":
            "classification-with-negs-validation-dataset",
            "dataset_name":
            "HYPER_TR_0",
            "all_datadir":
            "/Users/dhruv/UnsyncedDocuments/IESL/kb_completion/models/.data"
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
		"type":"basic",
		"batch_size":100,
                "cache_instances":true
	},
        "model": {
            "type": "max-margin-conditional-classification-box-model",
            "num_entities": 82114,
            "num_relations": 1,
            "embedding_dim": 10,
            "number_of_negative_samples":5,
            "box_type": "DeltaBoxTensor",
	    "debug": false,
            "regularization_weight": 0.001,
            "init_interval_center": 0.25,
            "init_interval_delta": 0.1,
            "margin": 5
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
				"patience": 15,
				"validation_metric": "+fscore"
			}
		],
		"optimizer": {
			"type": "adam",
			"lr":0.01
		},
		"cuda_device": -1,
		"num_epochs": 400,
		"shuffle": true
	}

}
