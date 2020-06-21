## Neural Topic Modeling with Continual Lifelong Learning (ICML 2020)

	Lifelong Neural Topic Modeling Framework using DocNADE topic model.
	This code consists of the implementations for the models proposed in the paper submitted: "Neural Topic Modeling with Continual Lifelong Learning"


Authors: Pankaj Gupta, Yatin Chaudhary, Thomas Runkler, Hinrich Schütze


## Requirements

	NOTE: installation of correct dependencies and version ensure the correct working of code.

	Requires Python 3 (tested with `3.6.5`). The remaining dependencies can then be installed via:

        $ pip install -r requirements.txt
        $ python -c "import nltk; nltk.download('all')"


## Data format

	"datasets"     --> this directory contains different sub-directories for different datasets. Each sub-directory contains CSV format files for training, validation and test sets. The CSV files in the directory must be named accordingly: "training_docnade.csv", "validation_docnade.csv", "test_docnade.csv". For this task, each CSV file (prior to preprocessing) consists of 2 string fields with a comma delimiter - the first is the label and the second is the document body (in bag-of-words representation). Each sub-directory also contains vocabulary file named "vocab_docnade.vocab", with 1 vocabulary token per line.

	"new_datasets" --> this directory contains different sub-directories for each target dataset i.e., "20NSshort", "TMNtitle" and "R21578title". Each sub-directory contains one target dataset and four source datasets where each source dataset is mapped to the vocabulary of the target dataset.

## Data preparation

	1) Configure dataset parameters present on the top of "Prepare_data_target_vocab.py" file.
	2) Run "python Prepare_data_target_vocab.py" command.

## How to use

	The script "train_model_lifelong_projection.py" will train the DocNADE Topic Model within lifelong framework and save it in a repository based on perplexity per word (PPL) or information retrieval (IR). It will also log all the training information in the same model folder. Following is the command line to use the script:
		
		$ python train_model_lifelong_projection.py --dataset  --docnadeVocab  --model  --initialize-docnade  --bidirectional  --activation  --use-embeddings-prior  --lambda-embeddings  --lambda-embeddings-list  --learning-rate  --batch-size  --num-steps  --log-every  --validation-bs  --test-bs  --validation-ppl-freq  --validation-ir-freq  --test-ir-freq  --test-ppl-freq  --num-classes  --reload-source-num-classes  --multi-label  --reload-source-multi-label  --patience 100 --supervised  --hidden-size  --combination-type  --generative-loss-weight  --vocab-size  --deep  --deep-hidden-sizes --reload  --reload-model-dir  --trainfile  --valfile  --testfile  --pretraining-target  --pretraining-epochs  --bias-sharing  --dataset-old    --reload-source-data-list  --W-old-path-list  --U-old-path-list  --bias-W-old-path-list  --bias-U-old-path-list  --W-old-vocab-path-list  --sal-loss  --sal-gamma  --sal-gamma-init  --sal-threshold  --ll-loss  --ll-lambda  --ll-lambda-init  --projection 
		
		The description of the above command line arguments is provided below (at the bottom).
		
		Ready to run shell scripts for all target datasets have been included in this folder. There are 6 scripts in total, corresponding to three target corpora (20NSshort, TMNtitle and R21578title) used to build topic model based on perplexity (PPL). The model built using PPL metric is used in extracting topics and then, compute topic coherence. Additionally, we also provide scripts to build topic models for information retrieval task for of the target corpora:
		
		1) "train_20NSshort_ALL_docnade_sigmoid_LL.sh"   --> script to run lifelong learning for "20NSshort" as target dataset with "perplexity" as evaluation measure
		1) "train_20NSshort_ALL_docnade_tanh_LL.sh"      --> script to run lifelong learning for "20NSshort" as target dataset with "information retrieval" as evaluation measure
		1) "train_TMNtitle_ALL_docnade_sigmoid_LL.sh"    --> script to run lifelong learning for "TMNtitle" as target dataset with "perplexity" as evaluation measure
		1) "train_TMNtitle_ALL_docnade_tanh_LL.sh"       --> script to run lifelong learning for "TMNtitle" as target dataset with "information retrieval" as evaluation measure
		1) "train_R21578title_ALL_docnade_sigmoid_LL.sh" --> script to run lifelong learning for "R21578title" as target dataset with "perplexity" as evaluation measure
		1) "train_R21578title_ALL_docnade_tanh_LL.sh"    --> script to run lifelong learning for "R21578title" as target dataset with "information retrieval" as evaluation measure
		
		Hyperparameter settings for different configurations are provided below:

			1) EmbTF:
				set argument "--use-embeddings-prior" to True
				
			2) TR: 
				set argument "--ll-loss" to True
				set argument "--projection" to True
				
			3) EmbTF + TR:
				set argument "--use-embeddings-prior" to True
				set argument "--ll-loss" to True
				set argument "--projection" to True
				
			4) EmbTF + TR + SAL:
				set argument "--use-embeddings-prior" to True
				set argument "--ll-loss" to True
				set argument "--projection" to True
				set argument "--sal-loss" to True


## Directory structure for results and datasets

# Experiments directory
	"model"  -->  this directory contains all saved models

# Experiment directory
Based on the hyperparameter settings of an experiment, the following directory structure will be generated for the experiment.

[Experiment directory] (dummy example: output after training): ./model/20NSshort_ALL_target_vocab_DocNADE_emb_lambda_manual_1.0_0.1_0.1_0.1_act_sigmoid_hid_200_vocab_1448_lr_0.001_19_2_2020/
	|
	|------ params.json    (file with hyperparameter settings saved in JSON format)
	|
	|------ ./model_ir/    (directory containing model saved on the criteria of Information Retrieval (IR))
	|
	|------ ./model_ppl/   (directory containing model saved on the criteria of Perplexity (PPL))
	|
	|------ ./logs/        (directory containing logs of model training and model reload)
				|
				|------ training_info.txt     (file containing negative log-likelihood loss and PPL/IR evaluation score on validation dataset during training)
				|
				|------ reload_info_ppl_target.txt      (file containing results of PPL evaluation score on validation and test set of "target" dataset)
				|
				|------ reload_info_ppl_source_0.txt    (file containing results of PPL evaluation score on validation and test set of "20NS" dataset)
				|
				|------ reload_info_ppl_source_1.txt    (file containing results of PPL evaluation score on validation and test set of "R21578" dataset)
				|
				|------ reload_info_ppl_source_2.txt    (file containing results of PPL evaluation score on validation and test set of "TMN" dataset)
				|
				|------ reload_info_ppl_source_3.txt    (file containing results of PPL evaluation score on validation and test set of "AGnews" dataset)
				|
				|------ reload_info_ir_target.txt      (file containing results of IR evaluation score on validation and test set of "target" dataset)
				|
				|------ reload_info_ir_source_0.txt    (file containing results of IR evaluation score on validation and test set of "20NS" dataset)
				|
				|------ reload_info_ir_source_1.txt    (file containing results of IR evaluation score on validation and test set of "R21578" dataset)
				|
				|------ reload_info_ir_source_2.txt    (file containing results of IR evaluation score on validation and test set of "TMN" dataset)
				|
				|------ reload_info_ir_source_3.txt    (file containing results of IR evaluation score on validation and test set of "AGnews" dataset)


# In case of reload a saved model
	set "--reload" parameter to True
	set "--reload-model-dir" parameter to saved model directory  --> "20NSshort_ALL_target_vocab_DocNADE_emb_lambda_manual_1.0_0.1_0.1_0.1_act_sigmoid_hid_200_vocab_1448_lr_0.001_19_2_2020/"