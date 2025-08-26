.PHONY: train
.PHONY: inference
.PHONY: submission
.PHONY: all-datasets-train
.PHONY: profile

# Usage: make train DATASET=CSC or make train CSC
train:
	python regression.py \
	--model_id google/t5gemma-s-s-prefixlm \
	--saved_models_dir ./saved_models \
	--datasets $(or $(DATASET),$(filter-out train,$(MAKECMDGOALS)),CSC) \
	--train True \
	--train_include_no_persona False \
	--preds_file ./saved_preds/$(DATASET)-train-preds.parquet

all-datasets-train:
	python regression.py \
	--model_id google/t5gemma-2b-2b-prefixlm \
	--saved_models_dir ./saved_models \
	--preds_file ./saved_models/all_datasets/preds.parquet \
	--datasets CSC,MP,Paraphrase \
	--train True \
	--train_include_no_persona False \
	--full_eval_split dev \

profile:
	python regression.py \
	--model_id google/t5gemma-s-s-prefixlm \
	--saved_models_dir ./saved_models \
	--datasets CSC,MP,Paraphrase \
	--train True \
	--n_exs_by_dataset_train 5 \
	--train_torch_compile False \
	--do_profile True \
	--run_final_eval False \
	--gradient_checkpointing True \
	--batch_size 32 \
	--gradient_accumulation_steps 1

# Usage: make inference DATASET=CSC or make inference CSC  
inference:
	python regression.py \
	--model_id google/t5gemma-s-s-prefixlm \
	--saved_models_dir ./saved_models \
	--datasets $(or $(DATASET),$(filter-out inference,$(MAKECMDGOALS)),CSC) \
	--train False \
	--train_include_no_persona False \
	--num_preds_per_problem 11 \
	--n_exs_by_dataset_full_eval 50 \
	--full_eval_split dev

submission:
	python regression.py \
	--model_id google/t5gemma-s-s-prefixlm \
	--saved_models_dir ./saved_models \
	--datasets $(or $(DATASET),$(filter-out inference,$(MAKECMDGOALS)),CSC) \
	--train False \
	--train_include_no_persona False \
	--num_preds_per_problem 11 \
	--n_exs_by_dataset_full_eval 5000 \
	--full_eval_split test_clear \
	--preds_file ./saved_preds/$(DATASET)-test_clear-preds.parquet

# Prevent make from treating dataset names as targets
%:
	@: