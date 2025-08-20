.PHONY: train
.PHONY: inference
.PHONY: submission

# Usage: make train DATASET=CSC or make train CSC
train:
	python regression.py \
	--model_id google/t5gemma-s-s-prefixlm \
	--saved_models_dir ./saved_models \
	--datasets $(or $(DATASET),$(filter-out train,$(MAKECMDGOALS)),CSC) \
	--train True \
	--train_include_no_persona False

all-datasets-train:
	python regression.py \
	--model_id google/t5gemma-s-s-prefixlm \
	--saved_models_dir ./saved_models \
	--datasets CSC,MP,Paraphrase \
	--train True \
	--train_include_no_persona False \
	--n_exs_by_dataset_dev 200 \
	--n_exs_by_dataset_train 1000

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
	--preds_file ./saved_preds/Paraphrase-test_clear-preds.parquet

# Prevent make from treating dataset names as targets
%:
	@: