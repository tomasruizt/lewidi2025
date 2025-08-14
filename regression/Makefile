.PHONY: train
.PHONY: inference

# Usage: make train DATASET=CSC or make train CSC
train:
	python regression.py \
	--root_dir . \
	--datasets $(or $(DATASET),$(filter-out train,$(MAKECMDGOALS)),CSC) \
	--train True \
	--train_include_no_persona False

# Usage: make inference DATASET=CSC or make inference CSC  
inference:
	python regression.py \
	--root_dir . \
	--datasets $(or $(DATASET),$(filter-out inference,$(MAKECMDGOALS)),CSC) \
	--train False \
	--train_include_no_persona False \
	--num_preds_per_problem 10 \
	--n_exs_by_dataset_full_eval 500

# Prevent make from treating dataset names as targets
%:
	@: