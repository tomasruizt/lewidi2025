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
	--train_include_no_persona False

# Prevent make from treating dataset names as targets
%:
	@: