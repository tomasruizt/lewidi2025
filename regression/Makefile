.PHONY: train
.PHONY: regression

train:
	python regression.py \
	--root_dir . \
	--datasets CSC \
	--train True \
	--train_include_no_persona False

regression:
	python regression.py \
	--root_dir . \
	--datasets CSC \
	--train False \
	--train_include_no_persona False