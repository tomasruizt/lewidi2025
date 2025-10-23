.PHONY: train
.PHONY: inference
.PHONY: submission
.PHONY: all-datasets-train
.PHONY: profile
.PHONY: copy-data

# Usage: make train DATASET=CSC or make train CSC
MODEL_ID ?= google/t5gemma-s-s-prefixlm
DATASET ?= CSC

train:
	@mkdir -p logs/$(MODEL_ID)

	python regression.py \
	--model_id $(MODEL_ID) \
	--saved_models_dir ./saved_models/$(MODEL_ID) \
	--datasets $(DATASET) \
	--train True \
	--logging_steps 10 \
	--train_include_no_persona False \
	--gradient_checkpointing True \
	--train_torch_compile True \
	--preds_file ./saved_preds/$(MODEL_ID)/$(DATASET)-train-preds.parquet | tee logs/$(MODEL_ID)/$(DATASET).log

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
	--preds_file ./saved_preds/$(or $(DATASET),$(filter-out submission,$(MAKECMDGOALS)),CSC)-test_clear-preds.parquet

copy-data:
	rsync -avH login.ai.lrz.de:/dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/lewidi-data/rlm/ /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/


loss-plots:
	mkdir -p imgs/loss/google_t5gemma-s-s-prefixlm
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm-pre-optimization/rlm/google_t5gemma-s-s-prefixlm/MP/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-s-s-prefixlm/MP.pdf
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm-pre-optimization/rlm/google_t5gemma-s-s-prefixlm/CSC/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-s-s-prefixlm/CSC.pdf
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm-pre-optimization/rlm/google_t5gemma-s-s-prefixlm/Paraphrase/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-s-s-prefixlm/Paraphrase.pdf

	mkdir -p imgs/loss/google_t5gemma-2b-2b-prefixlm
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-2b-2b-prefixlm/MP/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-2b-2b-prefixlm/MP.pdf
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-2b-2b-prefixlm/CSC/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-2b-2b-prefixlm/CSC.pdf
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-2b-2b-prefixlm/Paraphrase/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-2b-2b-prefixlm/Paraphrase.pdf

	mkdir -p imgs/loss/google_t5gemma-9b-2b-prefixlm
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-9b-2b-prefixlm/MP/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-9b-2b-prefixlm/MP.pdf
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-9b-2b-prefixlm/CSC/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-9b-2b-prefixlm/CSC.pdf
	lewidi-plot-loss --file /mnt/md0/tomasruiz/dss_home/lewidi-data/rlm/google_t5gemma-9b-2b-prefixlm/Paraphrase/logs/0_out.logs && mv loss.pdf imgs/loss/google_t5gemma-9b-2b-prefixlm/Paraphrase.pdf

# Prevent make from treating dataset names as targets
%:
	@: