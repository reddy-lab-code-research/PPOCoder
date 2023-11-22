#Example of a run for Java-C++ Translation
python rl_run.py \
--run 1 \
--l1 java \
--l2 cpp \
--asp 5 \
--ns 10 \
--data_path PPOCodder/data/ \
--output_path PPOCodder/saved_models/ \
--load_model_path PPOCodder/baselines/saved_models/java-cpp/pytorch_model.bin \
--baseline_output_path /PPOCodder/baselines/saved_models/java-cpp/ \
--max_source_length 400 \
--max_target_length 400 \
--train_batch_size 32 \
--test_batch_size 48 \
--lr 1e-6 \
--kl_coef 0.1 \
--kl_target 1 \
--vf_coef 1e-3
