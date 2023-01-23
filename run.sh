###########################################
#XLCoST

###########################################
#HumanEval-X
#CPP-PY (RL Model)
# python3 rl_codet5_test.py --l1 python --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/python-cpp/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/python-cpp/
# python3 rl_codet5_test.py --l1 cpp --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/cpp-python/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/cpp-python/
# python3 rl_codet5_test.py --l1 cpp --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/cpp-python/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoints_0-1reward_r3/pytorch_model_ep0.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/cpp-python/
# python3 rl_codet5_test.py --l1 python --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/python-cpp/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoints_0-1reward_r3/pytorch_model_ep1.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/python-cpp/
# python3 rl_codet5_test.py --l1 cpp --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/cpp-python/leetcode_bs16_len400_as2_ns20_r5/checkpoint-best-comp/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/cpp-python/leetcode_bs16_len400_as2_ns20_r5/
# python3 rl_codet5_test.py --l1 python --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/python-cpp/leetcode_bs16_len400_as2_ns20_r5/checkpoint-best-comp/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/python-cpp/leetcode_bs16_len400_as2_ns20_r5/


#CPP-PY (Baseline)
# python3 rl_codet5_test.py --l1 python --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/python-cpp/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/python-cpp/
# python3 rl_codet5_test.py --l1 cpp --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/cpp-python/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/cpp-python/
# python3 rl_codet5_test.py --l1 python --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Python-C++/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Python-C++/
# python3 rl_codet5_test.py --l1 cpp --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/C++-Python/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/C++-Python/


##########
#Java-PY (RL Model)
# python3 rl_codet5_test.py --l1 java --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/java-python/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-python/
# python3 rl_codet5_test.py --l1 python --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/python-java/codet5_ppo_reward2_bs16_in-len320_out-len320_r1/checkpoints_0-1reward_r1/pytorch_model_ep5.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/python-java/
# python3 rl_codet5_test.py --l1 java --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-python/leetcode_bs16_len400_as2_ns10_r5/checkpoint-best-comp/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-python/leetcode_bs16_len400_as2_ns10_r5/

#Java-PY (Baseline)
# python3 rl_codet5_test.py --l1 java --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-python/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-python/
# python3 rl_codet5_test.py --l1 python --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/python-java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/python-java/
# python3 rl_codet5_test.py --l1 java --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Java-Python/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Java-Python/
# python3 rl_codet5_test.py --l1 python --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Python-Java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Python-Java/

##########
#Java-CPP (RL Model)
# python3 rl_codet5_test.py --l1 java --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/java-cpp/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-cpp/
# python3 rl_codet5_test.py --l1 cpp --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/cpp-java/codet5_ppo_reward2_bs16_in-len320_out-len320_r1/checkpoints_0-1reward_r1/pytorch_model_ep3.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/cpp-java/
# python3 rl_codet5_test.py --l1 java --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-cpp/leetcode_bs16_len400_as2_ns50_r5/checkpoints/pytorch_model_ep0.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-cpp/leetcode_bs16_len400_as2_ns50_r5/
# python3 rl_codet5_test.py --l1 java --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-cpp/leetcode_bs16_len400_as2_ns50_r5/checkpoint-best-comp/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-cpp/leetcode_bs16_len400_as2_ns50_r5/

#Java-CPP (Baseline)
# python3 rl_codet5_test.py --l1 java --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-cpp/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-cpp/
# python3 rl_codet5_test.py --l1 cpp --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/cpp-java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/cpp-java/
# python3 rl_codet5_test.py --l1 java --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Java-C++/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Java-C++/
# python3 rl_codet5_test.py --l1 cpp --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/C++-Java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/C++-Java/

##############################
#CodeXGlUE:
#Java-CSharp (RL Model)
# python3 rl_codet5_test.py --l1 java --l2 c_sharp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/java-c_sharp/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-c_sharp/
# python3 rl_codet5_test.py --l1 c_sharp --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/java-c_sharp/codet5_ppo_reward2_bs16_in-len320_out-len320_r1/checkpoints_0-1reward_r1/pytorch_model_ep1.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/c_sharp-java/

#Java-CSharp (Baseline)
# python3 rl_codet5_test.py --l1 java --l2 c_sharp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-c_sharp/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-c_sharp/
# python3 rl_codet5_test.py --l1 c_sharp --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/c_sharp-java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/c_sharp-java/


##############################
#LeetCode:
#Java-CPP (RL Model)
# python3 rl_codet5_test.py --l1 java --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/java-cpp/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-cpp/
# python3 rl_codet5_test.py --l1 cpp --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/cpp-java/codet5_ppo_reward2_bs16_in-len320_out-len320_r1/checkpoints_0-1reward_r1/pytorch_model_ep3.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/cpp-java/

#Java-CPP (Baseline)
# python3 rl_codet5_test.py --l1 java --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-cpp/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-cpp/
# python3 rl_codet5_test.py --l1 cpp --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/cpp-java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/cpp-java/
# python3 rl_codet5_test.py --l1 java --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Java-C++/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Java-C++/
# python3 rl_codet5_test.py --l1 cpp --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/C++-Java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/C++-Java/

###########
#Java-PY (RL Model)
# python3 rl_codet5_test.py --l1 java --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/java-python/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-python/
# python3 rl_codet5_test.py --l1 java --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/java-python/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoints_0-1reward_r3/pytorch_model_ep5.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/java-python/
# python3 rl_codet5_test.py --l1 python --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/python-java/codet5_ppo_reward2_bs16_in-len320_out-len320_r1/checkpoints_0-1reward_r1/pytorch_model_ep4.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/python-java/
#Java-PY (Baseline)
# python3 rl_codet5_test.py --l1 java --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-python/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/java-python/
# python3 rl_codet5_test.py --l1 python --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/python-java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/python-java/
# python3 rl_codet5_test.py --l1 java --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Java-Python/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Java-Python/
# python3 rl_codet5_test.py --l1 python --l2 java --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Python-Java/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Python-Java/

###########
#CPP-PY (RL Model)
# python3 rl_codet5_test.py --l1 cpp --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/cpp-python/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/cpp-python/
# python3 rl_codet5_test.py --l1 python --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/saved_models/codet5/saved_models/python-cpp/codet5_ppo_reward2_bs16_in-len400_out-len400_r3/checkpoint-best-comp/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/rl_code_repo/saved_models/codet5/saved_models/python-cpp/

#CPP-PY (Baseline)
# python3 rl_codet5_test.py --l1 cpp --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/cpp-python/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/cpp-python/
# python3 rl_codet5_test.py --l1 python --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/python-cpp/checkpoint-best-bleu/pytorch_model.bin --n_gpu 2 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/saved_models/python-cpp/
# python3 rl_codet5_test.py --l1 cpp --l2 python --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/C++-Python/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/C++-Python/
# python3 rl_codet5_test.py --l1 python --l2 cpp --load_model_path /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Python-C++/checkpoint-best-bleu/pytorch_model.bin --n_gpu 3 --output_dir /home/grads/parshinshojaee/trl_code/trl_code/baselines/codet5/leetcode_saved_models/Python-C++/

###########