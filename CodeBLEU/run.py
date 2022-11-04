import os
import syntax_match
import dataflow_match
from calc_code_bleu import calc_code_bleu

l1 = 'cpp'
l2 = 'python'
direct_path = "/home/grads/parshinshojaee/trl_code/trl_code/"
output_dir = direct_path + 'saved_models/codet5/saved_models/'+l1+'-'+l2
class Args():
    def __init__(self):
        self.max_source_length = 320#400
        self.max_target_length = 320#400
        self.train_batch_size = 16
        self.output_dir = output_dir
        self.reward_id = 2
        self.run = 4
        self.loss_W = 10
        self.lr = 1e-6
        self.kl_coef = 1
        self.reward_W = 0.01
        self.epoch = 3
        if not(os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir) 
args = Args()
path = args.output_dir + '/codet5_ppo' + '_reward%d'%(args.reward_id) +  '_bs%d'%(args.train_batch_size) + '_in-len%d'%(args.max_source_length) + '_out-len%d'%(args.max_target_length) +'_r%d/'%(args.run)
references = os.path.join(path, "test.gold_ep%d"%(args.epoch) ) 
hypothesis = os.path.join(path, "test.model_ep%d"%(args.epoch) ) 
lang = l2
keywords_dir = 'CodeBLEU/keywords/'

codebleu_score = calc_code_bleu(references, hypothesis, lang, keywords_dir)
breakpoint()
