from tqdm import tqdm
from code_prepro.lang_processors import *
from compiler.terminal_compiler import TerminalCompiler
import os
import json


lang2compiler = {
                    "Python": TerminalCompiler('Python'),
                    "C++": TerminalCompiler('C++'),
                    "C": TerminalCompiler('C'),
                    "C#": TerminalCompiler('C#'),
                    "PHP": TerminalCompiler('PHP'),
                    "Java": TerminalCompiler('Java')
                }

ext2lang = {
    "py" : "Python",
    "cpp": "C++",
    "java": "Java"
}

file_tokenizers = {"Java": java_tokenizer, "C++": cpp_tokenizer, "C": c_tokenizer, "Python": py_tokenizer,
                   "Javascript": js_tokenizer, "PHP": php_tokenizer, "C#": cs_tokenizer}
file_detokenizers = {"Java": java_detokenizer, "C++": cpp_detokenizer, "C": c_detokenizer, "Python": py_detokenizer,
                   "Javascript": js_detokenizer, "PHP": php_detokenizer, "C#": cs_detokenizer}


experiment2lang = {
    "python": "Python",
    "c": "C",
    "cpp": "C++",
    "c_sharp": "C#",
    "java": "Java",
    "php": "PHP"
}


def read_hypotheses(hypo_path):
    hypo = []
    with open(hypo_path, "r") as f:
        for line in f.readlines():
            hypo.append(line.strip())
    return hypo

def write_summary(summary, path):
    with open(path, "w+") as f:
        for line in summary:
            f.write(json.dumps(line, ensure_ascii=False))
            f.write("\n")
            
################################################
os.environ["CUDA_VISIBLE_DEVICES"]="2"
parent_dir = '/home/grads/parshinshojaee/trl_code/trl_code/datasets/'
direct_path = "/home/grads/parshinshojaee/trl_code/trl_code/"
dir_dict = {'javascript':'Javascript', 'java':'Java', 'c_sharp':'C#', 'php':'PHP', 'python':'Python', 'c':'C', 'cpp':'C++'}
end_dict = {'javascript':'js', 'java':'java', 'c_sharp':'cs', 'php':'php', 'python':'py', 'c':'c', 'cpp':'cpp'}
# l1, l2 = 'php', 'python'
l1, l2 = 'python', 'c_sharp'
data_dir = parent_dir + dir_dict[l1] + '-' + dir_dict[l2] + '/'
template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
template = template.replace('XXX', dir_dict[l1]).replace('YYY', dir_dict[l2])
if not(os.path.exists(data_dir)):
    data_dir = parent_dir + dir_dict[l2] + '-' + dir_dict[l1] + '/'
    template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
    template = template.replace('XXX', dir_dict[l2]).replace('YYY', dir_dict[l1])
train_filename = template.replace('xxx', end_dict[l1]).replace('yyy', end_dict[l2])
dev_filename = train_filename.replace('train', 'val')
test_filename = train_filename.replace('train', 'test')
baseline_output_dir = direct_path + 'baselines/codet5/saved_models/'+l1+'-'+l2+'/'
load_model_path = direct_path + 'baselines/codet5/saved_models/'+l1+'-'+l2+'/checkpoint-best-bleu2/pytorch_model.bin'
output_dir = direct_path + 'saved_models/codet5/saved_models/'+l1+'-'+l2

class Args():
    def __init__(self):
        self.train_filename = train_filename
        self.dev_filename = dev_filename
        self.test_filename = test_filename
        self.baseline_output_dir = baseline_output_dir
        self.load_model_path = load_model_path
        self.max_source_length = 400#400
        self.max_target_length = 400#400
        self.train_batch_size = 16
        self.output_dir = output_dir
        self.reward_id = 2
        self.run = 4
        self.loss_W = 10
        self.lr = 1e-5
        self.kl_coef = 1
        self.reward_W = 0.01
        self.action_space = 10000
        self.ns = 5
        if not(os.path.exists(self.output_dir)):
            os.makedirs(self.output_dir)  
            
            
args = Args()
# path = args.output_dir + '/codet5_ppo' + '_reward%d'%(args.reward_id) +  '_bs%d'%(args.train_batch_size) + '_in-len%d'%(args.max_source_length) + '_out-len%d'%(args.max_target_length) +'_r%d/'%(args.run)  
# path = args.output_dir + '/codet5_ppo' + '_reward%d'%(args.reward_id) +  '_bs%d'%(args.train_batch_size) + '_in-len%d'%(args.max_source_length) + '_out-len%d'%(args.max_target_length) +'_r%d'%(args.run)+'_as%d/'%(args.action_space)    
path = args.output_dir + '/codet5_ppo' + '_reward%d'%(args.reward_id) +  '_bs%d'%(args.train_batch_size) + '_in-len%d'%(args.max_source_length) + '_out-len%d'%(args.max_target_length) +'_r%d'%(args.run)+'_as%d'%(args.action_space)+'_ns%d/'%(args.ns)  
# path = args.output_dir + '/codet5_ppo' + '_reward%d'%(args.reward_id) +  '_bs%d'%(args.train_batch_size) + '_in-len%d'%(args.max_source_length) + '_out-len%d'%(args.max_target_length) +'_r%d'%(args.run)+'_as%d'%(args.action_space)+'_ns%d'%(args.ns)+'_v2/'  
################################################
#  
# data_path = "./RL_experiments/preds/codet5_ppo_r67/"
data_path = path
# data_path = 'saved_models/codet5/saved_models/'+l1+'-'+l2 +'/'
# data_path = baseline_output_dir
# data_path = direct_path + 'datasets3/humaneval-x/'
# data_path = direct_path + 'datasets3/leetcode/'
# data_path = direct_path + 'baselines/codet5/leetcode_saved_models/'+dir_dict[l1]+'-'+dir_dict[l2]+'/'
# data_path = 'saved_models/codet5/saved_models/'+l1+'-'+l2 +'/leetcode_bs16_len400_as2_ns50_r5/'
lang_pair = 'Python-C#'
print(lang_pair,'- AS:',  args.action_space, '- NS: ', args.ns)
#all_experiments = os.listdir(data_path)

#all_experiments = ['Java-C++', 'Java-Python', 'Python-C++', 'Python-Java', 'C++-Java', 'C++-Python']
                #    'C++-Python', 'C++-PHP', 'C++-C', 'Python-C#', 'Python-Java', 'Python-C', 'Python-PHP', 'Python-C++']
all_experiments = ['train.model_ep1']
# all_experiments = ['test.model_ep0','train.model_ep0','test.model_ep1','train.model_ep1','test.model_ep2','train.model_ep2','test.model_ep3','train.model_ep3']
# all_experiments = ['test.model_ep2','train.model_ep2','test.model_ep3','train.model_ep3','test.model_ep4','train.model_ep4','test.model_ep5','train.model_ep5','test.model_ep6','train.model_ep6']
# all_experiments = ['test.model_ep0','train.model_ep0','test.model_ep1','train.model_ep1','test.model_ep2','train.model_ep2','test.model_ep3','train.model_ep3','test.model_ep4','train.model_ep4']
# all_experiments = ['train.model_ep9','test.model_ep10','train.model_ep10','test.model_ep11','train.model_ep11','test.model_ep12','train.model_ep12','test.model_ep13','train.model_ep13','test.model_ep14','train.model_ep14','test.model_ep15','train.model_ep15','test.model_ep16','train.model_ep16']
# 'test.model_ep6','train.model_ep6','test.model_ep7','train.model_ep7','test.model_ep8','train.model_ep8','test.model_ep9','train.model_ep9','test.model_ep10','train.model_ep10','test.model_ep11','train.model_ep11','test.model_ep12','train.model_ep12','test.model_ep13','train.model_ep13','test.model_ep14','train.model_ep14','test.model_ep15','train.model_ep15'
# all_experiments = ['train.output_maxlen320_v3-2']
# all_experiments = ['test.model.humanevalx']
# all_experiments = ['train.model.codexglue']
# all_experiments = ['train.model.leetcode']
# all_experiments = ['test-cpp-humaneval.cpp']
# all_experiments = ['test-python-humaneval.py']
# all_experiments = ['test-java-humaneval.java']
# all_experiments = ['test.txt.py']



#all_experiments = [exp for exp in all_experiments if os.path.isdir(os.path.join(data_path, exp))]
# all_experiments.remove('java-c_sharp_untrained')
#all_experiments.remove('aggregate_compilation_summary.jsonl')
#all_experiments.remove('java-c_sharp_untrained')
#all_experiments.remove('.ipynb_checkpoints')


print(all_experiments)
#hypothesis_filename = 'train.model_ep0'

# java_mapping = read_hypotheses("/home/aneesh/leetcode_data/leetcode_compiled/test_data/java-mapping.txt")
# python_mapping = read_hypotheses("/home/aneesh/leetcode_data/leetcode_compiled/test_data/py-mapping.txt")
# cpp_mapping = read_hypotheses("/home/aneesh/leetcode_data/leetcode_compiled/test_data/cpp-mapping.txt")

# lang2mapping = {
#     "Python": python_mapping,
#     "Java": java_mapping,
#     "C++": cpp_mapping
# } 

            
            
compilation_stats = {}
#["python-php"]:
for experiment in all_experiments:
    
    uncompiled_count = 0
    compiled_count = 0
    summary = []
    
    print(experiment)
    
    #src, trg = experiment.split('-')
    src, trg = lang_pair.split('-')
    
    lang = trg
    src_lang = src
    #lang = experiment2lang[trg]
    #src_lang = experiment2lang[src]
    #hypo_path = os.path.join(data_path, experiment, hypothesis_filename)
    hypo_path = os.path.join(data_path, experiment)
    
    hypotheses = read_hypotheses(hypo_path)
    
    for i, code_string in enumerate(tqdm(hypotheses)):
        
        if lang != "PHP":
            code_string = file_detokenizers[lang](code_string)
        
        # if lang == 'C++':
        #     code_string = code_string + 'int main() { return 0;}'

        error, output, did_compile = lang2compiler[lang].compile_code_string(code_string)
        #print(error, output, did_compile)
        
        if lang == "PHP":
            if "[ERROR]" in output:
                did_compile = False
            elif "[OK] No errors" in output:
                did_compile = True

        if error or not did_compile:
            uncompiled_count+=1
        elif did_compile:
            compiled_count+=1
            
        line_item = {
                        #"pid": lang2mapping[src_lang][i],
                        "pid": i,
                        "code_string": code_string,
                        "did_compile": did_compile,
                        "error": error,
                        "output": output
                    }
        summary.append(line_item)
    
    #summary_path = os.path.join(data_path, experiment, experiment+"-summary.jsonl")
    summary_path = os.path.join(data_path, experiment+"-summary.jsonl")
    write_summary(summary, summary_path)
    
    compilation_stats[experiment] = {"compilation_ratio":compiled_count/len(hypotheses),
                                     "compiled_count": compiled_count,
                                     "uncompiled_count": uncompiled_count}

for key, value in compilation_stats.items():
    print(f"{key}: {value}")
with open(os.path.join(data_path, "aggregate_compilation_summary.jsonl"), "w+") as f:
    for key, value in compilation_stats.items():
        f.write(json.dumps({key:value}))
        f.write("\n")
    
    


