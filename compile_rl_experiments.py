from tqdm import tqdm
from code_prepro.lang_processors import *
from compiler.terminal_compiler import TerminalCompiler
import os
import json
import argparse
import torch


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
 
 
 
parser = argparse.ArgumentParser()
## Required parameters  
parser.add_argument("--l1", default=None, type=str,
                        help="source language")  
parser.add_argument("--l2", default=None, type=str,
                    help="target language") 
parser.add_argument("--asp", default=2, type=int,
                    help="action space")  
parser.add_argument("--ns", default=5, type=int,
                    help="num syn samples") 
parser.add_argument("--data_path", default=None, type=str,
                    help="data parent directory")  
parser.add_argument("--output_path", default=None, type=str,
                    help="output directory")
parser.add_argument("--load_model_path", default=None, type=str,
                    help="path to load models")
parser.add_argument("--baseline_output_path", default=None, type=str,
                    help="path to load models")
parser.add_argument("--run", default=1, type=int,help="run ID")
args = parser.parse_args()
args.device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")



data_parent_dir = args.data_path
dir_dict = {'javascript':'Javascript', 'java':'Java', 'c_sharp':'C#', 'php':'PHP', 'python':'Python', 'c':'C', 'cpp':'C++'}
end_dict = {'javascript':'js', 'java':'java', 'c_sharp':'cs', 'php':'php', 'python':'py', 'c':'c', 'cpp':'cpp'}
l1, l2 = args.l1, args.l2
data_dir = data_parent_dir + '/' + dir_dict[l1] + '-' + dir_dict[l2] + '/'
template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
template = template.replace('XXX', dir_dict[l1]).replace('YYY', dir_dict[l2])
if not(os.path.exists(data_dir)):
    data_dir = data_parent_dir + '/' + dir_dict[l2] + '-' + dir_dict[l1] + '/'
    template = data_dir+'train-XXX-YYY-tok.xxx,'+data_dir+'train-XXX-YYY-tok.yyy'
    template = template.replace('XXX', dir_dict[l2]).replace('YYY', dir_dict[l1])
train_filename = template.replace('xxx', end_dict[l1]).replace('yyy', end_dict[l2])
dev_filename = train_filename.replace('train', 'val')
test_filename = train_filename.replace('train', 'test')
baseline_output_dir = args.baseline_output_path + '/'+l1+'-'+l2+'/'
load_model_path = args.load_model_path
output_dir = args.output_path + '/'+l1+'-'+l2+'/'


data_path = output_dir
lang_pair = 'Java-C++'
print(lang_pair,'- AS:',  args.asp, '- NS: ', args.ns)
all_experiments = ['test.model_ep0','test.model_ep1','test.model_ep2','test.model_ep3']

print(all_experiments)       
            
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

        error, output, did_compile = lang2compiler[lang].compile_code_string(code_string)

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
    
    


