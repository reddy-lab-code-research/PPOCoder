from subprocess import Popen, PIPE
import os.path, subprocess
import os
import shutil
import re
import json
from tqdm import tqdm
import chardet
import jsonlines 
import tempfile as tfile
import json
import threading
import time


def compile_prog(filepath, lang):
    '''
    filepath: path of the file you would like to compile
    lang: prog. language; 'Py', 'Java', 'CPP', 'C', 'PHP', 'JS', 'CS'
    Dependencies:
    Java: Java Development kit (JDK) (https://www.oracle.com/java/technologies/downloads/)
    JS: Node.js (https://nodejs.org/en/download/)
    CS: Install mono library (brew install mono) (http://www.mono-project.com/Mono:OSX)
    '''
    if lang=='Py':
        cmd = 'python3 -m py_compile '+filepath
        #cmd = 'pylint -E ' + filepath
    elif lang=='Java':
        cmd = 'javac '+filepath
    elif lang=='CPP' or lang == 'C':
        cmd = 'g++ -std=c++11 '+ filepath
    # elif lang=='C':
    #     cmd = 'gcc '+filepath
    elif lang=='PHP':
        # cmd = "/home/aneesh/MuST-CoST/vendor/bin/phpstan analyse -l 5 --no-progress " + filepath 
        cmd = 'php -l ' + filepath
        #cmd = 'php -l -d display_errors=on' + filepath
    elif lang=='JS':
        cmd = 'node '+filepath
    elif lang=='CS':
        cmd = 'mcs '+filepath
        #cmd = 'csc '+filepath
    else:
        print('invalid argument')
        return
    proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE,shell=True)
    error = [i.decode('utf-8') for i in proc.stderr.readlines()]
    err = '\n'.join(error)
    output = [i.decode('utf-8') for i in proc.stdout.readlines()]
    op = '\n'.join(output)
    return err, op

def remove_comments(string, lang):
    if lang == 'Python':
        pattern = "('''[\s\S]*''')|(''[\s\S]*''')"
        string = re.sub(pattern, '', string)
        return re.sub(r'(?m)^ *#.*\n?', '', string)
    else:
        pattern = '\/\*[\s\S]*\*\/'
        pattern2 = '[^:]//.*|/\\*((?!=*/)(?s:.))+\\*/'
        string = re.sub(pattern, '', string)
        string = re.sub(pattern2, '', string)                                              
        return string
    
    
def php_compiler(code_str):
    prefix = '''"<?php '''
    
    code = """echo 'hello world';"""

    suffix = '" | php -l'

    cmd = "echo " +  code_str + code + suffix
    
    print(cmd)
    
    proc = subprocess.Popen(cmd, stdout=PIPE, stderr=PIPE,shell=True)
    
    error = [i.decode('utf-8') for i in proc.stderr.readlines()]
    err = '\n'.join(error)
    output = [i.decode('utf-8') for i in proc.stdout.readlines()]
    op = '\n'.join(output)
    
    return err, op


