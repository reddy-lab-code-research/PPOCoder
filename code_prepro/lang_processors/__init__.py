import code_prepro.lang_processors.cpp_processor
import code_prepro.lang_processors.java_processor
import code_prepro.lang_processors.python_processor
import code_prepro.lang_processors.csharp_processor
import code_prepro.lang_processors.c_processor
import code_prepro.lang_processors.php_processor
import code_prepro.lang_processors.javascript_processor
from code_prepro.lang_processors.lang_processor import LangProcessor

def get_detokenizer(lang):
    processor = LangProcessor.processors[lang](root_folder=so_path)
    tokenizer = processor.detokenize_code
    return tokenizer

def get_tokenizer(lang):
    processor = LangProcessor.processors[lang](root_folder=so_path)
    tokenizer = processor.tokenize_code
    return tokenizer

so_path = "./code_prepro/lang_processors/"
lang_py = 'python'
lang_java = 'java'
lang_cs = 'csharp'
lang_cpp = 'cpp'
lang_c = 'c'
lang_php = 'php'
lang_js = 'javascript'

file_extensions = {"Java": ".java", "C++": ".cpp", "C": ".c", "Python": ".py","Javascript": ".js",
                   "PHP":".php", "C#":".cs"}
lang_lower = {"Java": "java", "C++": "cpp", "C": "c", "Python": "python","Javascript": "javascript",
                   "PHP":"php", "C#":"csharp"}
lang_upper = {"java": "Java", "cpp": "C++", "c": "C", "python": "Python","javascript": "Javascript",
                   "php":"PHP", "csharp":"C#"}
tags = ['train', 'val', 'test']


py_tokenizer = get_tokenizer(lang_py)
cs_tokenizer = get_tokenizer(lang_cs)
java_tokenizer = get_tokenizer(lang_java)
cpp_tokenizer = get_tokenizer(lang_cpp)
js_tokenizer = get_tokenizer(lang_js)
c_tokenizer = get_tokenizer(lang_c)
# php_tokenizer = get_tokenizer(lang_php)
php_tokenizer = c_tokenizer

py_detokenizer = get_detokenizer(lang_py)
cs_detokenizer = get_detokenizer(lang_cs)
java_detokenizer = get_detokenizer(lang_java)
cpp_detokenizer = get_detokenizer(lang_cpp)
js_detokenizer = get_detokenizer(lang_js)
c_detokenizer = get_detokenizer(lang_c)
# php_tokenizer = get_detokenizer(lang_php)
php_detokenizer = c_detokenizer

file_tokenizers = {"Java": java_tokenizer, "C++": cpp_tokenizer, "C": c_tokenizer, "Python": py_tokenizer,
                   "Javascript": js_tokenizer, "PHP": php_tokenizer, "C#": cs_tokenizer}
file_detokenizers = {"Java": java_detokenizer, "C++": cpp_detokenizer, "C": c_detokenizer, "Python": py_detokenizer,
                   "Javascript": js_detokenizer, "PHP": php_detokenizer, "C#": cs_detokenizer}
