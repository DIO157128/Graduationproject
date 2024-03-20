import json
import os

import javalang
import tqdm
from javalang.ast import Node
from anytree import AnyNode

from utils import getvocabdict, create_tokenized_ast, write_jsonl


def read_json(js_path):
    codes = []
    docs = []
    with open(js_path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            js = json.loads(line)
            codes.append(js['code'])
            docs.append(js['docstring'])
    return codes,docs


def read_abs_json(js_path):
    codes = []
    docs = []
    abs_codes = []
    with open(js_path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            js = json.loads(line)
            codes.append(js['code'])
            docs.append(js['doc_string'])
            abs_codes.append(js['abs_code'])
    return codes,docs,abs_codes

def split(file_path,towrite):
    res = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            res.append(js)
            if idx>100:
                break
    write_jsonl(towrite,res)

def trans2dir(path,sub_path):
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm.tqdm(f):
            line = line.strip()
            js = json.loads(line)
            code = js['code']
            f1 = open(sub_path+'/{}.java'.format(count),'w', encoding="utf-8")
            f1.write(code)
            f1.close()
            count+=1

def getAbs(code_type):
    code_path = './codes/{}/'.format(code_type)
    code_abs_path = './codes_abstract/{}/'.format(code_type)
    codes = os.listdir(code_path)
    for i in tqdm.tqdm(range(len(codes))):
        code = code_path+'{}.java'.format(i)
        code_abs = code_abs_path+'{}.java'.format(i)
        cmd = "java -jar ../src2abs/src2abs-0.1-jar-with-dependencies.jar single method {} {} ../src2abs/idioms.csv".format(
        code,code_abs)
        os.system(cmd)
def combineAbs(code_type):
    json_file = './original_data/{}.jsonl'.format(code_type)
    codes,docs = read_json(json_file)
    code_abs_path = './codes_abstract/{}/'.format(code_type)
    codes_abs = os.listdir(code_abs_path)
    abs_codes = []
    assert len(codes_abs)/2==len(codes)
    for i in tqdm.tqdm(range(len(codes))):
        code_abs = code_abs_path + '{}.java'.format(i)
        f = open(code_abs,'r',encoding='utf-8')
        abs_codes.append(f.read().strip())
        f.close()
    res = [{'code':c,'abs_code': abs_c, 'doc_string': d} for c, abs_c,d in zip(codes,abs_codes,docs)]
    write_jsonl('abs_data/{}.jsonl'.format(code_type), res)

def get_all_data():
    file_paths = ['./abs_data/train.jsonl','./abs_data/valid.jsonl','./abs_data/test.jsonl']

    ##get vocab_dict
    all_abs_codes = []
    for file_path in file_paths:
        tem_codes, tem_docs,tem_abs_codes = read_abs_json(file_path)
        all_abs_codes.extend(tem_abs_codes)
        print(len(tem_abs_codes))


    vocabdict = getvocabdict(all_abs_codes)

    for file_path in file_paths:
        codes,doc_strings,abs_codes = read_abs_json(file_path)
        final_codes = []
        final_docs = []
        final_asts = []
        final_abs_codes = []
        count = 0
        for idx in tqdm.tqdm(range(len(codes))):
            try:
                code = codes[idx]
                abs_code = abs_codes[idx]
                programtokens = javalang.tokenizer.tokenize(abs_code)
                # print(list(programtokens))
                parser = javalang.parse.Parser(programtokens)
                programast = parser.parse_member_declaration()
                final_asts.append(programast)
                final_codes.append(code)
                final_abs_codes.append(abs_code)
                final_docs.append(doc_strings[idx])
            except Exception:
                count+=1
                continue
        print("failed:{}".format(count))
        treedict = create_tokenized_ast(final_asts, vocabdict)
        res = [{'code':c,'abs_code':ac,'doc_string':d,'ast':a}for c,ac,d,a in zip(final_codes,final_abs_codes,final_docs,treedict)]
        write_jsonl(file_path.replace('abs_data/',''),res)

if __name__ == '__main__':
    # combineAbs('train')
    # combineAbs('valid')
    # combineAbs('test')
    get_all_data()