import json
import os

import javalang
import numpy as np
import tqdm
from javalang.ast import Node
from anytree import AnyNode
from utils import create_tokenized_ast,get_sequence,write_jsonl

def parseAST(programtext, alltokens):
    programtokens = javalang.tokenizer.tokenize(programtext.strip())
    # print(list(programtokens))
    parser = javalang.parse.Parser(programtokens)
    programast = parser.parse_member_declaration()
    get_sequence(programast, alltokens)
    return programast


def createFinetune():
    directorys = ['./abs_methods/CORRECT', './abs_methods/INCORRECT']
    pairs = []
    cur_label = 1
    for directory in directorys:
        for root, dirs, files in os.walk(directory):
            for file in files:
                path1 = os.path.join(root.replace('abs_',''), 'buggy.java')
                path2 = os.path.join(root.replace('abs_',''), 'fixed.java')
                f1 = open(path1, 'r', encoding='utf-8')
                f2 = open(path2, 'r', encoding='utf-8')
                abs_path1 = os.path.join(root, 'buggy.java')
                abs_path2 = os.path.join(root, 'fixed.java')
                abs_f1 = open(abs_path1, 'r', encoding='utf-8')
                abs_f2 = open(abs_path2, 'r', encoding='utf-8')
                pairs.append([f1.read(), f2.read(), abs_f1.read(),abs_f2.read(),cur_label])
                f1.close()
                f2.close()
                abs_f1.close()
                abs_f2.close()
                break
        cur_label = -1
    alltokens = []
    count = 0
    pairs_with_ast = []
    for p in tqdm.tqdm(pairs):
        code1 = p[0]
        code2 = p[1]
        abs_code1 = p[2]
        abs_code2 = p[3]
        label = p[4]
        try:
            ast1 = parseAST(abs_code1, alltokens)
            ast2 = parseAST(abs_code2, alltokens)
            pairs_with_ast.append([code1, code2, abs_code1,abs_code2,ast1, ast2, label])
        except Exception:
            count += 1
            continue
    print("failed:{}".format(count))
    alltokens = list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    print("vocabsize:{}".format(vocabsize))
    vocabdict = dict(zip(alltokens, tokenids))
    pairs_with_ast_tokenized = []
    for p in tqdm.tqdm(pairs_with_ast):
        code1 = p[0]
        code2 = p[1]
        abs_code1 = p[2]
        abs_code2 = p[3]
        ast1 = p[4]
        ast2 = p[5]
        label = p[6]
        trees = create_tokenized_ast([ast1, ast2], vocabdict)
        pairs_with_ast_tokenized.append([code1, code2, abs_code1,abs_code2,trees[0], trees[1], label])
    res = [{'code1': p[0], 'code2': p[1], 'abs_code1': p[2], 'abs_code2': p[3], 'ast1': p[4], 'ast2': p[5],'label': p[6]} for p in
           pairs_with_ast_tokenized]
    write_jsonl('./all_abs.jsonl', res)


def getAbs(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            path1 = os.path.join(root, 'buggy.java')
            path2 = os.path.join(root, 'fixed.java')
            abs_root = root.replace('methods','abs_methods')
            if not os.path.exists(abs_root):
                os.makedirs(abs_root)
            abs_path1 = os.path.join(abs_root, 'buggy.java')
            abs_path2 = os.path.join(abs_root, 'fixed.java')
            cmd = 'java -jar ../src2abs/src2abs-0.1-jar-with-dependencies.jar pair class {} {} {} {} ../src2abs/idioms.csv'.format(path1,path2,abs_path1,abs_path2)
            os.system(cmd)
            break
def split(path):
    all = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            all.append(js)
    np.random.shuffle(all)
    train = int(len(all)*0.8)
    valid = int(len(all)*0.1)
    test = int(len(all) * 0.1)
    train_split = all[0:train]
    val_split = all[train:train+valid]
    test_split = all[train + valid:]
    write_jsonl('train_abs.jsonl',train_split)
    write_jsonl('valid_abs.jsonl', val_split)
    write_jsonl('test_abs.jsonl', test_split)

if __name__ == '__main__':
    # getAbs('./methods/CORRECT')
    # getAbs('./methods/INCORRECT')
    createFinetune()
    split('all_abs.jsonl')
