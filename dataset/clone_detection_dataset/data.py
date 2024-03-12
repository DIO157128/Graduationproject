import json
import os

import javalang
import tqdm
from javalang.ast import Node
from anytree import AnyNode
def read_json(file_path):
    codes = []
    doc_strings = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = js['code']
            doc_string = js['docstring']
            codes.append(code)
            doc_strings.append(doc_string)

    return codes,doc_strings
def get_token(node):
    token = ''
    #print(isinstance(node, Node))
    #print(type(node))
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    #print(node.__class__.__name__,str(node))
    #print(node.__class__.__name__, node)
    return token
def get_child(root):
    #print(root)
    if isinstance(root, Node):
        children = root.children
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    #print(sub_item)
                    yield sub_item
            elif item:
                #print(item)
                yield item
    return list(expand(children))
def get_sequence(node, sequence):
    token, children = get_token(node), get_child(node)
    sequence.append(token)
    #print(len(sequence), token)
    for child in children:
        get_sequence(child, sequence)
def getnodeandedge(node,nodeindexlist,vocabdict,src,tgt,edgetype):
    token=node.token
    nodeindexlist.append([vocabdict[token]])
    for child in node.children:
        src.append(node.id)
        tgt.append(child.id)
        edgetype.append([0])
        src.append(child.id)
        tgt.append(node.id)
        edgetype.append([0])
        getnodeandedge(child,nodeindexlist,vocabdict,src,tgt,edgetype)

def createtree(root,node,nodelist,parent=None):
    id = len(nodelist)
    #print(id)
    token, children = get_token(node), get_child(node)
    if id==0:
        root.token=token
        root.data=node
    else:
        newnode=AnyNode(id=id,token=token,data=node,parent=parent)
    nodelist.append(node)
    for child in children:
        if id==0:
            createtree(root,child, nodelist, parent=root)
        else:
            createtree(root,child, nodelist, parent=newnode)
def create_graph(asts,vocabdict):
    pathlist = []
    treelist = []
    for tree in asts:
        #print(tree)
        #print(path)
        nodelist = []
        newtree=AnyNode(id=0,token=None,data=None)
        createtree(newtree, tree, nodelist)
        #print(path)
        #print(newtree)
        x = []
        edgesrc = []
        edgetgt = []
        edge_attr=[]
        getnodeandedge(newtree, x, vocabdict, edgesrc, edgetgt,edge_attr)

        #x = torch.tensor(x, dtype=torch.long, device=device)
        edge_index=[edgesrc, edgetgt]
        #edge_index = torch.tensor([edgesrc, edgetgt], dtype=torch.long, device=device)
        astlength=len(x)
        #print(x)
        #print(edge_index)
        #print(edge_attr)
        treelist.append([[x,edge_index,edge_attr],astlength])
    return treelist
def write_jsonl(filename, data):
    with open(filename, 'w') as file:
        for item in data:
            json_line = json.dumps(item)
            file.write(json_line + '\n')
def createFinetune():
    data_dir = '../BCB/bigclonebenchdata'
    data_splits = ['./traindata11.txt','./devdata.txt','./testdata.txt']
    data_names = ['./train_finetune_part.jsonl','./valid_finetune_part.jsonl','./test_finetune_part.jsonl']
    all_codes = []
    all_asts = []
    all_paths = []
    alltokens = []
    lengths = []
    count = 0
    for filename in tqdm.tqdm(os.listdir(data_dir)):
        full_dir = data_dir+'/'+filename
        f = open(full_dir,'r',encoding='utf-8')
        programtext=f.read()
        try:
            programtokens = javalang.tokenizer.tokenize(programtext.strip())
            # print(list(programtokens))
            parser = javalang.parse.Parser(programtokens)
            programast = parser.parse_member_declaration()
            get_sequence(programast, alltokens)
            all_codes.append(programtext)
            all_paths.append(full_dir)
            all_asts.append(programast)
        except Exception:
            count+=1
    print(count)
    alltokens = list(set(alltokens))
    vocabsize = len(alltokens)
    tokenids = range(vocabsize)
    print(vocabsize)
    vocabdict = dict(zip(alltokens, tokenids))
    all_asts = create_graph(all_asts,vocabdict)
    database = {}
    for p,c,a in zip(all_paths,all_codes,all_asts):
        database[p] = [c,a]
    for ds,dn in  zip(data_splits,data_names):
        this_code1 = []
        this_ast1 = []
        this_code2 = []
        this_ast2 = []
        this_label = []
        data_list = open(ds,'r').readlines()
        count = 0
        for dl in data_list:
            d1 = dl.split()[0]
            d2 = dl.split()[1]
            label = dl.split()[2]
            this_code1.append(database[d1][0])
            this_ast1.append(database[d1][1])
            this_code2.append(database[d2][0])
            this_ast2.append(database[d2][1])
            this_label.append(int(label))
            count+=1
            if count>100:
                break
        res = [{'code1':c1,'code2':c2,'ast1':a1,'ast2':a2,'label':l}for c1,c2,a1,a2,l in zip(this_code1,this_code2,this_ast1,this_ast2,this_label)]
        write_jsonl(dn,res)
if __name__ == '__main__':
    createFinetune()
