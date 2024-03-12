from __future__ import absolute_import, division, print_function
import argparse
import json
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          T5ForConditionalGeneration, RobertaTokenizer, T5Config)
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from model import ConcatenateModel, CloneDetectionConcatenateModel, ClassificationConcatenateModel

cpu_cont = 16
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code1,
                 ast1,
                 label
                 ):
        self.code1_ids = code1
        self.label = label
        self.ast1 = ast1

def read_examples(file_path):
    code1s = []
    ast1s = []
    labels = []
    with open(file_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code1 = js['code']
            ast1 = js['ast']
            label = js['label']
            code1s.append(code1)
            ast1s.append(ast1)
            labels.append(label)
    return code1s,ast1s,labels
class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        self.examples = []
        code1s,ast1s,labels = read_examples(file_path)
        for i in tqdm(range(len(code1s))):
            self.examples.append(convert_examples_to_features(code1s[i],ast1s[i],labels[i],tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("code1_ids: {}".format(' '.join(map(str, example.code1_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].code1_ids, self.examples[i].code1_ids.ne(0), self.examples[i].ast1,self.examples[i].label

def custom_collate(batch):
    batch_size = len(batch)
    code1_ids = [batch[i][0] for i in range(batch_size)]
    code1_ne = [batch[i][1] for i in range(batch_size)]
    ast1 = [batch[i][2][0] for i in range(batch_size)]
    label = [batch[i][3] for i in range(batch_size)]

    code1_ids = torch.stack(code1_ids, dim=0)
    code1_ne = torch.stack(code1_ne, dim=0)
    label = torch.stack(label, dim=0)
    return [code1_ids,code1_ne,label,ast1]
def convert_examples_to_features(code1,ast1,label,tokenizer, args):
    # encode
    code1 = code1.replace('<s>', '')
    code1 = code1.replace('</s>', '')
    code1_ids = tokenizer.encode(code1, truncation=True, max_length=args.encoder_block_size, padding='max_length',
                                  return_tensors='pt')
    label = torch.tensor(label)
    return InputFeatures(code1_ids,ast1,label)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, collate_fn=custom_collate,batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1

    args.warmup_steps = args.max_steps // 5

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        device_ids = []
        for i in range(args.n_gpu):
            device_ids.append(i)
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100
    stop = 0
    writer_path = "tb/codet5_training_loss"

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (code1_ids,code1_ne) = [x.squeeze(1).to(args.device) for x in batch[:2]]
            label = batch[2].to(args.device)
            ast1 = batch[3]
            model.train()
            # the forward function automatically creates the correct decoder_input_ids
            loss, logits = model(code1_ids, code1_ne, label,ast1)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_loss = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        stop = 0
                        best_loss = eval_loss
                        logger.info("  " + "*" * 20)
                        logger.info("  Best Loss:%s", round(best_loss, 4))
                        logger.info("  " + "*" * 20)
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    else:
                        stop += 1
                    if stop >= 3:
                        logger.info("two epoches passed after last saving,early stopped.")
                        return 0

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, collate_fn=custom_collate, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))
    for batch in bar:
        (code1_ids, code1_ne) = [x.squeeze(1).to(args.device) for x in batch[:2]]
        label = batch[2].to(args.device)
        ast1 = batch[3]
        loss, logits = model(code1_ids, code1_ne, label,ast1)
        if args.n_gpu > 1:
            loss = loss.mean()
        eval_loss += loss.item()
        num += 1
    eval_loss = round(eval_loss / num, 5)
    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss

def split_string_by_braces(input_string):
    counter = 0
    start_index = 0
    substrings = []

    for i, char in enumerate(input_string):
        if char == '{':
            counter += 1
        elif char == '}':
            counter -= 1

        if counter == 0 and (char =='{' or char=='}'):
            # 当计数器为0时，切割字符串
            substring = input_string[start_index:i + 1]
            substrings.append(substring)
            start_index = i + 1

    return substrings
def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, collate_fn=custom_collate, batch_size=args.eval_batch_size, num_workers=0)
    # multi-gpu evaluate
    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        (code1_ids, code1_ne) = [x.squeeze(1).to(args.device) for x in batch[:2]]
        label = batch[2].to(args.device)
        ast1 = batch[3]
        with torch.no_grad():
            out_puts = model(code1_ids, code1_ne, label,ast1,True)
            logits.append(out_puts.cpu().numpy())
            y_trues.append(label.cpu().numpy())
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,
    }
    f = open("./results/awi_res.txt", "a")
    for key in sorted(result.keys()):
        f.write(key+"="+str(round(result[key],4))+"\n")

def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--train_data_file", default=None, type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                        help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_model_from_checkpoint", default=False, action='store_true',
                        help="Whether to load model from checkpoint.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--output_name", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--n_gpu', type=int, default=2,
                        help="using which gpu")
    parser.add_argument("--stage", default='type', type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed
    set_seed(args)
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    model = ClassificationConcatenateModel(args)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset(tokenizer, args, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_type='eval')
        if args.load_model_from_checkpoint:
            checkpoint_prefix = f'checkpoint-best-loss/{args.checkpoint_model_name}'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model.load_state_dict(torch.load(output_dir),strict=False)
            model.to(args.device)
        train(args, train_dataset, model, tokenizer, eval_dataset)
    if args.do_test:
        # checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        # output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        # model.load_state_dict(torch.load(output_dir, map_location=args.device))
        model.to(args.device)
        test_dataset = TextDataset(tokenizer, args, file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)

if __name__ == "__main__":
    main()
