
from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sklearn
from collections import namedtuple

from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from scipy.stats import entropy
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax
from sklearn.metrics import matthews_corrcoef, f1_score


from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from transformers.modeling_bert import BertConfig, BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class GeneratedDiscoFuseProcessor(DataProcessor):
    """Processor for the Domain Adaptation Sentiment classification data set."""
    def get_dev_examples(self, data_dir, data_type='sports', soph_flag=False, generation_flag=False):
        """See base class."""
        if generation_flag:
            generation_flag_str = "Counter pred:"
        else:
            generation_flag_str = "origin trg:"
        if soph_flag:
            soph_mode = ' <soph> '
        else:
            soph_mode = ' <nsoph> '

        file_path = os.path.join(data_dir, "generated_fuse_{}.txt".format(data_type))
        dev_list = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    continue
                else:
                    if generation_flag_str in line:
                        splits = line.split(soph_mode)
                        if len(splits) > 1:
                            dev_list.append(splits[1])

        if data_type == 'sports':
            labels = [1] * len(dev_list)
        elif data_type == 'wiki':
            labels = [0] * len(dev_list)
        else:
            print("Error, data type can be sports or wiki.")
            exit(1)

        return self._create_examples(dev_list, labels, 'dev')

    def get_labels(self):
        """See base class."""
        return ["negative", "positive"]

    def _create_examples(self, x, label, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_point) in enumerate(zip(x, label)):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            #             print(data_point)
            text_a = data_point[0]
            text_b = None
            label = "positive" if (data_point[1] == 1 or data_point[1] == "positive") else "negative"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
            
            
class DiscoFuseProcessor(DataProcessor):
    """Processor for the Domain Adaptation Sentiment classification data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        num_wiki = 2110554
        num_sports = 4176676
        with open(data_dir + "wiki_sports_train.pkl", 'rb') as fp:
            train_list = pickle.load(fp)
        labels = [0]*num_wiki + [1]*num_sports
       
        return self._create_examples(train_list, labels, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        with open(data_dir + "sports_dev.pkl", 'rb') as fp:
            sports_dev_list = pickle.load(fp)
        with open(data_dir + "wiki_dev.pkl", 'rb') as fp:
            wiki_dev_list = pickle.load(fp)
              
        dev_list = wiki_dev_list + sports_dev_list
        num_wiki = len(wiki_dev_list)
        num_sports = len(sports_dev_list)
        labels = [0]*num_wiki + [1]*num_sports
       
        return self._create_examples(dev_list, labels, 'dev')

    def get_labels(self):
        """See base class."""
        return ["negative", "positive"]

    def _create_examples(self, x, label, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_point) in enumerate(zip(x, label)):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
#             print(data_point)
            text_a = data_point[0].fused
            text_b = None
            label = "positive" if (data_point[1]==1 or data_point[1] == "positive") else "negative"
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
        

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 1:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "sentiment":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sentiment_cnn":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)
        

def evaluate(eval_dataloader, model, device, tokenizer, output_mode="classification", num_labels=2):
    model.eval()
    eval_loss = 0
    mean_prob = 0
    nb_eval_steps = 0
    preds = []
    cnt = 0

    for eval_element in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids, input_mask, segment_ids, label_ids = eval_element[:4]
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        if input_ids is not None:
            with torch.no_grad():
                outputs = model(input_ids, segment_ids, input_mask, labels=None)
                logits = outputs[0]
                probs = softmax(logits.detach().cpu().numpy(), axis=1)


            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            for prob, label in zip(probs, label_ids.detach().cpu().numpy()):
                mean_prob += prob[label]
                cnt += 1

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                all_label_ids = label_ids.detach().cpu().numpy()
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
                all_label_ids = np.append(all_label_ids, label_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    mean_prob = mean_prob / cnt
    preds = preds[0]
    if output_mode == "classification" or output_mode:
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    acc = simple_accuracy(preds, all_label_ids)
    wrong_pred_ids = []
    mistakes = []
    print("Mean Prob =", mean_prob)
    model.train()
    return acc, eval_loss, mistakes, wrong_pred_ids


def make_DataLoader(data_dir, processor, tokenizer, label_list, max_seq_length, batch_size=1,
                    output_mode="clssification", local_rank=-1, mode="train", N=-1, data_type=None, soph_flag=False,
                    generation_flag=False):
    if mode == "train":
        examples = processor.get_train_examples(data_dir)
    elif mode == "dev":
        if data_type is not None:
            examples = processor.get_dev_examples(data_dir, data_type, soph_flag, generation_flag)
        else:
            examples = processor.get_dev_examples(data_dir)
    elif mode == "dev_cross":
        examples = processor.get_test_examples(data_dir)
    if N > 0:
        examples = examples[:N]
    features = convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_mode)
    print("***** Running evaluation on {}-set *****".format(mode))
    print("  Num examples = %d", len(examples))
    print("  Batch size = %d", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if mode == "train":
        if local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    elif mode == "dev" or mode == "dev_cross":
        # Run prediction for full data
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader
    
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_dir",
                        default='./',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--test_data_dir",
                        default='./',
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str, required=False,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='ci_evaluation',
                        type=str,
                        required=False,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="./saves/",
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--load_model",
                        action='store_true',
                        help="Whether to load a fine-tuned model from output directory.")
    parser.add_argument("--model_name",
                        default=None,
                        type=str,
                        help="The name of the model to load, relevant only in case that load_model is positive.")
    parser.add_argument("--load_model_path",
                        default='./saves/pytorch_model.bin',
                        type=str,
                        help="Path to directory containing fine-tuned model.")
    parser.add_argument("--save_on_epoch_end",
                        action='store_true',
                        help="Whether to save the weights each time an epoch ends.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--N_train",
                        type=int,
                        default=-1,
                        help="number of training examples")
    parser.add_argument("--N_dev",
                        type=int,
                        default=-1,
                        help="number of development examples")
    parser.add_argument("--save_best_weights",
                        type=bool,
                        default=True,
                        help="saves model weight each time epoch accuracy is maximum")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--save_according_to',
                        type=str,
                        default='acc',
                        help="save results according to in domain dev acc or in domain dev loss")
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help="which optimizer model to use: adam or sgd")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--data_type',
                        type=str,
                        default='sports',
                        help="Evaluate classifier on sports or wiki")
    parser.add_argument("--soph_flag",
                        action='store_true',
                        help="Evaluate on sophisticated examples or not (unsophisticated).")
    parser.add_argument("--generation_flag",
                        action='store_true',
                        help="Evaluate on original examples or caunterfactuals (generated ones).")

    args = parser.parse_args()

    processors = {
        "sentiment": DiscoFuseProcessor,
        "ci_evaluation": GeneratedDiscoFuseProcessor,
    }

    output_modes = {
        "sentiment": "classification",
        "ci_evaluation": "classification",
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    print("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    print("learning rate: {}, batch size: {}".format(
        args.learning_rate, args.train_batch_size))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.load_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir) and args.task_name != "ci_evaluation":
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.train_data_dir)
        train_examples = train_examples[:args.N_train] if args.N_train > 0 else train_examples
        num_train_optimization_steps = int(
            len(train_examples) / train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Load a trained model and vocabulary that you have fine-tuned
    if args.load_model or args.load_model_path != '':

        # path to directory to load from fine-tuned model
        load_path = args.load_model_path if args.load_model_path != '' else args.output_dir
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                             'distributed_{}'.format(args.local_rank))

        if task_name == 'sentiment' or task_name == 'ci_evaluation':
            model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=args.cache_dir, num_labels=num_labels)
        else:
            print('Error! no task named: {}'.format(task_name))
            exit()

        # load pre train modek weights
        if args.load_model_path is not None:
            print("--- Loading model:", args.load_model_path)
            model.load_state_dict(torch.load(args.load_model_path), strict=False)
        else:
            model.load_state_dict(torch.load(os.path.join(load_path, "pytorch_model.bin")), strict=False)

        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        if not tokenizer:
            tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model.to(device)

    else:
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                             'distributed_{}'.format(args.local_rank))

        if task_name == "sentiment":
            model = BertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=args.cache_dir,
                                                                  num_labels=num_labels)
        else:
            print('Error! no task named: {}'.format(task_name))
            exit()

        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
        model.to(device)

    if args.local_rank != -1:
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = model.named_parameters()
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        if args.optimizer == 'adam':
            optimizer = AdamW(optimizer_grouped_parameters,
                                 lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.sgd(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
        # scheduler = ReduceLROnPlateau(optimizer, 'min',
        #                               patience=hparams.reduce_lr_on_plateau_patience,
        #                               factor=hparams.reduce_lr_on_plateau_factor, verbose=True)

    global_step = 0

    # prepare dev-set evaluation DataLoader
    # if do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    if task_name == 'ci_evaluation':
        eval_dataloader = make_DataLoader(data_dir=args.test_data_dir,
                                          processor=processor,
                                          tokenizer=tokenizer,
                                          label_list=label_list,
                                          max_seq_length=args.max_seq_length,
                                          batch_size=args.eval_batch_size,
                                          output_mode=output_mode,
                                          local_rank=args.local_rank,
                                          mode="dev",
                                          N=args.N_dev,
                                          data_type=args.data_type,
                                          soph_flag=soph_flag,
                                          generation_flag=generation_flag)
    else:
        eval_dataloader = make_DataLoader(data_dir=args.test_data_dir,
                                          processor=processor,
                                          tokenizer=tokenizer,
                                          label_list=label_list,
                                          max_seq_length=args.max_seq_length,
                                          batch_size=args.eval_batch_size,
                                          output_mode=output_mode,
                                          local_rank=args.local_rank,
                                          mode="dev",
                                          N=args.N_dev)


    if args.do_train:
        # prepare training DataLoader
        train_dataloader = make_DataLoader(data_dir=args.train_data_dir,
                                           processor=processor,
                                           tokenizer=tokenizer,
                                           label_list=label_list,
                                           max_seq_length=args.max_seq_length,
                                           batch_size=train_batch_size,
                                           output_mode=output_mode,
                                           local_rank=args.local_rank,
                                           mode="train",
                                           N=args.N_train)
        model.train()

        # main training loop
        best_dev_acc = 0.0
        best_dev_loss = 100000.0
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):

            tr_loss = 0
            tr_acc = 0

            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch[:4]

                # define a new function to compute loss values for both output_modes
                outputs = model(input_ids, segment_ids, input_mask, labels=None)
                logits = outputs[0]
                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss(ignore_index=-1)
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                    preds = logits.detach().cpu().numpy()
                    preds = np.argmax(preds, axis=1)
                    tr_acc += compute_metrics(task_name, preds, label_ids.detach().cpu().numpy())["acc"]

                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            # run evaluation on dev set
            # dev-set loss
            eval_results_dev = evaluate(eval_dataloader=eval_dataloader,
                                          model=model,
                                          device=device,
                                          tokenizer=tokenizer,
                                          output_mode=output_mode,
                                          num_labels=num_labels)

            dev_acc, dev_loss = eval_results_dev[:2]

            # train-set loss
            tr_loss /= nb_tr_steps
            tr_acc /= nb_tr_steps

            # print and save results
            result = {"acc": tr_acc, "loss": tr_loss, "dev_acc":dev_acc, "dev_loss": dev_loss}

            print('Epoch {}'.format(epoch + 1))
            for key, val in result.items():
                print("{}: {}".format(key, val))

            print("***** Evaluation results *****")
            for key in sorted(result.keys()):
                print("  %s = %s", key, str(result[key]))

            # Save model, configuration and tokenizer on the first epoch
            # If we save using the predefined names, we can load using `from_pretrained`
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            if epoch == 0:
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                model_to_save.config.to_json_file(output_config_file)
                tokenizer.save_vocabulary(args.output_dir)

            if args.save_on_epoch_end:
                # Save a trained model
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME + '.Epoch_{}'.format(epoch+1))
                torch.save(model_to_save.state_dict(), output_model_file)

            # save model with best performance on dev-set
            if args.save_best_weights and dev_acc > best_dev_acc:
                print("Saving model, accuracy improved from {} to {}".format(best_dev_acc, dev_acc))
                output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                best_dev_acc = dev_acc
            
            if args.save_according_to == 'acc':
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    
            elif args.save_according_to == 'loss':
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss

            if args.save_according_to == 'acc':
                print('Best results: Acc - {}'.format(best_dev_acc))
            elif args.save_according_to == 'loss':
                print('Best results: Loss - {}'.format(best_dev_loss))
            if args.model_name is not None:
                final_output_eval_file = os.path.join(args.output_dir, args.model_name + "-final_eval_results.txt")
            else:
                final_output_eval_file = os.path.join(args.output_dir, "final_eval_results.txt")

    elif args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # dev-set loss
        eval_results_dev = evaluate(eval_dataloader=eval_dataloader,
                                    model=model,
                                    device=device,
                                    tokenizer=tokenizer,
                                    output_mode=output_mode,
                                    num_labels=num_labels)
        print("eval_results_dev -", eval_results_dev)

        dev_acc, dev_loss = eval_results_dev[:2]
        # print results
        print('Accuracy: {}'.format(dev_acc))
        print('Loss: {}'.format(dev_loss))

    else:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


DiscoFuseExampleCI = namedtuple('DiscoFuseExampleCI', 'infused fused phenomanon connective sophisticated semantic')
run()
