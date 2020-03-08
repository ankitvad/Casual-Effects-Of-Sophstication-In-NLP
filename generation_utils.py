import csv
import sys
import pickle
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from collections import namedtuple
import os

# import unicode

DiscoFuseExampleCI = namedtuple('DiscoFuseExampleCI', 'infused fused phenomanon connective sophisticated semantic')


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, infused, fused, soph_flag):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            infused: string. The untokenized text of the sequence.
            Only must be specified for sequence pair tasks.
            fused: string. The untokenized text of the output sequence.
            soph_flag: Bool.Sophistication flag.
        """
        self.guid = guid
        self.infused = infused
        self.fused = fused
        self.soph_flag = soph_flag


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, fused_ids, fused_mask):
        self.input_ids = input_ids  # infused
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.fused_ids = fused_ids  # fused
        self.fused_mask = fused_mask


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


class DiscoFuseProcessor(DataProcessor):
    """Processor for the Seq2Seq discofuse generation data set."""

    def get_train_examples(self, data_dir, domain="wiki"):
        """See base class."""
        with open(data_dir + "wiki_sports_train.pkl", 'rb') as fp:
            train_list = pickle.load(fp)
        return self._create_examples(train_list, 'train')

    def get_dev_examples(self, data_dir, domain="wiki"):
        """See base class."""
        with open(data_dir + domain + "_dev.pkl", 'rb') as fp:
            dev_list = pickle.load(fp)
        return self._create_examples(dev_list, 'dev')

    def get_test_examples(self, data_dir, domain="wiki"):
        """See base class."""
        with open(data_dir + domain + "_dev.pkl", 'rb') as fp:
            test_list = pickle.load(fp)
        return self._create_examples(test_list, 'test')

    def _create_examples(self, x, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, data_point) in enumerate(x):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            #             print("data_point -",data_point)
            infused = data_point.infused
            fused = data_point.fused

            # TAL: maybe here is the bug
            soph = data_point.sophisticated if set_type != 'test' else not data_point.sophisticated

            examples.append(
                InputExample(guid=guid, infused=infused, fused=fused, soph_flag=soph))
        return examples


def convert_examples_to_features(examples, max_seq_length, tokenizer, SOPH=None, NSOPH=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        # TODO: remove the first condition when running on a real computer
        # if ex_index > 20:
        #     continue
        if ex_index % 10000 == 0:
            print("Writing example " + str(ex_index) + " of " + str(len(examples)))

        infused_tokens = tokenizer.tokenize(example.infused)
        fused_tokens = tokenizer.tokenize(example.fused)

        # Account for [CLS], [SEP] and [SOPH] with "- 3"
        if len(infused_tokens) > max_seq_length - 3:
            infused_tokens = infused_tokens[:(max_seq_length - 3)]

        if len(fused_tokens) > max_seq_length - 3:
            fused_tokens = fused_tokens[:(max_seq_length - 3)]

        s_token = SOPH if example.soph_flag else NSOPH

        infused_tokens = ["[CLS]"] + [s_token] + infused_tokens + ["[SEP]"]
        segment_ids = [0] * len(infused_tokens)
        fused_tokens = ["[CLS]"] + [s_token] + fused_tokens + ["[SEP]"]

        enc_input_ids = tokenizer.convert_tokens_to_ids(infused_tokens)
        dec_input_ids = tokenizer.convert_tokens_to_ids(fused_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        enc_input_mask = [1] * len(enc_input_ids)
        dec_input_mask = [1] * len(dec_input_ids)

        # Zero-pad up to the sequence length.
        enc_padding = [0] * (max_seq_length - len(enc_input_ids))
        enc_input_ids += enc_padding
        enc_input_mask += enc_padding
        segment_ids += enc_padding

        dec_padding = [0] * (max_seq_length - len(dec_input_ids))
        dec_input_ids += dec_padding
        dec_input_mask += dec_padding

        assert len(enc_input_ids) == max_seq_length
        assert len(dec_input_ids) == max_seq_length
        assert len(enc_input_mask) == max_seq_length
        assert len(dec_input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 1:
            print("*** Example ***")
            print("guid:", (example.guid))
            print("infused_tokens: " + " ".join(
                [str(x) for x in infused_tokens]))
            print("fused_tokens: " + " ".join(
                [str(x) for x in fused_tokens]))
            print("enc_input_ids: " + " ".join([str(x) for x in enc_input_ids]))
            print("enc_input_mask: " + " ".join([str(x) for x in enc_input_mask]))
            print(
                "segment_ids: " + " ".join([str(x) for x in segment_ids]))
            print("dec_input_ids: " + " ".join([str(x) for x in dec_input_ids]))
            print("dec_input_mask: " + " ".join([str(x) for x in dec_input_mask]))

        features.append(
            InputFeatures(input_ids=enc_input_ids,
                          input_mask=enc_input_mask,
                          segment_ids=segment_ids,
                          fused_ids=dec_input_ids,
                          fused_mask=dec_input_mask))
    return features


def make_DataLoader(data_dir, processor, tokenizer, max_seq_length, batch_size=1, local_rank=-1, mode="train", N=-1,
                    SOPH=None, NSOPH=None, overwrite_cache=False, domain="wiki"):
    if mode == "train":
        examples = processor.get_train_examples(data_dir=data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(data_dir=data_dir, domain=domain)
    elif mode == "test":
        examples = processor.get_test_examples(data_dir=data_dir, domain=domain)
    if N > 0:
        examples = examples[:N]
    cached_features_file = os.path.join(
        data_dir,
        "cached_{}_{}".format(mode, max_seq_length),
    )
    if os.path.exists(cached_features_file) and not overwrite_cache:
        print("Loading features from cached file", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at", data_dir)
        features = convert_examples_to_features(examples, max_seq_length, tokenizer, SOPH, NSOPH)
        print("Saving features into cached file", cached_features_file)
        torch.save(features, cached_features_file)
    print("***** Running evaluation on {}-set *****".format(mode))
    print("  Num examples =", len(examples))
    print("  Batch size =", batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_fused_ids = torch.tensor([f.fused_ids for f in features], dtype=torch.long)
    all_fused_mask = torch.tensor([f.fused_mask for f in features], dtype=torch.long)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_fused_ids)

    if mode == "train":
        if local_rank == -1:
            sampler = RandomSampler(data)
        else:
            sampler = DistributedSampler(data)
    elif mode == "dev" or mode == "test":
        # Run prediction for full data
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)

    return dataloader, len(examples)

