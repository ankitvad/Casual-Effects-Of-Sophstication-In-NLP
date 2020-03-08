import torch
import torch.nn as nn
import numpy as np
import os
import random
import math
import time

from transformers.modeling_bert import BertConfig
from transformers.tokenization_bert import BertTokenizer

from models import Decoder, Seq2Seq
from utils import count_parameters, initialize_weights, train, evaluate, epoch_time, generate
from generation_utils import make_DataLoader, DiscoFuseProcessor
from collections import namedtuple

DiscoFuseExampleCI = namedtuple('DiscoFuseExampleCI', 'infused fused phenomanon connective sophisticated semantic')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    if torch.cuda.is_available():
        print("current device: ", torch.cuda.current_device())

    # special token
    SOPH = '<soph>'
    NSOPH = '<nsoph>'

    config = BertConfig.from_pretrained('bert-base-uncased')

    # constant the seed
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    num_added_token = tokenizer.add_tokens(([SOPH, NSOPH]))

    INPUT_DIM = len(tokenizer)  # len(SRC.vocab)
    OUTPUT_DIM = len(tokenizer)  # len(TRG.vocab)
    HID_DIM = 768
    DEC_LAYERS = 3
    DEC_HEADS = 8
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    SRC_PAD_IDX = 0
    TRG_PAD_IDX = 0
    BATCH_SIZE = 100
    MAX_SEQ_LEN = 50
    N_EPOCHS = 5
    CLIP = 1
    LEARNING_RATE = 0.0005
    SAVE_PATH = 'tut6-model.pt'
    LOAD_PATH = 'tut6-model.pt'

    unfreeze_bert = False
    do_load = False

    do_train = False
    do_eval = False
    do_generate = True

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device)

    model = Seq2Seq(dec, SRC_PAD_IDX, TRG_PAD_IDX, config, device).to(device)

    # Resize tokenizer
    model.bert_encoder.resize_token_embeddings(len(tokenizer))

    model.decoder.apply(initialize_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_valid_loss = float('inf')

    processor = DiscoFuseProcessor()

    valid_iterator, num_val_ex = make_DataLoader(data_dir='./',
                                                 processor=processor,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=MAX_SEQ_LEN,
                                                 batch_size=BATCH_SIZE,
                                                 mode="dev",
                                                 SOPH=SOPH,
                                                 NSOPH=NSOPH,
                                                 domain="sports")

    if do_train:
        for param in model.bert_encoder.parameters():
            param.requires_grad = unfreeze_bert

        print(f'The model has {count_parameters(model):,} trainable parameters')

        train_iterator, num_tr_ex = make_DataLoader(data_dir='./',
                                                    processor=processor,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=MAX_SEQ_LEN,
                                                    batch_size=BATCH_SIZE,
                                                    mode="train",
                                                    SOPH=SOPH,
                                                    NSOPH=NSOPH)

        print("---- Begin Training ----")
        if do_load and os.path.exists(LOAD_PATH):
            print("---- Loading model from {} ----".format(LOAD_PATH))
            model.load_state_dict(torch.load(LOAD_PATH))

        for epoch in range(N_EPOCHS):

            start_time = time.time()

            num_batches_in_epoch = int(num_tr_ex/BATCH_SIZE)  # 10000

            train_loss = train(model, train_iterator, optimizer, criterion, CLIP,  num_batches_in_epoch, device=device)
            valid_loss, valid_exact = evaluate(model, valid_iterator, criterion, device=device, tokenizer=tokenizer)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), SAVE_PATH)

            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
            print(f'\t Val. EXACT: {valid_exact:.2f}')

    elif do_eval:
        print("Doing only evaluation")
        model.load_state_dict(torch.load(LOAD_PATH))
        valid_loss, valid_exact = evaluate(model, valid_iterator, criterion, device=device, tokenizer=tokenizer)
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. EXACT: {valid_exact:3.3f}')

    elif do_generate:
        print("Doing only generation")
        model.load_state_dict(torch.load(LOAD_PATH))
        all_predictions, all_trgs, all_counter_predictions = generate(model, valid_iterator, device, tokenizer)
        all_counter_pred_str = [" ".join(a).replace(" ##", "") for a in all_counter_predictions]
        all_pred_str = [" ".join(a).replace(" ##", "") for a in all_predictions]
        all_trgs_str = [" ".join(a).replace(" ##", "") for a in all_trgs]
        with open("generated_fuse.txt", 'a') as fp:
            for i in range(len(all_predictions)):
                counter_pred_line = "Counter pred: " + all_counter_pred_str[i] + "\n"
                pred_line = "Origin pred:  " + all_pred_str[i] + "\n"
                trg_line = "origin trg:   " + all_trgs_str[i] + "\n\n"
                fp.writelines(counter_pred_line)
                fp.writelines(pred_line)
                fp.writelines(trg_line)

    else:
        raise ValueError("Error - must either train evaluate, or generate!")


if __name__ == '__main__':
    main()
