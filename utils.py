import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import namedtuple
import numpy as np
from copy import deepcopy

DiscoFuseExampleCI = namedtuple('DiscoFuseExampleCI', 'infused fused phenomanon connective sophisticated semantic')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, clip, num_batches_in_epoch, device, print_interval=100):
    model.train()

    epoch_loss = 0
    accumulated_sum_loss = 0

    for i, batch in enumerate(iterator):
        src, src_mask, src_segment, trg = batch
        src, src_mask, src_segment, trg = src.to(device), src_mask.to(device), src_segment.to(device), trg.to(device)

        output, _ = model(input_ids=src, input_mask=src_mask, segment_ids=src_segment, trg=trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        accumulated_sum_loss += loss.item()

        epoch_loss += loss.item()

        if ((i + 1) % print_interval) == 0:
            accumulated_mean_loss = accumulated_sum_loss / print_interval
            print("Batch {}/{}: Loss - {}".format(i + 1, num_batches_in_epoch, accumulated_mean_loss))
            accumulated_sum_loss = 0
        if (i + 1) > num_batches_in_epoch:
            batch_size = len(src)
            return epoch_loss / (num_batches_in_epoch * batch_size)

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device, tokenizer):
    model.eval()

    epoch_loss, corrects = 0, 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_mask, src_segment, trg = batch
            src, src_mask, src_segment, trg = src.to(device), src_mask.to(device), src_segment.to(device), trg.to(
                device)

            output, _ = model(input_ids=src, input_mask=src_mask, segment_ids=src_segment, trg=trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            orig_output = output.cpu().detach().numpy()  # [batch_size, max_seq_len-1, vocab_size]
            orig_preds = np.argmax(orig_output, axis=2)
            orig_trg = trg[:, 1:].cpu().detach().numpy()  # [batch_size, max_seq_len-1]

            output = output.contiguous().view(-1, output_dim)  # [batch_size * (max_seq_len-1), vocab_size]
            trg = trg[:, 1:].contiguous().view(-1)  # [batch_size * (max_seq_len-1)]

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            # print("orig_output_logits -", orig_output[0])
            # print("orig_preds_ids -", orig_preds[0])
            # print("orig_trg_ids -", orig_trg[0])
            # print("pred -", tokenizer.convert_ids_to_tokens(orig_preds[0]))
            # print("trg -", tokenizer.convert_ids_to_tokens(orig_trg[0]))

            corrects += calc_corrects(orig_preds, orig_trg)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator), 100 * corrects / len(iterator)


def generate(model, iterator, device, tokenizer):
    model.eval()

    corrects, total_cnt = 0, 0

    with torch.no_grad():

        all_predictions_ids, all_trgs_ids, all_counter_predictions_ids = [], [], []
        all_predictions, all_trgs, all_counter_predictions = [], [], []
        for i, batch in enumerate(iterator):
            batch_src, batch_src_mask, batch_src_segment, batch_trg = batch
            max_seq_len = batch_src.shape[1]
            for j in range(batch_src.shape[0]):
                src = batch_src[j, :]
                counter_src = deepcopy(src)
                if counter_src[1].data.cpu().item() == 30523:
                    counter_src[1] -= 1
                else:
                    counter_src[1] += 1
                src_mask = batch_src_mask[j, :]
                src_segment = batch_src_segment[j, :]
                trg = batch_trg[j, :]
                pred_indexes = [trg[0].data.cpu().item()]
                counter_pred_indexes = [trg[0].data.cpu().item()]

                src, counter_src, attn_mask, src_segment, trg = src.unsqueeze(0).to(device), \
                                                                counter_src.unsqueeze(0).to(device),\
                                                                src_mask.unsqueeze(0).to(device), \
                                                                src_segment.unsqueeze(0).to(device), \
                                                                trg.unsqueeze(0).to(device)

                src_mask = model.make_src_mask(src)
                with torch.no_grad():
                    enc_src = model.bert_encoder(
                        src,
                        attention_mask=attn_mask,
                        token_type_ids=src_segment
                    )
                    counter_enc_src = model.bert_encoder(
                        counter_src,
                        attention_mask=attn_mask,
                        token_type_ids=src_segment
                    )

                for i in range(max_seq_len):
                    pred_tensor = torch.LongTensor(pred_indexes).unsqueeze(0).to(device)
                    pred_mask = model.make_trg_mask(pred_tensor)
                    with torch.no_grad():
                        output, _ = model.decoder(pred_tensor, enc_src, pred_mask, src_mask)
                    pred_token = output.argmax(2)[:, -1].item()
                    pred_indexes.append(pred_token)
                    if pred_token == 102:
                        break
                all_predictions_ids.append(pred_indexes)
                all_predictions.append(tokenizer.convert_ids_to_tokens(pred_indexes))

                for i in range(max_seq_len):
                    counter_pred_tensor = torch.LongTensor(counter_pred_indexes).unsqueeze(0).to(device)
                    pred_mask = model.make_trg_mask(counter_pred_tensor)
                    with torch.no_grad():
                        counter_output, _ = model.decoder(counter_pred_tensor, counter_enc_src, pred_mask, src_mask)
                    counter_pred_token = counter_output.argmax(2)[:, -1].item()
                    counter_pred_indexes.append(counter_pred_token)
                    if counter_pred_token == 102:
                        break
                all_counter_predictions_ids.append(counter_pred_indexes)
                all_counter_predictions.append(tokenizer.convert_ids_to_tokens(counter_pred_indexes))

                non_padded_trg = [x for x in trg.data.cpu().numpy()[0] if x not in [0]]
                all_trgs_ids.append(non_padded_trg)
                all_trgs.append(tokenizer.convert_ids_to_tokens(non_padded_trg))

                if pred_indexes == non_padded_trg:
                    corrects += 1
                total_cnt += 1

                if total_cnt > 2000000:
                    print("Exact score is -", 100 * corrects / total_cnt)
                    return all_predictions, all_trgs, all_counter_predictions

    print("Exact score is -", 100 * corrects / total_cnt)
    return all_predictions, all_trgs, all_counter_predictions


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):
    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'],
                           rotation=45)
        ax.set_yticklabels([''] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def calc_corrects(padded_preds, padded_trgs):
    count_correct = 0
    for padded_pred, padded_trg in zip(padded_preds, padded_trgs):
        trg = [x for x in padded_trg if x not in [0]]
        pred = padded_pred[:len(trg)]
        pred = pred.tolist()

        if pred == trg:
            count_correct += 1
    return count_correct
