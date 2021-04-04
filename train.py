import time
import logging
import random
import json


import tqdm
import numpy as np
import torch
from transformers import set_seed
from rouge_score import rouge_scorer


SEED = 42
set_seed(SEED)
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def serialize_args(args):
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except:
            return False
    
    dct = {k: v for k, v in args.__dict__.items() if is_jsonable(v) }
    return dct


def single_epoch_train(model, optimizer, train_loader, args):
    """
        Fine-tuning for a single epoch. This was done 
        in order to validate after each epoch.
    """
    model.train()
    logger = args.logger
    loader = tqdm.tqdm(train_loader)
    device = args.device
    
    loss_acumm = 0

    for idx, batch in enumerate(loader):
        input_ids, attention_mask, labels, decoder_input_ids, decoder_attention_mask = (
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['labels'].to(device),
            batch['decoder_input_ids'].to(device),
            batch['decoder_attention_mask'].to(device),
        )
        
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels, 
                        decoder_input_ids=decoder_input_ids, 
                        decoder_attention_mask=decoder_attention_mask)
        loss = outputs.loss
        
        # Update loss on tqdm loader description
        loader.set_description(f"Train Batch Loss: {loss.item():.3f}")
        loader.refresh()
        try:
            import wandb
            wandb.log({'loss': loss.item()})
        except:
            pass
        
        # Backward
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        # If accumulation step, then descend
        if idx % args.gradient_accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()
            
#         # Log every log_every batches
#         if not idx % args.log_every:
#             logger.info(f"Loss: {loss.item()}")
            
        loss_acumm += loss.item()
        
    return loss_acumm / len(loader)


def single_epoch_validate(model, tokenizer, valid_loader, args):
    """
        Testing for a single epoch.
        
    Generation
    """
    model.eval()
    logger = args.logger
    loader = tqdm.tqdm(valid_loader)
    device = args.device

    gend_outputs = []
    loss_acumm = 0
        
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            input_ids, attention_mask, labels, decoder_input_ids, decoder_attention_mask = (
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['labels'].to(device),
                batch['decoder_input_ids'].to(device),
                batch['decoder_attention_mask'].to(device),
            )

#             repetition_penalty = 2.5
#             length_penalty=1.0
#             no_repeat_ngram_size=3
            pred_ids = model.generate(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                max_length=args.dec_max_len
            )
            gend_outputs += [tokenizer.decode(ids, skip_special_tokens=True) for ids in pred_ids]
#                 max_length=64, 
#                 num_beams=3,
#                 repetition_penalty=repetition_penalty, 
#                 length_penalty=length_penalty, 
#                 no_repeat_ngram_size=no_repeat_ngram_size,
#                 early_stopping=True,
                # top_k=50,
                # top_p=1.0,
                # do_sample=False,
                # temperature=1.0,
                # num_return_sequences=10,
                # length_penalty=2,
                # min_length=3,
                # decoder_start_token_id=model.config.eos_token_id,
#             )

#             decoded_inputs = [tokenizer.decode(c, 
#                               skip_special_tokens=False, 
#                               clean_up_tokenization_spaces=False) 
#                               for c in decoder_input_ids]

#             decoded_preds = [tokenizer.decode(c, 
#                               skip_special_tokens=False, 
#                               clean_up_tokenization_spaces=False) 
#                               for c in pred_ids]

#             decoded_labels = [tokenizer.decode(c, 
#                               skip_special_tokens=False, 
#                               clean_up_tokenization_spaces=False) 
#                               for c in labels]

#             for inputs, preds, labels, kinds in zip(decoded_inputs, decoded_preds, decoded_labels, kind_batch):
#                 o = {'inputs': inputs, 'preds': preds, 'labels': labels, 'kinds': kinds}
#                 outputs.append(o)


            #with torch.cuda.amp.autocast():

            outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels, 
                        decoder_input_ids=decoder_input_ids, 
                        decoder_attention_mask=decoder_attention_mask)
            loss = outputs.loss

#             losses.append(loss.item())
            # Update loss on tqdm loader description
            loader.set_description(f"Valid Batch Loss: {loss.item():.3f}")
            loader.refresh()
            try:
                import wandb
                wandb.log({'val_loss': loss.item()})
            except:
                pass

            if not idx % args.log_every:
                logger.info(f"Loss: {loss.item()}")

            loss_acumm += loss.item()
        
    return loss_acumm / len(loader), gend_outputs
        # compute metrics based on outputs
#         metrics = compute_metric(tokenizer, outputs)
#         metrics['val_avg_loss'] = avg_loss

#     return avg_loss




def train(model, optimizer, tokenizer, train_loader, valid_loader, valid_texts, args):#, , test_loader, args):
    logger = args.logger
    
    with open(f"{args.checkpoint}/args.json", "w") as f:
        json.dump(serialize_args(args), f)

    
    for epoch in range(args.epochs):
        
        #with experiment.train():
        start_time = time.time()
        logger.info(f"Epoch {epoch + 1} (Globally {args.checkpoint_count})")

        # Training
        logger.info(f"Begin Training ... ")
        train_loss = single_epoch_train(model, optimizer, train_loader, args)
        mins = round((time.time() - start_time) / 60 , 2)
        valid_loss, valid_output = single_epoch_validate(model, tokenizer, valid_loader, args)
        
        logger.info(f"Training Finished!")
        logger.info(f"Time taken for training epoch {epoch+1} (globally {args.checkpoint_count}): {mins} min(s)")
        logger.info(f"Epoch : {epoch+1}, Training Average Loss : {train_loss}, Validation Average Loss : {valid_loss}")

        scores = []
        for gend, gold in zip(valid_output, valid_texts):
            scores.append(scorer.score(gold, gend))
        scores_r1 = [score['rouge1'].recall for score in scores]
        scores_rL = [score['rougeL'].recall for score in scores]
        score_r1 = sum(scores_r1)/len(scores_r1)
        score_rL = sum(scores_rL)/len(scores_rL)
        
        logger.info(f"Validation Rouge1 : {score_r1}, Validation RougeL : {score_rL}")

        # Saving model
        model.save_pretrained(f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
        logger.info(f"Checkpoint saved at {args.checkpoint}/saved_checkpoint_{args.checkpoint_count}")
        
            
        args.checkpoint_count += 1

    return 