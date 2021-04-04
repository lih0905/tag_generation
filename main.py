
import os
import re
import sys
import math
import yaml
import logging
import argparse
import datetime
import json

import torch
import transformers
from transformers import BartModel, AutoModelForSeq2SeqLM
from kobart import get_pytorch_kobart_model, get_kobart_tokenizer

# from configloader import train_config
from dataloader import get_dataloader, SummaryDataset, SummaryBatchGenerator
from train import train


def gen_checkpoint_id(args):
    engines = "".join([engine.capitalize() for engine in args.ENGINE_ID.split("-")])
    tasks   = "".join([task.capitalize() for task in args.TASK_ID.split("-")])
    timez   = datetime.datetime.now().strftime("%Y%m%d%H%M")
    checkpoint_id = "_".join([engines, tasks, timez])
    return checkpoint_id

def get_logger(args):
    log_path = f"{args.checkpoint}/info/"

    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    train_instance_log_files = os.listdir(log_path)
    train_instance_count = len(train_instance_log_files)

    logging.basicConfig(
        filename=f'{args.checkpoint}/info/train_instance_{train_instance_count}_info.log', 
        filemode='w', 
        format="%(asctime)s | %(filename)15s | %(levelname)7s | %(funcName)10s | %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)
    logger.addHandler(streamHandler)

    logger.info("-"*40)
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")
    logger.info("-"*40)\

    return logger

def checkpoint_count(checkpoint):
    _, folders, files = next(iter(os.walk(checkpoint)))
    files = list(filter(lambda x: "saved_checkpoint_" in x, files))
    # regex used to extract only integer elements from the list of files in the corresponding folder
    # this is to extract the most recent checkpoint in case of continuation of training
    checkpoints = map(lambda x: int(re.search(r"[0-9]{1,}", x).group()[0]), files)
    
    try:
        last_checkpoint = sorted(checkpoints)[-1]
    except IndexError:
        last_checkpoint = 0
    return last_checkpoint

def get_args():
    global train_config

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_class",
        type=str,
        default='AutoModelForSeq2SeqLM'
    )
    parser.add_argument(
        "--tokenizer_class",
        type=str,
        default='AutoTokenizer'
    )
    parser.add_argument(
        "--optimizer_class",
        type=str,
        default='AdamW'
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda'
    )    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='checkpoint_20210328_large'
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='data'
    )
    parser.add_argument(
        "--enc_max_len",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--dec_max_len",
        type=int,
        default=128
    )
    args = parser.parse_args()
    args.device = args.device if args.device else 'cpu'
    
    return args

def main():
    # Get ArgParse
    args = get_args()
    if args.checkpoint:
        args.checkpoint = (
            "./model_checkpoint/" + args.checkpoint[-1]
            if args.checkpoint[-1] == "/"
            else "./model_checkpoint/" + args.checkpoint
        )
    else:
        args.checkpoint = "./model_checkpoint/" + gen_checkpoint_id(args)


    # If checkpoint path exists, load the last model
    if os.path.isdir(args.checkpoint):
        # EXAMPLE: "{engine_name}_{task_name}_{timestamp}/saved_checkpoint_1"     
        args.checkpoint_count = checkpoint_count(args.checkpoint)
        logger = get_logger(args)
        logger.info(f"Checkpoint path directory exists")
        logger.info(f"Loading model from saved_checkpoint_{args.checkpoint_count}")
        model = torch.load(f"{args.checkpoint}/saved_checkpoint_{args.checkpoint_count}") 
        
        args.checkpoint_count += 1 #
    # If there is none, create a checkpoint folder and train from scratch
    else:
        try:
            os.makedirs(args.checkpoint)
        except:
            print("Ignoring Existing File Path ...")

#         model = BartModel.from_pretrained(get_pytorch_kobart_model())
        model = AutoModelForSeq2SeqLM.from_pretrained(get_pytorch_kobart_model())
        
        args.checkpoint_count = 0
        logger = get_logger(args)

        logger.info(f"Creating a new directory for {args.checkpoint}")
    
    args.logger = logger
    
    model.to(args.device)
    
    # Define Tokenizer
    tokenizer = get_kobart_tokenizer()

    # Add Additional Special Tokens 
    #special_tokens_dict = {"sep_token": "<sep>"}
    #tokenizer.add_special_tokens(special_tokens_dict)
    #model.resize_token_embeddings(new_num_tokens=len(tokenizer))

    # Define Optimizer
    optimizer_class = getattr(transformers, args.optimizer_class)
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate)

    logger.info(f"Loading data from {args.data_dir} ...")
    with open("data/Brunch_accm_20210328_train.json", 'r') as f:
        train_data = json.load(f)
    train_context = [data['text'] for data in train_data]
    train_tag = [data['tag'] for data in train_data]
    with open("data/Brunch_accm_20210328_test.json", 'r') as f:
        test_data = json.load(f)
    test_context = [data['text'] for data in test_data]
    test_tag = [data['tag'] for data in test_data]
    
    train_dataset = SummaryDataset(train_context, train_tag, tokenizer, args.enc_max_len, args.dec_max_len, ignore_index=-100)    
    test_dataset = SummaryDataset(test_context, test_tag, tokenizer, args.enc_max_len, args.dec_max_len, ignore_index=-100)    
#     train_dataset = Seq2SeqDataset(data_path=os.path.join(args.data_dir, "train.json"))
#     valid_dataset = Seq2SeqDataset(data_path=os.path.join(args.data_dir, "valid.json"))
#     test_dataset = Seq2SeqDataset(data_path=os.path.join(args.data_dir, "test.json"))
    

    batch_generator = SummaryBatchGenerator(tokenizer)
    
    train_loader = get_dataloader(
        train_dataset, 
        batch_generator=batch_generator,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    
    test_loader = get_dataloader(
        test_dataset, 
        batch_generator=batch_generator,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    
#     test_loader = get_dataloader(
#         test_dataset, 
#         batch_generator=batch_generator,
#         batch_size=args.eval_batch_size,
#         shuffle=False,
#     )
    

    train(model, optimizer, tokenizer, train_loader, test_loader, test_tag, args)# test_loader, args)

if __name__ == "__main__":
    main()
