# -*- coding: utf-8 -*-

# !pip install -q transformers==4.1.1 sentencepiece
# !pip install -q pytorch-lightning

# import argparse
# import csv
# import os
# import shutil
# import ast
# import pandas as pd
# import re
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from pytorch_lightning import LightningModule, Trainer, seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from transformers import (
#     T5Tokenizer,
#     MT5ForConditionalGeneration,
#     AdamW,
#     get_linear_schedule_with_warmup
# )
# from transformers.models.bart.modeling_bart import shift_tokens_right

def preprocess(x):
    attrs = x['pos_attr'].split(' ')
    for attr in attrs:
      x['target'] = x['target'].replace('<mask>',attr,1)
    return x

class DialogueDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length):
        # Preprocessing
        data = pd.read_csv('generation/retrieve_output.csv')
        data['pos_attr'] = data['pos_attr'].apply(lambda x: x[2:-2].replace("', '",' '))
        data['target'] = data['neg_content']
        data['neg_content'] = data['neg_content'].apply(lambda x: x.replace('<mask>',''))
        data['source'] = '<ATTRS> ' + data['pos_attr'] + ' <CONT_START> ' + data['neg_content'] + ' <START>'
        data = data.apply(preprocess, axis=1)
        data = data.drop(['neg_sentence','neg_content','pos_attr'], axis=1)

        model_input = data['source']
        src_texts = model_input.to_list()
        tgt_texts = data['target'].to_list()
        
        self.batch = tokenizer.prepare_seq2seq_batch(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return self.batch['input_ids'].size(0)

    def __getitem__(self, index):
        input_ids = self.batch['input_ids'][index]
        attention_mask = self.batch['attention_mask'][index]
        labels = self.batch['labels'][index]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    '''From fairseq'''
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

class MT5Trainer(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.tokenizer = T5Tokenizer.from_pretrained('google/mt5-small')
        self.model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')

        # Special tokens
        special_tokens_dict = {'additional_special_tokens': ['<ATTRS>','<CONT_START>','<START>']}
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)

        # loader
        dataset = DialogueDataset(
            data_dir=self.hparams.data_dir,
            split='train',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )
        self.train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.train_batch_size,
            shuffle=True
        )

    def forward(self, input_ids, attention_mask, decoder_input_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )

    def training_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        logits = outputs[0]
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=lprobs,
            target=labels,
            epsilon=self.hparams.label_smoothing,
            ignore_index=pad_token_id
        )

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        decoder_input_ids = shift_tokens_right(labels, pad_token_id)

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
        logits = outputs[0]
        lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs=lprobs,
            target=labels,
            epsilon=self.hparams.label_smoothing,
            ignore_index=pad_token_id
        )
        self.log('val_loss', loss, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # https://huggingface.co/blog/how-to-generate
        beam_outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        preds = [
            self.tokenizer.decode(beam_output, skip_special_tokens=True)
            for beam_output in beam_outputs
        ]
        return preds

    def test_epoch_end(self, outputs):
        with open(os.path.join(self.hparams.output_dir, 'preds.txt'), 'w') as f:
            for output in outputs:
                f.write('\n'.join(output) + '\n')

    def configure_optimizers(self):
        # optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': self.hparams.weight_decay
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            },
        ]
        betas = tuple(map(float, self.hparams.adam_betas[1:-1].split(',')))
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=betas,
            eps=self.hparams.adam_eps,
            lr=self.hparams.lr
        )

        # scheduler
        num_training_steps = (
            len(self.train_loader)
            // self.hparams.accumulate_grad_batches
            * self.hparams.max_epochs
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=num_training_steps
        )
        lr_dict = {'scheduler': lr_scheduler, 'interval': 'step'}

        return [optimizer], [lr_dict]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        dataset = DialogueDataset(
            data_dir=self.hparams.data_dir,
            split='val',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.val_batch_size
        )
        return loader

    def test_dataloader(self):
        dataset = DialogueDataset(
            data_dir=self.hparams.data_dir,
            split='test',
            tokenizer=self.tokenizer,
            max_length=self.hparams.max_length
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.val_batch_size
        )
        return loader

args = argparse.Namespace(
    data_dir='dataset',
    output_dir="weights",
    seed=42,
    label_smoothing=0.1,
    weight_decay=0.1,
    lr=1e-4,
    adam_betas='(0.9,0.999)',
    adam_eps=1e-6,
    num_warmup_steps=500,
    train_batch_size=4,
    val_batch_size=4,
    max_length=128,
    accumulate_grad_batches=16,
    gpus=1,
    gradient_clip_val=0.1,
    max_epochs=40
)

if os.path.isdir(args.output_dir):
    shutil.rmtree(args.output_dir)
os.mkdir(args.output_dir)

checkpoint_callback = ModelCheckpoint(
    dirpath=args.output_dir,
    monitor='val_loss',
    mode='min',
)
trainer = Trainer(
    callbacks=[checkpoint_callback],
    gradient_clip_val=args.gradient_clip_val,
    gpus=args.gpus,
    accumulate_grad_batches=args.accumulate_grad_batches,
    max_epochs=args.max_epochs,
)

def load_model():
    model = MT5Trainer(args)
    model = model.load_from_checkpoint('generation/weights')
    return model

def predict(model, attrs, context):
    txt = '<ATTRS> ' + ' '.join(attrs) + ' <CONT_START> ' + context.replace('<mask>','') + ' <START>'

    input_ids = model.tokenizer.encode(txt, return_tensors='pt')

    beam_output = model.model.generate(
        input_ids, 
        max_length=50,
        num_beams=10,
        early_stopping=True,
        num_return_sequences=5,
    )
    outputs = []
    for i in beam_output:
        output = model.tokenizer.decode(i)
        output = re.sub("</s>.*", "", output)
        output = re.sub("(<pad>|<mask>|<extra_id_[0-9]*>)\s?", "", output)
        outputs.append(output.strip())
    return outputs

# model = load_model()
# predict(model, ['อยาก'], '<mask>กินข้าวร้านนั้นเลยอ่ะ ไม่อร่อย รสชาติมาก')

