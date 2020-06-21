import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from data_loader import KoreanHateSpeechProcessor
from utils import MODEL_CLASSES, compute_metrics

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, tokenizer, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.bias_label_lst, self.hate_label_lst = KoreanHateSpeechProcessor.get_labels()

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]

        self.config = self.config_class.from_pretrained(args.model_name_or_path,
                                                        finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      bias_label_lst=self.bias_label_lst,
                                                      hate_label_lst=self.hate_label_lst)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * self.args.warmup_proportion),
                                                    num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)

        global_step = 0
        tr_loss = 0.0
        best_mean_weighted_f1 = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'bias_labels': batch[3],
                          'hate_labels': batch[4]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        results = self.evaluate("dev")
                        if results["mean_weighted_f1"] > best_mean_weighted_f1:  # Save best result based on mean f1 score
                            best_mean_weighted_f1 = results["mean_weighted_f1"]
                            self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0

        bias_preds = None
        bias_out_label_ids = None
        hate_preds = None
        hate_out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'bias_labels': batch[3],
                          'hate_labels': batch[4]}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, (bias_logits, hate_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Bias
            if bias_preds is None:
                bias_preds = bias_logits.detach().cpu().numpy()
                bias_out_label_ids = inputs['bias_labels'].detach().cpu().numpy()
            else:
                bias_preds = np.append(bias_preds, bias_logits.detach().cpu().numpy(), axis=0)
                bias_out_label_ids = np.append(
                    bias_out_label_ids, inputs['bias_labels'].detach().cpu().numpy(), axis=0)

            # Hate
            if hate_preds is None:
                hate_preds = hate_logits.detach().cpu().numpy()
                hate_out_label_ids = inputs['hate_labels'].detach().cpu().numpy()
            else:
                hate_preds = np.append(hate_preds, hate_logits.detach().cpu().numpy(), axis=0)
                hate_out_label_ids = np.append(
                    hate_out_label_ids, inputs['hate_labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        bias_preds = np.argmax(bias_preds, axis=1)
        hate_preds = np.argmax(hate_preds, axis=1)
        result = compute_metrics(bias_preds, hate_preds, bias_out_label_ids, hate_out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def predict(self):
        # Predict the test dataset which doesn't have label
        dataset = self.test_dataset
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running prediction on test dataset *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)

        bias_preds = None
        hate_preds = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'bias_labels': None,
                          'hate_labels': None}
                if self.args.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                _, (bias_logits, hate_logits) = outputs[:2]

            # Bias
            if bias_preds is None:
                bias_preds = bias_logits.detach().cpu().numpy()
            else:
                bias_preds = np.append(bias_preds, bias_logits.detach().cpu().numpy(), axis=0)

            # Hate
            if hate_preds is None:
                hate_preds = hate_logits.detach().cpu().numpy()
            else:
                hate_preds = np.append(hate_preds, hate_logits.detach().cpu().numpy(), axis=0)

        bias_preds = np.argmax(bias_preds, axis=1).tolist()
        hate_preds = np.argmax(hate_preds, axis=1).tolist()

        # Write the result
        logger.info("Writing Prediction to {}...".format(self.args.prediction_file))
        if not os.path.exists(self.args.pred_dir):
            os.makedirs(self.args.pred_dir)
        with open(os.path.join(self.args.pred_dir, self.args.prediction_file), "w", encoding="utf-8") as f:
            f.write("bias,hate\n")
            for bias_idx, hate_idx in zip(bias_preds, hate_preds):
                f.write("{},{}\n".format(self.bias_label_lst[bias_idx], self.hate_label_lst[hate_idx]))

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)
        self.tokenizer.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        self.config = self.config_class.from_pretrained(self.args.model_dir)
        self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                      config=self.config,
                                                      args=self.args,
                                                      bias_label_lst=self.bias_label_lst,
                                                      hate_label_lst=self.hate_label_lst)

        self.model.to(self.device)
        logger.info("***** Model Loaded *****")
