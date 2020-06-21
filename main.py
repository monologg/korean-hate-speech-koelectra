import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, set_seed, MODEL_CLASSES
from data_loader import load_examples


def main(args):
    init_logger()
    set_seed(args)

    tokenizer = load_tokenizer(args)
    train_dataset = load_examples(args, tokenizer, mode="train")
    dev_dataset = load_examples(args, tokenizer, mode="dev")
    test_dataset = load_examples(args, tokenizer, mode="test")
    trainer = Trainer(args,
                      tokenizer,
                      train_dataset,
                      dev_dataset,
                      test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_pred:
        trainer.load_model()
        trainer.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="korean-hate-speech", type=str, help="The name of the task to train")

    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--pred_dir", default="./preds", type=str, help="Directory that saves prediction files")

    parser.add_argument("--train_file", default="train.txt", type=str, help="Train file")
    parser.add_argument("--dev_file", default="validate.txt", type=str, help="Dev file")
    parser.add_argument("--test_file", default="test.txt", type=str, help="Test file")
    parser.add_argument("--prediction_file", default="prediction.csv", type=str, help="Output file for prediction")

    parser.add_argument("--model_type", default="koelectra-base-v2", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default="monologg/koelectra-base-v2-discriminator", type=str, help="Model name or path")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=100, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Warmup proportion for linear warmup")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log and save every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_pred", action="store_true", help="Whether to run prediction on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument('--bias_loss_coef', type=float, default=1.0, help='Coefficient for the bias loss.')
    parser.add_argument('--hate_loss_coef', type=float, default=1.0, help='Coefficient for the hate loss.')

    args = parser.parse_args()

    main(args)
