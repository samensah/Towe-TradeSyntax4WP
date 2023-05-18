import argparse

# training configurations
parser = argparse.ArgumentParser(description="configs for aspect-oriented opinion words extraction")

parser.add_argument('--dataset', type=str, default='14lap', help='dataset: laptop14, rest14, rest15 or rest16')
parser.add_argument("--bert_model", default="bert-base-uncased", type=str)
parser.add_argument("--train_batch_size", type=int, default=6, help="train batch size")
parser.add_argument("--eval_batch_size", type=int, default=32, help="eval batch size")
parser.add_argument("--chunk_size", type=int, default=10, help="chunk size for the hidden state")
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping.')
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--n_epoch", type=int, default=100, help="number of training epoch")
parser.add_argument("--early_stop", type=int, default=20, help="number of early stop epochs")
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--load_dir', type=str, default='./random1_16res/best_model.pt', help='Root dir for saving models.')
parser.add_argument('--print_step', type=int, default=10, help='Print log every k steps in training process.')

# network configurations
parser.add_argument("--dim_bilstm_hidden", type=int, default=100, help="hidden dimension for the bilstm")
parser.add_argument("--dim_bert", type=int, default=768, help="bert embedding dimension")
parser.add_argument("--dim_position", type=int, default=100, help="Position embedding dimension")
parser.add_argument("--use_A", type=int, default=0, help="if using the aspect enhancement")
parser.add_argument("--use_wordpiece", type=int, default=0, help="if using wordpiece")
parser.add_argument("--use_mask", type=int, default=0, help="if masking the aspect words")

args = parser.parse_args()
