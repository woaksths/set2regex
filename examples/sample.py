import os
import argparse
import logging

import time
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity, NLLLoss
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
from seq2seq.util.string_preprocess import get_set_num


try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--bidirectional', action='store_true', dest='bidirectional',
                    default=False,
                    help='Indicates if training model is bidirectional model or not')

parser.add_argument('--use_attn', action='store_true', dest='use_attn', default = False, help='use attention or not')
parser.add_argument('--attn_mode', action='store_true', dest='attn_mode', default = False, help='choose attention mode')


opt = parser.parse_args()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    
    train_file = opt.train_path
    valid_file = opt.dev_path
    set_num = get_set_num(train_file) 
    set_num = int(set_num/2)

    src = SourceField()
    tgt = TargetField()
    max_len = 50
    
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    train = torchtext.data.TabularDataset(
        path=train_file, format='tsv',
        fields= [('pos{}'.format(i+1), src) for i in range(set_num)] +
        [('neg{}'.format(i+1),src) for i in range(set_num)]+[('tgt', tgt)])
    
    dev = torchtext.data.TabularDataset(
        path=valid_file, format='tsv',
        fields=[('pos{}'.format(i+1),src) for i in range(set_num)] + 
        [('neg{}'.format(i+1),src) for i in range(set_num)]+[('tgt', tgt)])
        
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = NLLLoss(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size= 128
        bidirectional = opt.bidirectional
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size, dropout_p = 0.25,input_dropout_p = 0.25,
                             bidirectional=bidirectional, n_layers=2, variable_lengths=True, vocab = input_vocab)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size*2 if bidirectional else hidden_size,
                             dropout_p=0.2, input_dropout_p=0.25, use_attention=opt.use_attn, bidirectional=bidirectional, n_layers=1,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id, attn_mode = opt.attn_mode)
        seq2seq = Seq2seq(encoder, decoder)
        
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.1, 0.1)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters(), lr = 0.001), max_grad_norm=5)
        #scheduler = StepLR(optimizer.optimizer, 1)
        scheduler = ReduceLROnPlateau(optimizer.optimizer, 'min', factor = 0.1, verbose=True, patience=9)
        optimizer.set_scheduler(scheduler)
    expt_dir = opt.expt_dir + '_hidden_{}'.format(hidden_size)
    # train
    t = SupervisedTrainer(loss=loss, batch_size=2,
                          checkpoint_every=1800,
                          print_every=100, expt_dir=expt_dir, input_vocab=input_vocab, output_vocab=output_vocab)
    
    start_time = time.time()
    seq2seq = t.train(seq2seq, train,
                      num_epochs=100, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)
    end_time = time.time()
    print('total time > ', end_time-start_time)
    
predictor = Predictor(seq2seq, input_vocab, output_vocab)