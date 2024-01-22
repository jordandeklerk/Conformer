import numpy as np
import argparse
import random
import os
import gc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from torch.optim import AdamW, Adam
from torch.cuda.amp import autocast, GradScaler
from torch import Tensor
from torchmetrics.text.wer import WordErrorRate
from torch.utils.data import DataLoader

from utils import *
from model import *



parser = argparse.ArgumentParser("Conformer Librispeech")
parser.add_argument('--train_dir', type=str, default='hub://activeloop/LibriSpeech-train-clean-100', help='url for train data')
parser.add_argument('--test_dir', type=str, default='hub://activeloop/LibriSpeech-test-clean', help='url for test data')
parser.add_argument('--data_dir', type=str, default='./data', help='path to store the data')

parser.add_argument('--checkpoint_path', type=str, default='model_best.pt', help='path to store/load checkpoints')
parser.add_argument('--load_checkpoint', action='store_true', default=False, help='resume training from checkpoint')
parser.add_argument('--train_set', type=str, default='train-clean-100', help='train dataset')
parser.add_argument('--test_set', type=str, default='test-clean', help='test dataset')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--warmup_steps', type=float, default=10000, help='Multiply by sqrt(d_model) to get max_lr')
parser.add_argument('--peak_lr_ratio', type=int, default=0.05, help='Number of warmup steps for LR scheduler')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id (optional)')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--report_freq', type=int, default=100, help='training objective report frequency')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--use_amp', action='store_true', default=False, help='use mixed precision to train')
parser.add_argument('--attention_heads', type=int, default=4, help='number of heads to use for multi-head attention')
parser.add_argument('--d_input', type=int, default=80, help='dimension of the input (num filter banks)')
parser.add_argument('--d_encoder', type=int, default=144, help='dimension of the encoder')
parser.add_argument('--d_decoder', type=int, default=320, help='dimension of the decoder')
parser.add_argument('--encoder_layers', type=int, default=16, help='number of conformer blocks in the encoder')
parser.add_argument('--decoder_layers', type=int, default=1, help='number of decoder layers')
parser.add_argument('--conv_kernel_size', type=int, default=31, help='size of kernel for conformer convolution blocks')
parser.add_argument('--feed_forward_expansion_factor', type=int, default=4, help='expansion factor for conformer feed forward blocks')
parser.add_argument('--feed_forward_residual_factor', type=int, default=.5, help='residual factor for conformer feed forward blocks')
parser.add_argument('--dropout', type=float, default=.1, help='dropout factor for conformer model')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='model weight decay (corresponds to L2 regularization)')
parser.add_argument('--variational_noise_std', type=float, default=.0001, help='std of noise added to model weights for regularization')
parser.add_argument('--num_workers', type=int, default=2, help='num_workers for the dataloader')
parser.add_argument('--smart_batch', type=bool, default=True, help='Use smart batching for faster training')
parser.add_argument('--accumulate_iters', type=int, default=1, help='Number of iterations to accumulate gradients')
parser.add_argument('--seed', type=int, default=42, help='Seed for everything')

args, unknown = parser.parse_known_args()


class Trainer:
    def __init__(self, args, encoder, decoder, char_decoder, optimizer, scheduler, criterion, train_loader, test_loader):
        self.args = args
        self.encoder = encoder
        self.decoder = decoder
        self.char_decoder = char_decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.gpu = torch.cuda.is_available()
        self.grad_scaler = GradScaler(enabled=args.use_amp)
        self._setup()

    def _setup(self):
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        if self.gpu:
            torch.cuda.set_device(self.args.gpu)
            self.criterion = self.criterion.cuda()
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.char_decoder = self.char_decoder.cuda()
            torch.cuda.empty_cache()

        if self.args.load_checkpoint:
            self.start_epoch, self.best_loss = load_checkpoint(self.encoder, self.decoder, self.optimizer, self.scheduler, self.args.checkpoint_path)
            print(f'Resuming training from checkpoint starting at epoch {self.start_epoch}.')
        else:
            self.start_epoch = 0
            self.best_loss = float('inf')

    def train_epoch(self, epoch_progress_bar):
        wer = WordErrorRate()
        error_rate = AvgMeter()
        avg_loss = AvgMeter()
        text_transform = TextTransform()

        self.encoder.train()
        self.decoder.train()

        for i, batch in enumerate(self.train_loader):
            self.scheduler.step()
            gc.collect()
            spectrograms, labels, input_lengths, label_lengths, references, mask = batch

            if self.gpu:
                spectrograms = spectrograms.cuda()
                labels = labels.cuda()
                input_lengths = torch.tensor(input_lengths).cuda()
                label_lengths = torch.tensor(label_lengths).cuda()
                mask = mask.cuda()

            with autocast(enabled=self.args.use_amp):
                outputs = self.encoder(spectrograms, mask)
                outputs = self.decoder(outputs)
                loss = self.criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
            self.grad_scaler.scale(loss).backward()
            if (i+1) % self.args.accumulate_iters == 0:
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
            avg_loss.update(loss.detach().item())

            inds = self.char_decoder(outputs.detach())
            predictions = []
            for sample in inds:
                predictions.append(text_transform.int_to_text(sample))
            error_rate.update(wer(predictions, references) * 100)

            epoch_progress_bar.update(1)

        return error_rate.avg, avg_loss.avg

    def validate_epoch(self, epoch_progress_bar):
        avg_loss = AvgMeter()
        error_rate = AvgMeter()
        wer = WordErrorRate()
        text_transform = TextTransform()

        self.encoder.eval()
        self.decoder.eval()
        for i, batch in enumerate(self.test_loader):
            gc.collect()
            spectrograms, labels, input_lengths, label_lengths, references, mask = batch

            if self.gpu:
                spectrograms = spectrograms.cuda()
                labels = labels.cuda()
                input_lengths = torch.tensor(input_lengths).cuda()
                label_lengths = torch.tensor(label_lengths).cuda()
                mask = mask.cuda()

            with torch.no_grad():
                with autocast(enabled=self.args.use_amp):
                    outputs = self.encoder(spectrograms, mask)
                    outputs = self.decoder(outputs)
                    loss = self.criterion(F.log_softmax(outputs, dim=-1).transpose(0, 1), labels, input_lengths, label_lengths)
                avg_loss.update(loss.item())

                inds = self.char_decoder(outputs.detach())
                predictions = []
                for sample in inds:
                    predictions.append(text_transform.int_to_text(sample))
                error_rate.update(wer(predictions, references) * 100)

                epoch_progress_bar.update(1)

        return error_rate.avg, avg_loss.avg

    def train(self):
        print("\n--- Training In Progress ---\n")

        for epoch in range(self.start_epoch, self.args.epochs):
            torch.cuda.empty_cache()

            epoch_progress_bar = tqdm(total=len(self.train_loader) + len(self.test_loader), desc=f"Epoch {epoch + 1}/{self.args.epochs}")

            wer, loss = self.train_epoch(epoch_progress_bar)
            valid_wer, valid_loss = self.validate_epoch(epoch_progress_bar)

            if torch.is_tensor(wer):
                wer = wer.item()
            if torch.is_tensor(valid_wer):
                valid_wer = valid_wer.item()

            epoch_progress_bar.set_postfix({"Train Loss": loss, "Train WER": wer, "Val Loss": valid_loss, "Val WER": valid_wer})
            epoch_progress_bar.close()

            if valid_loss <= self.best_loss:
                print('Validation loss improved, saving checkpoint.')
                self.best_loss = valid_loss
                save_checkpoint(self.encoder, self.decoder, self.optimizer, self.scheduler, valid_loss, epoch+1, self.args.checkpoint_path)


def main():
    args, unknown = parser.parse_known_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print("\n--- GPU Information ---\n")

    if torch.cuda.is_available():
        print(f"Model is using device: {device}")
        print(f"CUDA Device: {torch.cuda.get_device_name(device)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024 ** 2} MB")
    else:
        print("Model is using CPU")

    print("\n--- Downloading Data ---\n")

    if not os.path.isdir(args.data_dir):
      os.mkdir(args.data_dir)
    train_data = torchaudio.datasets.LIBRISPEECH(root=args.data_dir, url=args.train_set, download=True)
    test_data = torchaudio.datasets.LIBRISPEECH(args.data_dir, url=args.test_set, download=True)

    train_loader = DataLoader(
                                dataset=train_data,
                                pin_memory=True,
                                num_workers=args.num_workers,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=lambda x: preprocess_example(x, 'train'))

    test_loader = DataLoader(
                              dataset=test_data,
                              pin_memory=True,
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=lambda x: preprocess_example(x, 'valid'))

    encoder = ConformerEncoder(
                            d_input=args.d_input,
                            d_model=args.d_encoder,
                            num_layers=args.encoder_layers,
                            conv_kernel_size=args.conv_kernel_size,
                            dropout=args.dropout,
                            feed_forward_residual_factor=args.feed_forward_residual_factor,
                            feed_forward_expansion_factor=args.feed_forward_expansion_factor,
                            num_heads=args.attention_heads).to(device)

    decoder = LSTMDecoder(
                            d_encoder=args.d_encoder,
                            d_decoder=args.d_decoder,
                            num_layers=args.decoder_layers).to(device)

    char_decoder = GreedyCharacterDecoder().eval()
    criterion = nn.CTCLoss(blank=28, zero_infinity=True)
    optimizer = get_adam_optimizer(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-4, eps=1e-05 if args.use_amp else 1e-09, weight_decay=args.weight_decay)
    scheduler = TransformerLrScheduler(optimizer, args.d_encoder, args.warmup_steps)

    trainer = Trainer(args, encoder, decoder, char_decoder, optimizer, scheduler, criterion, train_loader, test_loader)
    trainer.train()

if __name__ == "__main__":
    main()