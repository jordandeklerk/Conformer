import torchaudio
import torch
import torch.nn as nn
import os
import random

def load_checkpoint(encoder, decoder, optimizer, scheduler, checkpoint_path):
  ''' Load model checkpoint '''
  if not os.path.exists(checkpoint_path):
    raise 'Checkpoint does not exist'
  checkpoint = torch.load(checkpoint_path)
  scheduler.n_steps = checkpoint['scheduler_n_steps']
  scheduler.multiplier = checkpoint['scheduler_multiplier']
  scheduler.warmup_steps = checkpoint['scheduler_warmup_steps']
  encoder.load_state_dict(checkpoint['encoder_state_dict'])
  decoder.load_state_dict(checkpoint['decoder_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return checkpoint['epoch'], checkpoint['valid_loss']

def save_checkpoint(encoder, decoder, optimizer, scheduler, valid_loss, epoch, checkpoint_path):
  ''' Save model checkpoint '''
  torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'scheduler_n_steps': scheduler.n_steps,
            'scheduler_multiplier': scheduler.multiplier,
            'scheduler_warmup_steps': scheduler.warmup_steps,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)


class GreedyCharacterDecoder(nn.Module):
  ''' Greedy CTC decoder - Argmax logits and remove duplicates. '''
  def __init__(self):
    super(GreedyCharacterDecoder, self).__init__()

  def forward(self, x):
    indices = torch.argmax(x, dim=-1)
    indices = torch.unique_consecutive(indices, dim=-1)
    return indices.tolist()
  

class TextTransform:
  ''' Map characters to integers and vice versa '''
  def __init__(self):
    self.char_map = {}
    for i, char in enumerate(range(65, 91)):
      self.char_map[chr(char)] = i
    self.char_map["'"] = 26
    self.char_map[' '] = 27
    self.index_map = {}
    for char, i in self.char_map.items():
      self.index_map[i] = char

  def text_to_int(self, text):
      ''' Map text string to an integer sequence '''
      int_sequence = []
      for c in text:
        ch = self.char_map[c]
        int_sequence.append(ch)
      return int_sequence

  def int_to_text(self, labels):
      ''' Map integer sequence to text string '''
      string = []
      for i in labels:
          if i == 28:
            continue
          else:
            string.append(self.index_map[i])
      return ''.join(string)
  

class BatchSampler(object):
  ''' Sample contiguous, sorted indices. Leads to less padding and faster training. '''
  def __init__(self, sorted_inds, batch_size):
    self.sorted_inds = sorted_inds
    self.batch_size = batch_size

  def __iter__(self):
    inds = self.sorted_inds.copy()
    while len(inds):
      to_take = min(self.batch_size, len(inds))
      start_ind = random.randint(0, len(inds) - to_take)
      batch_inds = inds[start_ind:start_ind + to_take]
      del inds[start_ind:start_ind + to_take]
      yield batch_inds



def preprocess_example(data, data_type="train"):
  ''' Process raw LibriSpeech examples '''
  text_transform = TextTransform()
  train_audio_transform, valid_audio_transform = get_audio_transforms()
  spectrograms = []
  labels = []
  references = []
  input_lengths = []
  label_lengths = []
  for (waveform, _, utterance, _, _, _) in data:
    if data_type == 'train':
      spec = train_audio_transform(waveform).squeeze(0).transpose(0, 1)
    else:
      spec = valid_audio_transform(waveform).squeeze(0).transpose(0, 1)
    spectrograms.append(spec)

    references.append(utterance)
    label = torch.Tensor(text_transform.text_to_int(utterance))
    labels.append(label)

    input_lengths.append(((spec.shape[0] - 1) // 2 - 1) // 2)
    label_lengths.append(len(label))

  spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
  labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

  mask = torch.ones(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[1])
  for i, l in enumerate(input_lengths):
    mask[i, :, :l] = 0

  return spectrograms, labels, input_lengths, label_lengths, references, mask.bool()


def model_size(model, name):
  ''' Print model size in num_params and MB'''
  param_size = 0
  num_params = 0
  for param in model.parameters():
    num_params += param.nelement()
    param_size += param.nelement() * param.element_size()
  buffer_size = 0
  for buffer in model.buffers():
    num_params += buffer.nelement()
    buffer_size += buffer.nelement() * buffer.element_size()

  size_all_mb = (param_size + buffer_size) / 1024**2
  print(f'{name} - num_params: {round(num_params / 1000000, 2)}M,  size: {round(size_all_mb, 2)}MB')


def add_model_noise(model, std=0.0001, gpu=True):
  '''
    Add variational noise to model weights
  '''
  with torch.no_grad():
    for param in model.parameters():
        if gpu:
          param.add_(torch.randn(param.size()).cuda() * std)
        else:
          param.add_(torch.randn(param.size()).cuda() * std)


class AvgMeter(object):
  '''
    Keep running average for a metric
  '''
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = None
    self.sum = None
    self.cnt = 0

  def update(self, val, n=1):
    if not self.sum:
      self.sum = val * n
    else:
      self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def get_audio_transforms():
  time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=15, p=0.05) for _ in range(10)]
  train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
    *time_masks,
  )

  valid_audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160)

  return train_audio_transform, valid_audio_transform

train_audio_transform, valid_audio_transform = get_audio_transforms()


def find_max_transcript_length(ds):
    """
    Finds the length of the longest transcript in the dataset.

    :param ds: The dataset containing the transcripts.
    :return: The length of the longest transcript.
    """
    max_length = 0

    for transcript_tensor in train_data.transcripts:
        transcript_array = transcript_tensor.numpy()

        transcript = transcript_array.tobytes().decode('utf-8', errors='ignore')

        length = len(transcript.split())
        max_length = max(max_length, length)

    return max_length


