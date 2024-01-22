class TransformerLrScheduler():
  def __init__(self, optimizer, d_model, warmup_steps, multiplier=5):
    self._optimizer = optimizer
    self.d_model = d_model
    self.warmup_steps = warmup_steps
    self.n_steps = 0
    self.multiplier = multiplier

  def step(self):
    self.n_steps += 1
    lr = self._get_lr()
    for param_group in self._optimizer.param_groups:
        param_group['lr'] = lr

  def _get_lr(self):
    return self.multiplier * (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * (self.warmup_steps ** (-1.5)))