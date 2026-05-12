"""
This class implements generative recurrent neural network with augmented memory
stack as proposed in https://arxiv.org/abs/1503.01007
There are options of using LSTM or GRU, as well as using the generator without
memory stack.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import time
from tqdm import trange

from generators.release.utils import time_since
from generators.release.smiles_enumerator import SmilesEnumerator


class StackAugmentedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_type='GRU',
                 n_layers=1, is_bidirectional=False, has_stack=False,
                 stack_width=None, stack_depth=None, use_cuda=None,
                 optimizer_instance=torch.optim.Adadelta, lr=0.01):
        """
        Constructor for the StackAugmentedRNN object.
        [... docstring unchanged ...]
        """
        super(StackAugmentedRNN, self).__init__()
        
        if layer_type not in ['GRU', 'LSTM']:
            raise InvalidArgumentError('Layer type must be GRU or LSTM')
        self.layer_type = layer_type
        self.is_bidirectional = is_bidirectional
        if self.is_bidirectional:
            self.num_dir = 2
        else:
            self.num_dir = 1
        if layer_type == 'LSTM':
            self.has_cell = True
        else:
            self.has_cell = False
        self.has_stack = has_stack
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if self.has_stack:
            self.stack_width = stack_width
            self.stack_depth = stack_depth

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

        self.n_layers = n_layers
        
        if self.has_stack:
            self.stack_controls_layer = nn.Linear(in_features=self.hidden_size *
                                                              self.num_dir,
                                                  out_features=3)

            self.stack_input_layer = nn.Linear(in_features=self.hidden_size *
                                                           self.num_dir,
                                               out_features=self.stack_width)

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.has_stack:
            rnn_input_size = hidden_size + stack_width
        else:
            rnn_input_size = hidden_size
        if self.layer_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_size, hidden_size, n_layers,
                               bidirectional=self.is_bidirectional)
            self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        elif self.layer_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_size, hidden_size, n_layers,
                             bidirectional=self.is_bidirectional)
            self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        
        if self.use_cuda:
            self = self.cuda()
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = self.optimizer_instance(self.parameters(), lr=lr,
                                                 weight_decay=0.00001)

        # [PRINT] Post-init memory accounting — prints the CPU RAM consumed by
        # each layer's parameters. Analogous to TS printing reagent pool size at
        # startup. Critical for diagnosing OOM kills: the total here plus
        # optimizer state (2x for Adadelta) is the minimum RAM floor of the process.
        print(f"[StackAugmentedRNN.__init__] layer breakdown (fp32 weights):")
        total_bytes = 0
        for name, param in self.named_parameters():
            nb = param.numel() * 4
            total_bytes += nb
            print(f"  {name:40s}  shape={str(list(param.shape)):25s}  "
                  f"{nb/1e6:.3f} MB")
        print(f"[StackAugmentedRNN.__init__] total weights          = {total_bytes/1e6:.2f} MB")
        print(f"[StackAugmentedRNN.__init__] Adadelta optimizer est = {total_bytes*2/1e6:.2f} MB  (2x weights)")
        print(f"[StackAugmentedRNN.__init__] peak CPU RAM at init   = ~{total_bytes*3/1e6:.0f} MB  (weights + optimizer)")
        if self.has_stack:
            stack_bytes = 1 * stack_depth * stack_width * 4
            print(f"[StackAugmentedRNN.__init__] stack tensor/forward   = {stack_bytes/1e6:.2f} MB  "
                  f"(depth={stack_depth} x width={stack_width})")
        print(f"[StackAugmentedRNN.__init__] use_cuda={self.use_cuda}")
  
    def load_model(self, path):
        """
        Loads pretrained parameters from the checkpoint into the model.
        """
        # [PRINT] Checkpoint load — torch.load() with default map_location pulls
        # the entire checkpoint into CPU RAM first, even if the model is on GPU.
        # This creates a temporary peak = GPU model copy + CPU checkpoint copy.
        # For this architecture (~90MB weights) the peak is ~180MB extra RAM.
        print(f"[load_model] loading checkpoint from: {path}")
        weights = torch.load(path)
        checkpoint_bytes = sum(v.numel() * v.element_size() for v in weights.values())
        print(f"[load_model] checkpoint size on CPU = {checkpoint_bytes/1e6:.2f} MB")
        print(f"[load_model] map_location=None: checkpoint loaded to CPU, then copied to model device")
        self.load_state_dict(weights)
        print(f"[load_model] state_dict loaded successfully")

    def save_model(self, path):
        """
        Saves model parameters into the checkpoint file.
        """
        torch.save(self.state_dict(), path)
        print(f"[save_model] checkpoint saved to: {path}")

    def change_lr(self, new_lr):
        """
        Updates learning rate of the optimizer.
        """
        # [PRINT] Learning rate change — mirrors TS's fixed known_var: both control
        # how aggressively new evidence updates the model. A smaller lr = heavier
        # prior weighting, exactly like larger known_var in TS Bayesian updates.
        print(f"[change_lr] lr: {self.lr} -> {new_lr}")
        self.optimizer = self.optimizer_instance(self.parameters(), lr=new_lr)
        self.lr = new_lr

    def forward(self, inp, hidden, stack):
        """
        Forward step of the model.
        """
        inp = self.encoder(inp.view(1, -1))
        if self.has_stack:
            if self.has_cell:
                hidden_ = hidden[0]
            else:
                hidden_ = hidden
            if self.is_bidirectional:
                hidden_2_stack = torch.cat((hidden_[0], hidden_[1]), dim=1)
            else:
                hidden_2_stack = hidden_.squeeze(0)
            stack_controls = self.stack_controls_layer(hidden_2_stack)
            stack_controls = F.softmax(stack_controls, dim=1)

            # [PRINT] Stack operation probabilities — PUSH/POP/NO_OP weights per
            # forward step. The stack is what separates this from a plain GRU:
            # it provides unbounded memory for long-range SMILES dependencies
            # (e.g. matching opening/closing brackets). When PUSH dominates the
            # model is writing context; when POP dominates it is reading back.
            # Print only when called from evaluate() to avoid flooding fit() output.
            # To enable: change the condition below or set a module-level flag.
            # print(f"[forward] stack_controls: PUSH={stack_controls[0,0].item():.4f}  "
            #       f"POP={stack_controls[0,1].item():.4f}  "
            #       f"NO_OP={stack_controls[0,2].item():.4f}")

            stack_input = self.stack_input_layer(hidden_2_stack.unsqueeze(0))
            stack_input = torch.tanh(stack_input)
            stack = self.stack_augmentation(stack_input.permute(1, 0, 2),
                                            stack, stack_controls)
            stack_top = stack[:, 0, :].unsqueeze(0)
            inp = torch.cat((inp, stack_top), dim=2)
        output, next_hidden = self.rnn(inp.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, next_hidden, stack

    def stack_augmentation(self, input_val, prev_stack, controls):
        """
        Augmentation of the tensor into the stack.
        """
        batch_size = prev_stack.size(0)

        controls = controls.view(-1, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
        if self.use_cuda:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom.cuda())
        else:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom)
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
        return new_stack

    def init_hidden(self):
        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size))

    def init_cell(self):
        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size))

    def init_stack(self):
        result = torch.zeros(1, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return Variable(result.cuda())
        else:
            return Variable(result)

    def train_step(self, inp, target):
        """
        One train step: forward-backward and parameters update.
        """
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None
        self.optimizer.zero_grad()
        loss = 0

        # [PRINT] Pre-step hidden state norm — the hidden state is the GRU's
        # working memory at the start of each sequence. Its norm indicates how
        # "activated" the network is. A steadily growing norm across training
        # steps is an early warning of exploding gradients / loss of stability.
        # This is the neural analogue of tracking mu before a TS update.
        if isinstance(hidden, tuple):
            h_norm_pre = hidden[0].norm().item()
        else:
            h_norm_pre = hidden.norm().item()
        # (printed below alongside post-step values for compactness)

        for c in range(len(inp)):
            output, hidden, stack = self(inp[c], hidden, stack)
            loss += self.criterion(output, target[c].unsqueeze(0))

        loss.backward()

        # [PRINT] Gradient norms per layer — the most important diagnostic for
        # understanding the RL update dynamics. Analogous to TS's delta_mu/delta_std:
        # it shows how much each layer is being shifted by a single training step.
        # Large grad norms in encoder = the model is revising token embeddings heavily.
        # Large grad norms in rnn = the recurrent weights are being updated aggressively.
        # Near-zero grad norms = vanishing gradients; that layer is not learning.
        total_grad_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                g = param.grad.norm().item()
                total_grad_norm += g ** 2
                print(f"[train_step] grad_norm | layer={name:40s} | grad_norm={g:.6f}")
        total_grad_norm = total_grad_norm ** 0.5
        print(f"[train_step] total_grad_norm (all layers) = {total_grad_norm:.6f}")

        self.optimizer.step()

        # [PRINT] Post-step hidden state norm — compare to pre-step norm above to
        # see how much the sequence processing shifted the hidden state. Also print
        # weight norms for the key layers (encoder, rnn, decoder) to track cumulative
        # drift from the pretrained prior, exactly like TS's post-update mu/std print.
        if isinstance(hidden, tuple):
            h_norm_post = hidden[0].norm().item()
        else:
            h_norm_post = hidden.norm().item()
        print(f"[train_step] hidden_norm: pre={h_norm_pre:.4f} -> post={h_norm_post:.4f} "
              f"(delta={h_norm_post - h_norm_pre:+.4f})")
        print(f"[train_step] seq_len={len(inp)} | loss={loss.item()/len(inp):.6f}")

        # [PRINT] Key weight norms after optimizer.step() — encoder, rnn, decoder.
        # Watching these drift from their values at load_model() time tells you
        # how far RL fine-tuning has pushed the policy away from its pretrained state.
        for name, param in self.named_parameters():
            if any(k in name for k in ['encoder', 'rnn', 'decoder']):
                print(f"[train_step] weight_norm | {name:40s} | norm={param.data.norm().item():.4f}")

        return loss.item() / len(inp)
    
    def evaluate(self, data, prime_str='<', end_token='>', predict_len=100):
        """
        Generates new string from the model distribution.
        """
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None
        prime_input = data.char_tensor(prime_str)
        new_sample = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str)-1):
            _, hidden, stack = self.forward(prime_input[p], hidden, stack)
        inp = prime_input[-1]

        # [PRINT] Generation trace — prints the sampled token and its probability
        # at each step. This is the direct analogue of TS's winner_idx line:
        # it shows what the learned distribution chose at each position and
        # how confident it was. Low max_prob = high entropy = the policy is
        # uncertain / exploring. High max_prob = the policy has collapsed to
        # near-deterministic output (mode collapse warning).
        print(f"[evaluate] prime='{prime_str}' | predict_len={predict_len}")
        for p in range(predict_len):
            output, hidden, stack = self.forward(inp, hidden, stack)

            probs = torch.softmax(output, dim=1)
            top_i = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()

            predicted_char = data.all_characters[top_i]
            max_prob = probs.max().item()
            entropy = -(probs * torch.log(probs + 1e-9)).sum().item()

            # [PRINT] Per-token sampling: character chosen, its probability, and
            # the output distribution entropy. Entropy is the key exploration signal:
            #   high entropy (~3-4 bits for 45-token vocab) = policy is uncertain, exploring
            #   low entropy (~0) = policy has collapsed, same token always predicted
            print(f"[evaluate] step={p+1:3d} | token='{predicted_char}' | "
                  f"prob={max_prob:.4f} | entropy={entropy:.4f} | "
                  f"prefix='{new_sample[-10:]}'")

            new_sample += predicted_char
            inp = data.char_tensor(predicted_char)
            if predicted_char == end_token:
                break

        print(f"[evaluate] final_sample='{new_sample}' | total_tokens={len(new_sample)}")
        return new_sample

    def fit(self, data, n_iterations, all_losses=[], print_every=100,
            plot_every=10, augment=False):
        """
        Fits the parameters of the model (training loop).
        """
        start = time.time()
        loss_avg = 0

        if augment:
            smiles_augmentation = SmilesEnumerator()
        else:
            smiles_augmentation = None

        # [PRINT] Training session header — records the starting weight norms for
        # encoder, rnn, decoder. These are the baseline to compare against at the
        # end of training to measure total policy drift from pretrained prior.
        # Directly analogous to TS's warmup prior_mean/prior_std snapshot.
        print(f"\n[fit] === training session start | n_iterations={n_iterations} ===")
        for name, param in self.named_parameters():
            if any(k in name for k in ['encoder', 'rnn', 'decoder']):
                print(f"[fit] start weight_norm | {name:40s} | norm={param.data.norm().item():.4f}")

        for epoch in trange(1, n_iterations + 1, desc='[stackRNN.py] Training',
                            leave=False, ncols=80, unit='epochs'):
            inp, target = data.random_training_set(smiles_augmentation)
            loss = self.train_step(inp, target)
            loss_avg += loss

            if epoch % print_every == 0:
                # [PRINT] Periodic loss + sample + weight norm summary.
                # loss: the cross-entropy signal driving policy updates (the RL reward proxy).
                # loss_avg: smoothed learning curve — flat loss_avg = training has converged
                #   or collapsed. Sudden spikes = gradient instability.
                # weight_norm snapshot: cumulative drift since training started.
                elapsed = time_since(start)
                print(f"\n[fit] epoch={epoch}/{n_iterations} ({epoch/n_iterations*100:.1f}%) | "
                      f"loss={loss:.6f} | loss_avg={loss_avg/min(epoch, plot_every):.6f} | "
                      f"elapsed={elapsed}")
                for name, param in self.named_parameters():
                    if any(k in name for k in ['encoder', 'rnn', 'decoder']):
                        print(f"[fit] weight_norm | {name:40s} | norm={param.data.norm().item():.4f}")
                sample = self.evaluate(data=data, prime_str='<', predict_len=100)
                print(f"[fit] sample: '{sample}'")
                print()

                print('[%s (%d %d%%) %.4f]' % (elapsed, epoch,
                                               epoch / n_iterations * 100, loss))
                print(self.evaluate(data=data, prime_str = '<',
                                    predict_len=100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

        # [PRINT] Training session end — final weight norms. Delta vs start norms
        # shows total policy shift from the pretrained prior. Large delta = the RL
        # loop significantly changed the generator; small delta = fine-tuning was
        # conservative. Analogous to TS's before/after mu/std across all warmup replays.
        print(f"\n[fit] === training session end | n_iterations={n_iterations} ===")
        for name, param in self.named_parameters():
            if any(k in name for k in ['encoder', 'rnn', 'decoder']):
                print(f"[fit] end weight_norm | {name:40s} | norm={param.data.norm().item():.4f}")

        return all_losses