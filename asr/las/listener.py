import torch


class ReduceTimeDirectionByHalf(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return ReduceTimeDirectionByHalf._forward_impl(inputs)

    @staticmethod
    def _forward_impl(inputs):
        seq_len = inputs.shape[0]
        even_indices, odd_indices = ReduceTimeDirectionByHalf._gen_indices(
            seq_len)
        return torch.cat(
            (inputs[even_indices, :, :], inputs[odd_indices, :, :]), dim=2)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, = ctx.saved_tensors
        return ReduceTimeDirectionByHalf._backward_impl(grad_outputs, inputs)

    @staticmethod
    def _backward_impl(grad_outputs, inputs):
        seq_len = inputs.shape[0]
        even_indices, odd_indices = ReduceTimeDirectionByHalf._gen_indices(
            seq_len)
        new_grad_outputs = torch.zeros(inputs.shape).to(grad_outputs.device)
        even_tensor, odd_tensor = grad_outputs.split(inputs.shape[2], dim=2)
        new_grad_outputs[even_indices] = even_tensor
        new_grad_outputs[odd_indices] = odd_tensor
        return new_grad_outputs

    @staticmethod
    def _gen_indices(seq_len):
        # NOTE: When the sequence length is odd, the last frame is ignored
        if seq_len % 2 == 1:
            seq_len -= 1
        even_indices = [i for i in range(0, seq_len, 2)]
        odd_indices = [i for i in range(1, seq_len, 2)]
        return (even_indices, odd_indices)


class Listener(torch.nn.Module):

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        size = (self.hidden_state_size, batch_size, self.hidden_size)
        return (weight.new_zeros(size), weight.new_zeros(size))


class LSTMListener(Listener):

    def __init__(self, input_size, hidden_size, num_layers,
                 bidirectional=True):
        super(LSTMListener, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional is True else 1
        self.hidden_state_size = num_layers * self.num_directions
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers,
                                  bidirectional=bidirectional)

    def forward(self, inputs, hidden):
        return self.lstm(inputs, hidden)


class PyramidalLSTMListener(Listener):

    """Pyramidal recurrent neural network encoder"""

    def __init__(self, input_size, hidden_size, num_layers,
                 bidirectional=True):
        super(PyramidalLSTMListener, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional is True else 1
        self.hidden_state_size = self.num_layers * self.num_directions
        self.lstm = torch.nn.LSTM(input_size, hidden_size,
                                  bidirectional=bidirectional)
        self.pyramidal_lstms = torch.nn.ModuleList([])
        for _ in range(0, num_layers-1):
            self.pyramidal_lstms.append(
                torch.nn.LSTM(hidden_size * self.num_directions * 2,
                              hidden_size, bidirectional=bidirectional)
            )

    def forward(self, inputs, hidden):
        """
        Args:
            inputs (torch.Tensor):
                shape=(sequence length, batch size, feature size)
            hidden (tuple[torch.Tensor]):
                First tensor is `h` and its shape is
                    (num layers * num_directions, batch_size, hidden_size)
                Second tensor is `c` and its shape is also
                    (num layers * num_directions, batch_size, hidden_size)
        Returns:
            torch.Tensor:
                shape=(reduced sequence length, batch size, hidden size)
            tuple[torch.Tensor]:
                First tensor is `h` and its shape is
                    (num layers * num_directions, batch_size, hidden_size)
                Second tensor is `c` and its shape is also
                    (num layers * num_directions, batch_size, hidden_size)
        """
        h, c = hidden
        new_h = torch.zeros(h.shape).to(h.device)
        new_c = torch.zeros(c.shape).to(c.device)
        s, e = 0, self.num_directions
        outputs, (new_h[s:e], new_c[s:e]) = self.lstm(inputs, (h[s:e], c[s:e]))
        for layer_id, lstm in enumerate(self.pyramidal_lstms):
            layer_id += 1
            s = layer_id * self.num_directions
            e = (layer_id + 1) * self.num_directions
            outputs = ReduceTimeDirectionByHalf.apply(outputs)
            outputs, (new_h[s:e], new_c[s:e]) = lstm(outputs, (h[s:e], c[s:e]))
        return outputs, (new_h, new_c)
