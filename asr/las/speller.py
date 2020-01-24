import torch


class Attention(torch.nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, speller_rnn_outputs, listener_outputs):
        """
        Args:
            speller_rnn_outputs (torch.Tensor)
                shape=(batch size, hidden size)
            listener_outputs (torch.Tensor)
                shape=(sequence length, batch size, hidden size)
        Returns:
            torch.Tensor: context vector
                shape=(batch size, hidden size)
        """
        # attention_weight is (batch size, sequence length, 1)
        attention_weight = torch.bmm(
            # (batch size, sequence length, hidden size)
            listener_outputs.transpose(0, 1),
            # (batch size, hidden size, 1)
            speller_rnn_outputs.unsqueeze(2)
        )
        attention_weight = torch.nn.functional.softmax(attention_weight, dim=1)
        # attention_context is (batch size, hidden size)
        attention_context = torch.bmm(
            # (batch size, hidden size, sequence size)
            listener_outputs.transpose(0, 1).transpose(1, 2),
            # (batch size, sequence size, 1)
            attention_weight
        ).squeeze(2)
        return attention_context


class Speller(torch.nn.Module):

    pass


class LSTMSpeller(Speller):

    def __init__(self, input_size, hidden_size, vocab_size, num_layers):
        super(LSTMSpeller, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers)
        self.linear = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs, hidden, _):
        outputs, _ = self.lstm(inputs, hidden)
        outputs = self.linear(outputs)
        return torch.nn.functional.softmax(outputs, dim=2)


class AttentionSpeller(Speller):

    def __init__(self, input_size, hidden_size, vocab_size, num_layers):
        super(AttentionSpeller, self).__init__()
        # TODO
        assert num_layers == 2, "num_layers is assumed as 2"
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstm1 = torch.nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = torch.nn.LSTM(hidden_size, hidden_size)
        self.attention = Attention()
        self.linear = torch.nn.Linear(hidden_size, self.vocab_size)

    def forward(self, inputs, hidden, listener_outputs):
        """
        Args:
            inputs (torch.Tensor):
                shape=(sequence length, batch size, embedding size)
            hidden (tuple[torch.Tensor]):
                First tensor is `h` and its shape is
                    (num layers, batch_size, hidden_size)
                Second tensor is `c` and its shape is also
                    (num layers, batch_size, hidden_size)
            listener_outputs (torch.Tensor):
                shape=(sequence length of listener, batch size, hidden size)
        Returns:
            torch.Tensor: shape=(sequence length, batch size, vocab size)
        """
        assert hidden[0].shape[0] == self.num_layers
        assert hidden[1].shape[0] == self.num_layers
        sequence_size, batch_size, _ = inputs.shape
        shape = sequence_size, batch_size, self.hidden_size
        outputs = torch.zeros(shape).to(inputs.device)
        # TODO: num_layers == 2 is assumed
        h1 = hidden[0][0]
        h2 = hidden[0][1:]
        c1 = hidden[1][0]
        c2 = hidden[1][1:]
        for i in range(sequence_size):
            h1, c1 = self.lstm1(inputs[i], (h1, c1))
            outputs[i] = self.attention(h1, listener_outputs)
        outputs, _ = self.lstm2(outputs, (h2, c2))
        outputs = self.linear(outputs)
        return torch.nn.functional.softmax(outputs, dim=2)
