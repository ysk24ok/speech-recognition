import torch

from asr.las.speller import Attention, LSTMSpeller, AttentionSpeller


def test_Attention_forward():
    sequence_length = 5
    batch_size = 3
    hidden_size = 4
    speller_rnn_outputs = torch.randn(batch_size, hidden_size)
    listener_outputs = torch.randn(sequence_length, batch_size, hidden_size)
    attention = Attention()
    got = attention(speller_rnn_outputs, listener_outputs)
    assert got.shape == (batch_size, hidden_size)


def test_LSTMSpeller_forward():
    input_size = 8
    hidden_size = 16
    vocab_size = 10
    num_layers = 3
    sequence_length = 7
    batch_size = 4
    speller = LSTMSpeller(input_size, hidden_size, vocab_size, num_layers)
    inputs = torch.randn(sequence_length, batch_size, input_size)
    hidden_shape = num_layers, batch_size, hidden_size
    hidden = torch.zeros(*hidden_shape), torch.zeros(*hidden_shape)
    got = speller(inputs, hidden, None)
    assert got.shape == (sequence_length, batch_size, vocab_size)


def test_AttentionSpeller_forward():
    input_size = 8
    hidden_size = 16
    vocab_size = 10
    num_layers = 2
    sequence_length = 7
    listener_sequence_length = 13
    batch_size = 4
    speller = AttentionSpeller(input_size, hidden_size, vocab_size, num_layers)
    inputs = torch.randn(sequence_length, batch_size, input_size)
    hidden_shape = num_layers, batch_size, hidden_size
    hidden = torch.zeros(*hidden_shape), torch.zeros(*hidden_shape)
    listener_outputs = torch.randn(
        listener_sequence_length, batch_size, hidden_size)
    got = speller(inputs, hidden, listener_outputs)
    assert got.shape == (sequence_length, batch_size, vocab_size)
