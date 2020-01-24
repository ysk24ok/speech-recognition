import torch

from asr.las.listener import (
    ReduceTimeDirectionByHalf,
    LSTMListener,
    PyramidalLSTMListener
)


class TestReduceTimeDirectionByHalf(object):

    def _exec_forward_impl(self, forward_input_shape, forward_output_shape):
        sequence_length, batch_size, input_size = forward_input_shape
        inputs = torch.randn(*forward_input_shape)
        got = ReduceTimeDirectionByHalf._forward_impl(inputs)
        assert got.shape == forward_output_shape
        for batch_idx in range(batch_size):
            for seq_idx in range(forward_output_shape[0]):
                assert torch.allclose(
                    got[seq_idx, batch_idx, 0:input_size],
                    inputs[seq_idx*2, batch_idx, :]
                ) is True
                assert torch.allclose(
                    got[seq_idx, batch_idx, input_size:input_size*2],
                    inputs[seq_idx*2+1, batch_idx, :]
                ) is True

    def _exec_backward_impl(self, forward_input_shape, forward_output_shape):
        _, batch_size, input_size = forward_input_shape
        grad_outputs = torch.randn(*forward_output_shape)
        inputs = torch.randn(*forward_input_shape)
        got = ReduceTimeDirectionByHalf._backward_impl(grad_outputs, inputs)
        assert got.shape == forward_input_shape
        for batch_idx in range(batch_size):
            for seq_idx in range(grad_outputs.shape[0]):
                assert torch.allclose(
                    grad_outputs[seq_idx, batch_idx, 0:input_size],
                    got[seq_idx*2, batch_idx, :]
                ) is True
                assert torch.allclose(
                    grad_outputs[seq_idx, batch_idx, input_size:input_size*2],
                    got[seq_idx*2+1, batch_idx, :]
                ) is True

    def test_forward_impl_and_backward_impl_with_even_sequence_length(self):
        sequence_length, batch_size, input_size = 10, 2, 4
        forward_input_shape = sequence_length, batch_size, input_size
        forward_output_shape = (
            int(sequence_length / 2), batch_size, input_size * 2
        )
        self._exec_forward_impl(forward_input_shape, forward_output_shape)
        self._exec_backward_impl(forward_input_shape, forward_output_shape)

    def test_forward_impl_and_backward_impl_with_odd_sequence_length(self):
        sequence_length, batch_size, input_size = 11, 2, 4
        forward_input_shape = sequence_length, batch_size, input_size
        forward_output_shape = (
            int(sequence_length / 2), batch_size, input_size * 2
        )
        self._exec_forward_impl(forward_input_shape, forward_output_shape)
        self._exec_backward_impl(forward_input_shape, forward_output_shape)

    def _exec_forward_and_backward(self, forward_input_shape):
        _, batch_size, input_size = forward_input_shape
        inputs = torch.randn(*forward_input_shape, requires_grad=True)
        got = ReduceTimeDirectionByHalf.apply(inputs)
        loss = torch.nn.MSELoss()
        target = torch.randn(batch_size, input_size * 2)
        output = loss(got[-1], target)
        output.backward()

    def test_forward_and_backward_with_even_sequence_length(self):
        sequence_length, batch_size, input_size = 10, 2, 4
        self._exec_forward_and_backward(
            (sequence_length, batch_size, input_size))

    def test_forward_and_backward_with_odd_sequence_length(self):
        sequence_length, batch_size, input_size = 11, 2, 4
        self._exec_forward_and_backward(
            (sequence_length, batch_size, input_size))


class TestLSTMListener(object):

    def _exec_forward(self, listener, forward_input_shape):
        sequence_length, batch_size, input_size = forward_input_shape
        data = torch.randn(*forward_input_shape)
        hidden = listener.init_hidden(batch_size)
        got_data, got_hidden = listener(data, hidden)
        expected_shape = (
            sequence_length, batch_size,
            listener.hidden_size * listener.num_directions)
        assert got_data.shape == expected_shape
        expected_hidden_shape = (
            listener.num_layers * listener.num_directions, batch_size,
            listener.hidden_size)
        assert got_hidden[0].shape == expected_hidden_shape
        assert got_hidden[1].shape == expected_hidden_shape

    def test_forward_unidirectional(self):
        sequence_length, batch_size, input_size = 16, 2, 10
        hidden_size, num_layers = 4, 3
        listener = LSTMListener(input_size, hidden_size, num_layers,
                                bidirectional=False)
        self._exec_forward(listener, (sequence_length, batch_size, input_size))

    def test_forward_bidirectional(self):
        sequence_length, batch_size, input_size = 16, 2, 10
        hidden_size, num_layers = 4, 3
        listener = LSTMListener(input_size, hidden_size, num_layers)
        self._exec_forward(listener, (sequence_length, batch_size, input_size))


class TestPyramidalLSTMListener(object):

    def _exec_forward_single_layer(self, listener, forward_input_shape):
        sequence_length, batch_size, input_size = forward_input_shape
        hidden = listener.init_hidden(batch_size)
        data = torch.randn(sequence_length, batch_size, input_size)
        got_outputs, got_hidden = listener(data, hidden)
        expected_shape = (
            sequence_length, batch_size,
            listener.hidden_size * listener.num_directions)
        assert got_outputs.shape == expected_shape
        expected_hidden_shape = (
            listener.num_layers * listener.num_directions, batch_size,
            listener.hidden_size)
        assert got_hidden[0].shape == expected_hidden_shape
        assert got_hidden[1].shape == expected_hidden_shape

    def test_forward_single_layer_unidirectional(self):
        sequence_length, batch_size, input_size = 16, 2, 10
        hidden_size, num_layers = 4, 1
        listener = PyramidalLSTMListener(input_size, hidden_size, num_layers,
                                         bidirectional=False)
        self._exec_forward_single_layer(
            listener, (sequence_length, batch_size, input_size))

    def test_forward_single_layer_bidirectional(self):
        sequence_length, batch_size, input_size = 16, 2, 10
        hidden_size, num_layers = 4, 1
        listener = PyramidalLSTMListener(input_size, hidden_size, num_layers)
        self._exec_forward_single_layer(
            listener, (sequence_length, batch_size, input_size))

    def _exec_forward_mutiple_layers(self, listener, forward_input_shape):
        sequence_length, batch_size, input_size = forward_input_shape
        hidden = listener.init_hidden(batch_size)
        inputs = torch.randn(sequence_length, batch_size, input_size)
        got_outputs, got_hidden = listener(inputs, hidden)
        expected_shape = (
            sequence_length / (2 ** (listener.num_layers - 1)), batch_size,
            listener.hidden_size * listener.num_directions)
        assert got_outputs.shape == expected_shape
        expected_hidden_shape = (
            listener.num_layers * listener.num_directions, batch_size,
            listener.hidden_size)
        assert got_hidden[0].shape == expected_hidden_shape
        assert got_hidden[1].shape == expected_hidden_shape

    def test_forward_multiple_layers_unidirectional(self):
        sequence_length, batch_size, input_size = 16, 2, 10
        hidden_size, num_layers = 4, 3
        listener = PyramidalLSTMListener(input_size, hidden_size, num_layers,
                                         bidirectional=False)
        self._exec_forward_mutiple_layers(
            listener, (sequence_length, batch_size, input_size))

    def test_forward_multiple_layers_bidirectional(self):
        sequence_length, batch_size, input_size = 16, 2, 10
        hidden_size, num_layers = 4, 3
        listener = PyramidalLSTMListener(input_size, hidden_size, num_layers)
        self._exec_forward_mutiple_layers(
            listener, (sequence_length, batch_size, input_size))
