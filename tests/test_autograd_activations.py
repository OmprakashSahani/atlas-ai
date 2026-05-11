from atlas_ai.autograd.value import Value


def test_relu_positive():

    x = Value(2.0)

    y = x.relu()

    y.backward()

    assert y.data == 2.0
    assert x.grad == 1.0


def test_relu_negative():

    x = Value(-2.0)

    y = x.relu()

    y.backward()

    assert y.data == 0
    assert x.grad == 0.0


def test_tanh():

    x = Value(0.5)

    y = x.tanh()

    y.backward()

    assert isinstance(y.data, float)
    assert isinstance(x.grad, float)
