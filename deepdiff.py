import numpy as np
from collections import defaultdict

class node():
    # Initialize each node of the graph with links back to its parent inputs
    def __init__(self, *parents, name=None):
        self.parents = [p if isinstance(p, node) else constant(p) for p in parents]
        self.name = name

    def ancestors(self):
        return set([self] + [node for p in self.parents for node in p.ancestors()])

    # Derivatives and values can be recomputed if values change
    def clear(self):
        for node in self.parents: node.clear()
        self._value = None
        self._grads = None

    # Gradients are defined recursively using the chain rule
    def grad(self, x):
        if self == x:
            # If we've reached the input, return 1
            return constant(1)
        else:
            # Otherwise, apply the chain rule:
            # If y = w1 + w2, then dy/dx = (dy/dw1 * dw1/dx) + (dy/dw2 * dw2/dx)
            return add(*[
                multiply(self_grad_parent, parent.grad(x))
                    for self_grad_parent, parent in zip(self.grads, self.parents)])

    @property
    def value(self):
        if not hasattr(self, '_value') or self._value is None:
            self._value = self.compute_value()
        return self._value

    @property
    def grads(self):
        if not hasattr(self, '_grads') or self._grads is None:
            self._grads = self.compute_grads()
        return self._grads

    def __repr__(self):
        name = self.name or self.__class__.__name__
        return '{}(value={:.6f})'.format(name, self.value)

    def print_graph(self, indent=0):
        print('  '*indent, self)
        for node in self.parents:
            node.print_graph(indent+1)

    @property
    def parent_values(self):
        return [p.value for p in self.parents]

class constant(node):
    def __init__(self, value, name=None):
        self.name = name
        self._val = value
        self.parents = []
    def compute_value(self):
        return self._val
    def compute_grads(self):
        return []

class variable(constant):
    @property
    def value(self):
        if self._value is None:
            self._value = self.compute_value()
        return self._value
    @value.setter
    def value(self, val):
        self._val = val

class multiply(node):
    def compute_value(self):
        f, g = self.parent_values
        return f * g
    def compute_grads(self):
        f, g = self.parents
        return [g, f]

class add(node):
    def compute_value(self):
        return np.sum(self.parent_values)
    def compute_grads(self):
        return [constant(1) for _ in self.parents] # (f + g)' = f' + g'

class log(node):
    def compute_value(self):
        return np.log(self.parent_values[0])
    def compute_grads(self):
        return power(self.parent_values[0], -1)

class power(node):
    def compute_value(self):
        base, exp = self.parent_values
        return base ** exp
    def compute_grads(self):
        base, exp = self.parents
        return [multiply(exp, power(base, add(exp, -1))),
                multiply(self, log(base))]

class sigmoid(node):
    def compute_value(self):
        from scipy.special import expit
        return expit(self.parent_values[0])
    def compute_grads(self):
        return [multiply(self, add(1, multiply(-1, self)))] # σ'(x) = σ(x)(1 - σ(x))

class relu(node):
    def compute_value(self):
        x = self.parent_values[0]
        if x > 0:
            return x
        else:
            return 0
    def compute_grads(self):
        x = self.parent_values[0]
        if x > 0:
            return [constant(1)]
        else:
            return [constant(0)]

def vector(name, length, initializer=np.random.normal):
    return [variable(initializer(), name=name + '_{}'.format(i+1))
           for i in range(length)]

def matrix(name, cols, rows, initializer=np.random.normal):
    return [[variable(initializer(), name=name + '_{}{}'.format(i+1, j+1))
            for j in range(cols)]
            for i in range(rows)]

def dot(x, A):
    return [
        add(*[multiply(A[j][i], x[i])
             for j in range(len(A))])
             for i in range(len(x)) ]

def elemwise(fn, *vectors):
    return [fn(*els) for els in zip(*vectors)]

def gradient_descent(loss, vars, learning_rate=0.1, steps=250):
    history = defaultdict(list)

    for step in range(steps):
        # record the current state of all values in the network, for plotting
        for node in loss.ancestors():
            history[node].append(node.value)

        # compute gradients wrt. each variable
        grads = [loss.grad(var) for var in vars]

        # update each variable based on the gradient and learning rate
        for grad, var in zip(grads, vars):
            var.value = var.value - learning_rate * grad

        # recompute the loss and gradients
        loss.recompute()

    return history

if __name__ == '__main__':
    x = constant(0.5)
    y = sigmoid(x)
    import pdb; pdb.set_trace()
    pass
