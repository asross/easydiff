import numpy as np
from collections import defaultdict

class node():
    # Initialize each node of the graph with links back to its parent inputs
    def __init__(self, *parents, name=None):
        self.parents = [p if isinstance(p, node) else constant(p) for p in parents]
        self.name = name
        self.compute()

    def ancestors(self):
        return set([self] + [node for p in self.parents for node in p.ancestors()])

    # Each node will have to define how it's computed
    def compute(self):
        raise("Not implemented")

    # Derivatives and values can be recomputed if values change
    def recompute(self):
        for node in self.parents: node.recompute()
        self.compute()

    # Gradients are defined recursively using the chain rule
    def grad(self, x):
        if self == x:
            # If we've reached the input, return 1
            return 1
        else:
            # Otherwise, apply the chain rule:
            # If y = w1 + w2, then dy/dx = (dy/dw1 * dw1/dx) + (dy/dw2 * dw2/dx)
            return np.sum([
                self_grad_parent * parent.grad(x)
                    for self_grad_parent, parent in zip(self.grads, self.parents)])

    def __repr__(self):
        name = self.name or self.__class__.__name__
        return '{}(value={:.2f})'.format(name, self.value)

    def print_graph(self, indent=0):
        print('  '*indent, self)
        for node in self.parents:
            node.print_graph(indent+1)

    @property
    def parent_values(self):
        return [p.value for p in self.parents]

class variable(node):
    def compute(self): pass
    def __init__(self, value, name=None):
        self.name = name
        self.value = value
        self.grads = []
        self.parents = []

class constant(node):
    def compute(self): pass
    def __init__(self, value, name=None):
        self.name = name
        self._value = value
        self.grads = []
        self.parents = []

    @property
    def value(self):
        return self._value # only a getter, no setter

class multiply(node):
    def compute(self):
        f, g = self.parent_values
        self.value = f * g
        self.grads = [g, f] # (fg)' = f'g + g'f

class power(node):
    def compute(self):
        base, exp = self.parent_values
        self.value = base ** exp
        self.grads = [exp * base ** (exp-1), self.value * np.log(base)]

class add(node):
    def compute(self):
        self.value = np.sum(self.parent_values)
        self.grads = [1 for _ in self.parents] # (f + g)' = f' + g'

class sigmoid(node):
    def compute(self):
        from scipy.special import expit
        self.value = expit(self.parent_values[0])
        self.grads = [self.value * (1 - self.value)] # σ'(x) = σ(x)(1 - σ(x))

class relu(node):
    def compute(self):
        x = self.parent_values[0]
        if x > 0:
            self.value = x
            self.grads = [1]
        else:
            self.value = 0
            self.grads = [0]

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
