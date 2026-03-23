import jax.numpy as jnp
from flax import nnx
from random import random, randint
from typing import Literal


class DeepQAgent:
    def __init__(self, state_size, action_size):
        self.active_network = QNetwork(rngs=nnx.Rngs(0)) # the network that will be used to update the loss function
        self.eval_network = QNetwork(rngs=nnx.Rngs(0))

    def act(self, state, epsilon=0.05):
        if random() < epsilon:
            return randint(0, 3)

        y = self.active_network(state)
        argmax = jnp.argmax(y)
        print(y, int(argmax))
        return int(argmax)
    

    def get_action_values(
        self,
        state,
        model: Literal["active", "eval"]="active"
    ):
        m = self.active_network if model == "active" else self.eval_network
        return m(state)



class QNetwork(nnx.Module):
    def __init__(self, rngs: nnx.Rngs, d_in=8, d_hidden_1=128, d_hidden_2=128, d_out=4):
        self.lin_in = nnx.Linear(d_in, d_hidden_1, rngs=rngs)
        self.hidden_1 = nnx.Linear(d_hidden_1, d_hidden_2, rngs=rngs)
        self.hidden_2 = nnx.Linear(d_hidden_2, d_hidden_1, rngs=rngs)
        self.lin_out = nnx.Linear(d_hidden_1, d_out, rngs=rngs)


    def __call__(self, x):
        x = nnx.relu(self.lin_in(x))
        x = nnx.relu(self.hidden_1(x))
        x = nnx.relu(self.hidden_2(x))
        return self.lin_out(x)

def rmse_loss(self, pred, y):
    return jnp.sqrt(jnp.mean((pred - y) ** 2))
