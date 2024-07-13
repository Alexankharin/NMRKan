import torch
import sympy as sp
from typing import Tuple, Dict


class DenseKanLayer(torch.nn.Module):
    def __init__(
        self,
        N_input: int,
        N_output: int,
        activations: list = None,
        activations_reprs: dict = None,
        normalize: bool = True,
    ) -> None:
        super(DenseKanLayer, self).__init__()
        """
        layer with N_input inputs and N_output outputs
        activations: list of activation functions to apply to the output of the linear layer
        activations_reprs: dictionary of activation function representations for symbolic formula
        """
        self.N_input = N_input
        self.N_output = N_output
        self.normalize = normalize
        if activations is None:
            activations = [
                torch.nn.Identity(),
                torch.square,
                lambda x: x * 0.0,
            ]
        self.activations = activations
        if activations_reprs is None:
            self.activation_reprs = {0: "", 1: "^2", 2: "*0"}
        else:
            self.activation_reprs = activations_reprs
        self.linear = torch.nn.Linear(N_input, N_output * len(activations))
        # separable convolution over the Channel dimension
        self.regression_layer = torch.nn.Conv1d(
            in_channels=N_output,
            out_channels=N_output,
            kernel_size=len(activations),
            groups=N_output,
            bias=False,
        )

    def forward(self, x: torch.Tensor):
        """
        x: B x N_input
        output is B x N_output
        """
        # B x N_input
        x = self.linear(x)
        # B x len(activations) x N_output
        x = x.reshape(-1, self.N_output, len(self.activations))
        # B x len(activations) x N_output
        x = torch.stack([a(x[:, :, i]) for i, a in enumerate(self.activations)], dim=-1)
        # normalize regress_activations
        self.regression_layer.weight.data = self.regression_layer.weight.data.abs()
        # check if need to normalize (sum is greater than 1)
        if self.normalize:
            with torch.no_grad():
                normmap = (
                    torch.sum(self.regression_layer.weight.data, dim=2, keepdim=True)
                    > -1.0
                ).float()
                self.regression_layer.weight.data = (
                    self.regression_layer.weight.data * (1 - normmap)
                    + self.regression_layer.weight.data
                    * (normmap)
                    / (
                        torch.sum(
                            self.regression_layer.weight.data, dim=2, keepdim=True
                        )
                    )
                )
        x_regressed = self.regression_layer(x).squeeze(-1)
        return x_regressed

    def L1_reg(self):
        """
        L1 regularization on the regression layer
        """
        return torch.sum(torch.abs(self.regression_layer.weight))

    def simplify(self, threshold=1e-2) -> None:
        """
        set the weights of the regression layer to zero if they are below the threshold
        set corresponding weights in the linear layer to zero
        """
        self.regression_layer.weight.data[
            torch.abs(self.regression_layer.weight.data) < threshold
        ] = 0
        with torch.no_grad():
            self.linear.weight.data = (
                self.regression_layer.weight.data.reshape(1, -1).abs() > threshold
            ).T * self.linear.weight.data
        return None

    def symbolic_formula(self):
        """
        return a dictionary of symbolic formulas for each output of the regression layer
        """
        formula = {}
        for i in range(self.linear.out_features):
            symbolic_str = "0"
            for j in range(self.linear.in_features):
                if self.linear.weight[i, j].abs() > 0:
                    symbolic_str += f"+({self.linear.weight[i,j]})*x_{j}"
            # parse with sympy
            symbolic_str += f"+({self.linear.bias[i].item()})"
            symbolic_str = sp.sympify(symbolic_str)
            formula[f"y_{i}"] = symbolic_str
        for i in range(self.regression_layer.weight.shape[0]):
            symbolic_str = "0"
            for j in range(self.regression_layer.weight.shape[-1]):
                number = i * (self.regression_layer.weight.shape[-1]) + j
                if self.regression_layer.weight[i, 0, j].abs() > 0:
                    symbolic_str += f"+({self.regression_layer.weight[i,0,j]})*y_{number}{self.activation_reprs[j]}"
            symbolic_str = sp.sympify(symbolic_str)
            formula[f"z_{i}"] = symbolic_str
        # for each Z_i replace y_i with the formula
        for i in range(self.linear.out_features // len(self.activations)):
            for j in range(self.linear.out_features):
                formula[f"z_{i}"] = formula[f"z_{i}"].subs(f"y_{j}", formula[f"y_{j}"])
        return {k: v for k, v in formula.items() if k[0] == "z"}

    def anneal_linear(self, anneal_rate=1e-1):
        """
        regularize the linear layer by adding noise to the weights
        """
        with torch.no_grad():
            norm = torch.mean(torch.abs(self.linear.weight.data))
            self.linear.weight.data = (
                self.linear.weight.data
                + torch.randn_like(self.linear.weight.data) * anneal_rate * norm
            )
        return None

    def anneal_regression(self, anneal_rate=1e-1):
        """
        regularize the regression layer by adding noise to the weights
        """
        with torch.no_grad():
            self.regression_layer.weight.data = (
                self.regression_layer.weight.data
                + torch.randn_like(self.regression_layer.weight.data) * anneal_rate
            )
            self.regression_layer.weight.data = (
                self.regression_layer.weight.data.abs()
                / torch.sum(
                    self.regression_layer.weight.data.abs(), dim=2, keepdim=True
                )
            )

        return None

    def anneal(self, anneal_rate=1e-1):
        """
        anneal both linear and regression layers
        """
        self.anneal_linear(anneal_rate)
        self.anneal_regression(anneal_rate)
        return None


class KharKAN(torch.nn.Module):
    def __init__(
        self,
        layers: Tuple[int] = None,
        activations: Tuple[torch.nn.Module] = None,
        activations_reprs: dict = None,
        normalize: bool = True,
    ):
        super(KharKAN, self).__init__()
        if layers is None:
            layers = (2, 8, 3, 1)
        if activations is None:
            activations = [torch.nn.Identity(), torch.square, lambda x: x * 0.0]
        if activations_reprs is None:
            activations_reprs = {0: "", 1: "^2", 2: "*0"}
        assert len(activations) == len(activations_reprs)
        self.layers = torch.nn.ModuleList(
            [
                DenseKanLayer(
                    layers[i], layers[i + 1], activations, activations_reprs, normalize
                )
                for i in range(len(layers) - 1)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def L1_loss(self):
        l1loss = 0
        for layer in self.layers:
            l1loss = l1loss + layer.L1_reg()
        return l1loss

    # self.layer1.L1_reg()+self.layer2.L1_reg()+self.layer3.L1_reg()
    def simplify(self):
        for layer in self.layers:
            layer.simplify()
        return None

    def set_normalize(self, normalize: bool = True):
        for layer in self.layers:
            layer.normalize = normalize
        return None

    def symbolic_formula(self):
        formula = {}
        self.subs_dict = {}
        for i, layer in enumerate(self.layers):
            formula.update({f"layer_{i}": layer.symbolic_formula()})
        # replace x with z from previous layer
        max_length = max(
            weight.shape[0] for layer in self.layers for weight in layer.linear.weight
        )
        self.x_to_y_dicts = {f"x_{i}": f"y_{i}" for i in range(max_length)}
        self.y_to_x_dicts = {f"y_{i}": f"x_{i}" for i in range(max_length)}
        for i in range(1, len(self.layers)):
            self.subs_dict[i] = {
                f"x_{j}": formula[f"layer_{i-1}"][f"z_{j}"]
                for j in range(len(formula[f"layer_{i-1}"]))
            }
            self.subs_dict[i] = {
                k: v.subs(self.x_to_y_dicts) for k, v in self.subs_dict[i].items()
            }
        finalformula = formula[f"layer_{len(self.layers)-1}"]
        for i in range(len(self.layers) - 1, 0, -1):
            for j in range(len(finalformula)):
                finalformula[f"z_{j}"] = (
                    finalformula[f"z_{j}"]
                    .subs(self.subs_dict[i])
                    .subs(self.y_to_x_dicts)
                )
        return finalformula

    def anneal(self, anneal_rate=1e-1):
        for layer in self.layers:
            layer.anneal(anneal_rate)
        return None