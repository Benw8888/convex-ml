import torch
import torch.nn as nn
import numpy as np

class WideModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim_scale=10, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim_scale * input_dim
        self.output_dim = output_dim
                
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        
        self.softmax = nn.Sigmoid()
        self.nll = nn.NLLLoss(reduction="mean")
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.Adict = {}
        self.Bdict = {}

        self.w0 = self.flatten_parameters()
        self.Atensor = torch.empty((0,len(self.w0)))
        self.Btensor = torch.empty((0,))
        
    def forward(self, x):
        x = x.view(x.size()[0],-1)
        return self.mlp(x)
    
    def linearized_forward(self, x, w):
        A = self.Adict[x]
        B = self.Bdict[x]
        return A@(w) + B
    
    def batched_linearized_forward(self, w):
        return torch.matmul(self.Atensor, w) + self.Btensor
    
    def dl(self, f, y):
        # shape 1xp
        return (torch.exp(f)/(1+torch.exp(f)) - y) @ self.Atensor
    
    def invddl(self, f):
        # shape pxp
        # f is the model logits. Equivalently, self.batched_linearized_forward(w)
        factor = torch.exp(f)
        factor = (1+factor)**2/factor
        factor = factor.nan_to_num(0)
        factor.clip(0,10)
        return torch.einsum("pd,d,qd->pq", self.invA, factor, self.invA)
    
    def newton_update(self, w, y):
        f = self.batched_linearized_forward(w)
        print(f"f is nan any: {f.isnan().any()}")
        dl = self.dl(f,y) # 1xp
        print(f"dl[0], {dl[0]}")
        print(f"dl is nan any: {dl.isnan().any()}")
        invddl = self.invddl(f) # pxp
        print(f"invddl[0], {invddl[0]}")
        print(f"invddl is nan any: {invddl.isnan().any()}")
        return w - dl @ invddl
    
    def update_stored_linear_tensors(self, A, B):
        self.Atensor = A
        self.Btensor = B
        self.invA = torch.linalg.pinv(self.Atensor)
    
    def loss(self, x, labels):
        logits = self.forward(x)
        return self.BCEWithLogitsLoss(logits, labels)
    
    def flatten_parameters(self):
        # Extract parameters from the model
        params = [param.view(-1) for param in self.parameters()]
        # Flatten and combine into a single vector
        flat_params = torch.cat(params)
        return flat_params
    
    def update_parameters(self, param_vector):
        # Pointer to keep track of the position in param_vector
        pointer = 0

        # Iterate over each parameter in the model
        for param in self.parameters():
            # Calculate the number of elements in this parameter
            num_param_elements = param.numel()

            # Extract the corresponding part of param_vector
            param_data = param_vector[pointer:pointer + num_param_elements]

            # Reshape the extracted data to match the shape of the parameter
            param_data = param_data.view(param.size())

            # Update the parameter data
            param.data = param_data

            # Update the pointer
            pointer += num_param_elements
        
        self.w0 = self.flatten_parameters()
    
    # def flatten_gradient(self, x):
    #     # Ensure the model is in training mode to enable gradient calculation
    #     self.train()

    #     # Forward pass to compute logits
    #     logits = self(x)

    #     # Number of logits
    #     num_logits = logits.size(-1)

    #     # Initialize a 2D tensor to store the gradients with respect to parameters
    #     gradients = torch.zeros((num_logits, sum(p.numel() for p in self.parameters())))

    #     # Compute the gradient for each logit
    #     for i in range(num_logits):
    #         # Zero out previous gradients
    #         self.zero_grad()

    #         # Create a zero vector with the same size as the logits
    #         grad_output = torch.zeros_like(logits)

    #         # Set the i-th element of grad_output to 1
    #         grad_output[0, i] = 1

    #         # Compute the gradient of the logits w.r.t. the parameters
    #         logits.backward(gradient=grad_output, retain_graph=True)

    #         # Extract and flatten gradients
    #         grads_flat = torch.cat([p.grad.view(-1) for p in self.parameters() if p.grad is not None])

    #         # Store the flattened gradients
    #         gradients[i] = grads_flat

    #     return gradients
    
    def flatten_gradient(self, x):
        # Ensure the model is in training mode to enable gradient calculation
        self.train()

        # Forward pass to compute logits
        logits = self(x)
        
        # Zero out previous gradients
        self.zero_grad()

        # Compute the gradient of the logits w.r.t. the parameters
        logits.backward()

        # Extract and flatten gradients
        grads_flat = torch.cat([p.grad.view(-1) for p in self.parameters() if p.grad is not None])

        return grads_flat