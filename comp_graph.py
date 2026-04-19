import torch
from torchviz import make_dot

# 1. Create input tensors
# requires_grad=True is essential; without it, PyTorch doesn't build a graph
a = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([5.0], requires_grad=True)

# 2. Perform your requested operations
# Multiplication
c = a * b

# Division
d = c / a

# Square
y = d ** 2

# 3. Visualize the graph
# we pass the final output 'y' and a dictionary of named inputs
params = {"input_a": a, "input_b": b}
dot = make_dot(y, params=params)

# 4. Save and view
# This creates a file named 'pytorch_graph.pdf' (or .png if specified)
dot.render("pytorch_graph", format="pdf")

print("Graph saved as 'pytorch_graph.pdf'")