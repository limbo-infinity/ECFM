import torch


# Define a transform function that forces x to be between
# the upper and lower bounds of the biding prices
def transform_x(x, low, up):
    return low + (up - low) * torch.sigmoid(x)


def get_model_device(model):
    return next(model.parameters()).device
