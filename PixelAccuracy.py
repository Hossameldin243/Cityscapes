import torch

def pixel_accuracy(y_true, y_pred):
    y_pred_class = torch.argmax(y_pred, dim=1)
    correct_pixels = torch.sum(y_true == y_pred_class)
    total_pixels = torch.numel(y_true)
    pixel_acc = correct_pixels.float() / total_pixels
    return pixel_acc.item()