__author__ = "Hessam Bagherinezhad <hessam@xnor.ai>"

import torch
from torch.nn import functional as F
from torch.nn.modules import loss


class RefineryLoss(loss._Loss):
    """The KL-Divergence loss for the model and refined labels output.

    output must be a pair of (model_output, refined_labels), both NxC tensors.
    The rows of refined_labels must all add up to one (probability scores);
    however, model_output must be the pre-softmax output of the network."""

    def forward(self, output, target):
        if not self.training:
            # Loss is normal cross entropy loss between the model output and the
            # target.
            return F.cross_entropy(output, target,
                                   size_average=self.size_average)

        assert type(output) == tuple and len(output) == 2 and output[0].size() == \
            output[1].size(), "output must a pair of tensors of same size."

        # Target is ignored at training time. Loss is defined as KL divergence
        # between the model output and the refined labels.
        model_output, refined_labels = output
        if refined_labels.requires_grad:
            raise ValueError("Refined labels should not require gradients.")

        model_output_log_prob = F.log_softmax(model_output, dim=1)
        del model_output

        # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
        # for batch matrix multiplicatio
        refined_labels = refined_labels.unsqueeze(1)
        model_output_log_prob = model_output_log_prob.unsqueeze(2)

        # Compute the loss, and average/sum for the batch.
        cross_entropy_loss = -torch.bmm(refined_labels, model_output_log_prob)
        if self.size_average:
            cross_entropy_loss = cross_entropy_loss.mean()
        else:
            cross_entropy_loss = cross_entropy_loss.sum()
        # Return a pair of (loss_output, model_output). Model output will be
        # used for top-1 and top-5 evaluation.
        model_output_log_prob = model_output_log_prob.squeeze(2)
        return (cross_entropy_loss, model_output_log_prob)
