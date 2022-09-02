# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.dataclass.configs import FairseqDataclass

import torch
import torch.nn as nn
from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("l1_loss", dataclass=FairseqDataclass)
class GraphPredictionL1Loss(FairseqCriterion):
    """
    Implementation for the L1 loss (MAE loss) used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):#modified
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """ 
        weight_iou=model.args.weight_iou
        sample_size = sample["nsamples"]
        with torch.no_grad():
            natoms = sample["net_input"]["batched_data"]["x"].shape[1]
        logits = model(**sample["net_input"])
        x=sample["net_input"]["batched_data"]['x']-1
        x.requires_grad = False
        
        #logits = logits[:, 0, :]
        logits = logits[:,1:].view(logits.size()[0],-1)#modification
        
        #targets = model.get_targets(sample, [logits])
        targets = model.get_targets(sample, [logits])[:,0:logits.size()[1]][: logits.size(0)]#modification
        targets.requires_grad = False
        logits = logits.view(logits.size()[0],-1,2)#extra modification
        targets = targets.view(targets.size()[0],-1,2)#extra modification
        
        
        nonzero=torch.tensor(targets!=0, dtype=int)#extra modification
        nonzero.requires_grad = False#extra modification
        logits=nonzero*logits#extra modification
        x=x[:,:,0:2]
        x=nonzero*x#extra modification
        
        loss_iou=0
        
        
        from torchvision.ops.boxes import _box_inter_union
        for i in range(logits.size()[0]):
            boxes=(torch.stack((logits[i,:,0]-x[i,:,0]/2,logits[i,:,1]-x[i,:,1]/2,logits[i,:,0]+x[i,:,0]/2,logits[i,:,1]+x[i,:,1]/2),1))
            inter, union=_box_inter_union(boxes,boxes)
            inter, union=inter[targets[i,:,0]!=0][:,targets[i,:,0]!=0], union[targets[i,:,0]!=0][:,targets[i,:,0]!=0]
            #loss_iou=loss_iou+(torch.mean(inter/union)-1/torch.sum(x[i,:,0]!=0))**0.5
            if torch.sum(union)!=0:
              #loss_iou=loss_iou+(torch.sum(inter)-torch.sum(torch.diagonal(inter)))/(torch.sum(union)-torch.sum(torch.diagonal(union)))
              loss_iou=loss_iou+torch.sum(inter)/torch.sum(union)
              #print('iou', (torch.sum(inter)-torch.sum(torch.diagonal(inter)))/(torch.sum(union)-torch.sum(torch.diagonal(union))))
        
        #print(targets)
        #print(logits)
        loss = nn.L1Loss(reduction="sum")(logits, targets)/logits.size()[1]/logits.size()[2]+weight_iou*loss_iou#modification
        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)
        #metrics.log_scalar("loss", loss_sum, sample_size, round=6)#modification
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("l1_loss_with_flag", dataclass=FairseqDataclass)
class GraphPredictionL1LossWithFlag(GraphPredictionL1Loss):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def perturb_forward(self, model, sample, perturb, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        batch_data = sample["net_input"]["batched_data"]["x"]
        with torch.no_grad():
            natoms = batch_data.shape[1]
        logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
        targets = model.get_targets(sample, [logits])
        loss = nn.L1Loss(reduction="sum")(logits, targets[: logits.size(0)])

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output
