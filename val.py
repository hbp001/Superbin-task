import torch
from utils import accuracy, AverageMeter
import wandb
import matplotlib.pyplot as plt

def val(model, dataloader, epoch=9999):
    #acc1_meter = AverageMeter(name='accuracy top 1')
    #acc5_meter = AverageMeter(name='accuracy top 5')
    n_iters = len(dataloader)
    model.eval()
    tp, fn, fp, tn = 0, 0, 0, 0
    with torch.no_grad():
        for iter_idx, (images, labels) in enumerate(dataloader):

            images = images.cuda()
            labels = labels.cuda()
            outputs = model(images)
            ouputs = outputs.sigmoid()
           # attention_map = model(images)
            #print(f"output : {outputs}")
                        
            max_score = outputs.max(dim = 1, keepdim=True)
            positive_thresh = 0.5
            predict = (outputs > positive_thresh).long() # type change
            labels = labels.type_as(predict) # type change (=predict)

            tp += (predict + labels).eq(2).sum(dim=0)
            fp += (predict + labels).eq(1).sum(dim=0)
            fn += (predict + labels).eq(-1).sum(dim=0)
            tn += (predict + labels).eq(0).sum(dim=0)

            acc = ((tp+tn)/(tp+tn+fp+fn)).sum()/4
            
           # plt.imshow(attention_map, aspect='auto', origin='lower', interpolation='none')
           # plt.savefig('attention_map.png', figsize=(16,4))

        print(f"val_acc = {acc}")

    wandb.log({
        "val_acc" : acc,
       # "Attention Map" : [wandb.Image('attention_map.png')]
    })

#            acc1, acc4 = accuracy(outputs, labels, topk=(1, 4))
#            acc1_meter.update(acc1[0], images.shape[0])
#            acc4_meter.update(acc5[0], images.shape[0])

#           print(f"[Epoch {epoch}] iter {iter_idx} / {n_iters}: \tAcc top-1 {acc1_meter.val:.2f}({acc1_meter.avg:.2f}) \tAcc top-4 {acc5_meter.val:.2f}({acc5_meter.avg:.2f})", end='\r')
#    print("")
#    print(f"Epoch {epoch} validation: top-1 acc {acc1_meter.avg} top-4 acc {acc5_meter.avg}")
    return acc
