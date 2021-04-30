"""
Test model by load trained model
"""
import validate
import torch
import argparse
import models.densenet
import models.resnet
import models.inception
import dataloaders.datasetnormal


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    last_acc = 0
    best_acc = 0

    # ----------  Configuration  -------------
    NUM_FOLD = {'USC': 10, 'ESC': 5, 'GTZAN': 1}
    BATCH_SIZE = 1
    NUM_WORKERS = 0

    for i in range(1, NUM_FOLD['ESC']+1):
        val_loader = dataloaders.datasetnormal.fetch_dataloader("{}validation128mel{}.pkl".format('/media/disk4t/data/ESC/store/', i),
                                                                'ESC', BATCH_SIZE, NUM_WORKERS)

        model = models.resnet.ResNet('ESC', False).to(device)

        # best model for this fold
        checkpoint = torch.load("checkpoint/ESC/ResNet/model_best_{}.pth.tar".format(i))
        model.load_state_dict(checkpoint["model"])
        best_acc_fold = validate.evaluate(model, device, val_loader)
        best_acc += (best_acc_fold / NUM_FOLD['ESC'])

        # last model for this fold
        checkpoint = torch.load("checkpoint/ESC/ResNet/last{}.pth.tar".format(i))
        model.load_state_dict(checkpoint["model"])
        last_acc_fold = validate.evaluate(model, device, val_loader)
        last_acc += (last_acc_fold / NUM_FOLD['ESC'])

        print("Fold {} Best Acc:{} Last Acc:{}".format(i, best_acc_fold, last_acc_fold))

    print("Dataset:{} Model:{} Best Acc:{} Last Acc:{}".format('ESC', 'ResNet', best_acc, last_acc))

