import numpy as np
import torch
from sklearn.metrics import accuracy_score

def train(train_dataloader, my_model, optimizer, criterion_class, criterion_bbox, device):
    my_model.train()
    full_loss = 0
    for i, (images, targets) in enumerate(train_dataloader):
        images = torch.stack(images).to(device)
        targets = torch.stack(targets).to(device)
        labels = targets[:,-1:].long().squeeze(0)
        bboxes = tuple((targets[:,0:1], targets[:,1:2], targets[:,2:3], targets[:,3:4]))

        optimizer.zero_grad()
        outputs = my_model(images.to(device))
        label_preds = outputs[-1:][0]

        loss_cl = criterion_class(label_preds, labels.squeeze(1))

        loss_bbox = []
        for l in range(len(bboxes)):
            loss_bbox.append(criterion_bbox(outputs[:-1][l], bboxes[l]))

        loss = sum(loss for loss in loss_bbox) + loss_cl

        full_loss += loss
        loss.backward()
        optimizer.step()
    train_loss = full_loss/i         

    return train_loss


def validate(valid_loader, my_model, device):
    my_model.eval()

    cl_accuracy = 0.0
    batch_iou = 0.0
    mean_loss = 0.0

    with torch.no_grad():
        for i, (images, targets) in enumerate(valid_loader):
            images = torch.stack(images).to(device)
            targets = torch.stack(targets)
            labels = targets[:,-1:]
            bboxes = tuple((targets[:,0:1], targets[:,1:2], targets[:,2:3], targets[:,3:4]))
            bboxes = torch.stack(bboxes)
            _, h, _ = bboxes.shape

            bboxes = torch.reshape(bboxes, (-1,h)).T

            outputs = my_model(images)
            labels_preds = torch.argmax(outputs[-1:][0].to('cpu'), 1).int()
            bbox_pred = torch.stack(outputs[:-1]).to('cpu')
            bbox_pred = torch.reshape(bbox_pred, (-1,h)).T
            cl_accuracy += accuracy_score(labels_preds, labels.squeeze(1).numpy())

            iou = 0.0
            for k in range(len(bbox_pred)):
                iou += bb_IoU(bboxes[k], bbox_pred[k])
            batch_iou += iou / len(bbox_pred)

        mean_acc = cl_accuracy / len(valid_loader)
        mean_iou = batch_iou / len(valid_loader)

    return mean_acc, mean_iou

def bb_IoU(true_bbox, pred_bbox):
    '''
    IoU
    Функция рассчета отношения пересечения к объединению предсказанного бокса к истинному
    '''
    xA = max(true_bbox[0], pred_bbox[0])
    yA = max(true_bbox[1], pred_bbox[1])
    xB = min(true_bbox[2], pred_bbox[2])
    yB = min(true_bbox[3], pred_bbox[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (true_bbox[2] - true_bbox[0] + 1) * (true_bbox[3] - true_bbox[1] + 1)
    boxBArea = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)

def train_frcnn(train_dataloader, model, optimizer, device):
    model.train()
    running_loss = []
    for i, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        images, targets = data[0], data[1]
        new_targets = []
        for item in targets:
            temp_targets = {}            
            temp_targets["boxes"] = item[:-1].unsqueeze(0).to(device)
            temp_targets["labels"] = item[-1].unsqueeze(0)
            temp_targets["labels"] = torch.as_tensor(temp_targets["labels"], dtype=torch.int64).to(device)
            new_targets.append(temp_targets)

        images = list(image.to(device) for image in images)

        loss_dict = model(images, new_targets)

        loss = sum(loss for loss in loss_dict.values())
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    return np.mean(running_loss)

def validate_frcnn(valid_loader, model, device):
    model.eval()
    mean_acc = []
    mean_iou = []

    with torch.no_grad():
        for k, (images, targets) in enumerate(valid_loader):
            images = list(image for image in images)
            iou = []
            labels = []
            pred_labels = []

            for i, data_t in enumerate(targets):
                # в батче проходим по каждой картинке
                boxes = list(x for x in data_t[:-1].numpy())
                labels.append(int(data_t[-1]))

                predictions = model(images[i][None,...].to(device))
                preds = predictions[0]

                pred_boxes = list(x for x in preds['boxes'][:1].squeeze(0).to('cpu').numpy())
                # некоторые предсказание бывали пустыми, поэтому сделал такую заглушку
                if len(pred_boxes) == 0:
                    iou.append(0)
                    pred_labels.append(0)
                else:
                    iou.append(bb_IoU(boxes, pred_boxes))
                    pred_labels.append(int(preds['labels'][:1].to('cpu')))

            mean_acc.append(accuracy_score(pred_labels, labels))
            mean_iou.append(np.mean(iou))

    return round(np.mean(mean_acc), 5), round(np.mean(mean_iou), 5)