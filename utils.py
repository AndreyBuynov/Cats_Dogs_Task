import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import Resize


def plot_img(img_path, isboxes, model = None, device = torch.device('cuda')):
    """
    Функция возвращает картинку в виде массива с прорисованным bound box
    """

    numpy_img = cv2.imread(img_path) #[:,:,::-1]
    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = numpy_img.shape
    img = torch.from_numpy(numpy_img.astype('float32')).permute(2,0,1) / 255.
    font = cv2.FONT_HERSHEY_SIMPLEX

    labels = []
    boxes = []

    labels = [isboxes[0]]
    boxes.append([isboxes[1], isboxes[2], isboxes[3], isboxes[4]])

    if model is not None:
        sample = {}
        transform = A.Compose([
                  Resize(250, 250),
                  ToTensor()])
          
        model = model.eval()
        numpy_img_tr = transform(image = numpy_img)
        im_for_model = numpy_img_tr['image'].unsqueeze(0)
        predictions = model(im_for_model.to(device))
        label = torch.argmax(predictions[-1:][0].detach().cpu()).item()
        predictions = tuple(x.squeeze(0).detach().cpu().numpy() for x in predictions)
        predictions = [x[0] for x in predictions]
        box = []
        box.append([x * 250 for x in predictions])
        sample['image'] = im_for_model
        sample['bboxes'] = box

        labels = [label]
        boxes.append([int(sample['bboxes'][0][0] * (img_w/250)), 
                      int(sample['bboxes'][0][1] * (img_h/250)), 
                      int(sample['bboxes'][0][2] * (img_w/250)), 
                      int(sample['bboxes'][0][3] * (img_h/250))])

    for i, box in enumerate(boxes):
        numpy_img = cv2.rectangle(
            numpy_img, 
            (box[0],box[1]),
            (box[2],box[3]), 
            255,
            i*2
        )

    if len(boxes) > 0:
        return numpy_img, labels #.get()
    else:
        return numpy_img, labels
    
    
def show_random_pict(df, model = None, val_idx = None):
    """
    Функция выводит случайную картинку из датафрейма с обозначенным bound box
    """
    
    labels = {
        1 : 'cat',
        0 : 'dog'
    }
    if val_idx is None:
        random_animal = int(np.random.uniform(0,len(df)))
    else:
        random_animal = np.random.choice(val_idx)
    isboxes = tuple(df.loc[random_animal][['target', 'xmin', 'ymin', 'xmax', 'ymax']])
    img_with_boxes, label = plot_img(df.loc[random_animal]['img_name'], model = model, isboxes = isboxes)
    fig, ax = plt.subplots()
    ax.set_title(labels.get(label[0]))
    ax.imshow(img_with_boxes.astype('uint'));
    plt.show()