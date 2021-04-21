import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


def plot_img(img_path, model = None, isboxes = None):
    """
    Функция возвращает картинку в виде массива с прорисованным bound box
    """

    numpy_img = cv2.imread(img_path) #[:,:,::-1]
    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(numpy_img.astype('float32')).permute(2,0,1) / 255.
    font = cv2.FONT_HERSHEY_SIMPLEX

    if isboxes is not None:
        labels = []
        boxes = []
        labels.append(isboxes[0]) 
        boxes.append([isboxes[1], isboxes[2], isboxes[3], isboxes[4]])

    else:
        model = model.eval()
        predictions = model(img[None,...].to(device))
        preds = predictions[0]
        #print(preds)

        CONF_THRESH = 0.7
        boxes = preds['boxes'][preds['scores'] > CONF_THRESH]
        boxes_dict = {}
        boxes_dict['boxes'] = boxes

        labels = preds['labels'].cpu().detach().numpy()
        scores = preds['scores'].cpu().detach().numpy()

    for i, box in enumerate(boxes):
        numpy_img = cv2.rectangle(
            numpy_img, 
            (box[0],box[1]),
            (box[2],box[3]), 
            255,
            1
        )

    if len(boxes) > 0:
        return numpy_img #.get()
    else:
        return numpy_img
    
    
def show_random_pict(df, model = None):
    """
    Функция выводит случайную картинку из датафрейма с обозначенным bound box
    """
    
    labels = {
        1 : 'cat',
        0 : 'dog'
    }
    
    random_animal = int(np.random.uniform(0,len(df)))
    isboxes = tuple(df.loc[random_animal][['target', 'xmin', 'ymin', 'xmax', 'ymax']])
    img_with_boxes = plot_img(df.loc[random_animal]['img_name'], model = model, isboxes = isboxes)
    fig, ax = plt.subplots()
    ax.set_title(labels.get(isboxes[0]))
    ax.imshow(img_with_boxes.astype('uint'));
    plt.show()