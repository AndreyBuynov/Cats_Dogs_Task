# Cat_Dog_Task

Тестовое задание на классификацию и детекцию кошек и собак

В [первой](https://github.com/AndreyBuynov/Cat_Dog_Task/blob/main/task.ipynb) работе использована простая сверточная сеть с 5-ю выходами: 1-й класс объекта, остальные 4 - координаты bounding box (xmin, ymin, xmax, ymax)
Так же был создан свой класс для подачи картинок в процессе обучения, где применяется аугментация для предотвращения переобучения сети из-за малого количества данных.

В первом варианте сеть обучалась на всем датасете разделенном на обучающую (2708 картинок) и валидационную (677 картинок) выборки.
Одна эпоха занимает в среднем около 28 секунд и после 20 эпох метрики получаются accuracy : 0.675, mIoU : 0.697

Для второго варианта я увеличил датасет в 5 раз, но сделал его сбалансированным. То есть брал случайным образом столько же собачек, сколько в датасете кошечек и объединял с кошечками. Так сделал 5 раз, после чего объединил все в новый датасет с реиндексацией. Повторяющиеся картинки не должны переобучать сеть из-за использования аугментации на стадии загрузки данных в сеть. Получился датасет из обучающей выборки размером 16592 картинки и валидационной выборки размером 4148 картинок.
На новых данных одна эпоха занимает в среднем 3 минуты 8 секунд.
После 20 эпох метрики accuracy : 0.7432, mIoU : 0.7644

Используемая выше модель обучается и через 20 эпох может делать более-менее правдивые предсказания. Улучшение метрик и процесс обучения происходит достаточно долго. В процессе тренировок модели были опробованы оптимизаторы Adam и SGD с разными параметрами, в первую очередь с вариациями learning rate. Как видно, изменение объема датасета так же влияет на скорость сходимости.

Во [второй](https://github.com/AndreyBuynov/Cat_Dog_Task/blob/main/task_2.ipynb) работе я использовал для решения задачи модель FasterRCNN_Resnet50. Даталоадер использовал тот-же, что и в первом варианте, только в в процессе обучения модели немного изменял поданные данные до нужного формата и убрал нормализацию координат боксов.

Модель не предобученная учится долго и размер батча пришлось уменьшить (память гугл коллаба переполнялась). Проход нодной эпохи занимает в среднем 5 минут 8 секунд.
После 10 эпох метрики accuracy : 0.8093, mIoU : 0.51273
Размер выборки: тренировочная 2708, валидационная 677 картинок

Тот же экспиремент, как в предыдущем случае, но на увеличенном датасете. Проход эпохи занимает в среднем чуть меньше 31 минуты что увеличивает обучение на 10 эпохах до 5 с лишним часов.
После 10 эпох метрики accuracy : 0.66394, mIoU : 0.45016
Размер выборки: тренировочная 16592, валидационная 4148 картинок

На проверочных картинках FasterRCNN показывает лучше результат чем моя сверточная сеть, однако по метрикам все получается наоборот. Я заметил, что по некторым картинкам из валидационного датасета FasterRCNN вообще не находит никакого искомого класса и в таких случаях к метрикам добавляются нули, которые сразу снижают средние показатели. Возможно, при большем количестве эпох на этих картинках FasterRCNN тоже определит искомый класс и тогда метрики сразу вырастут значительно.
