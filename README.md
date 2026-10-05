# Интеллектуальная система

## Тема
- Тема и проблема: (29 Energy: две цели)[https://archive.ics.uci.edu/dataset/242/energy+efficiency]
- Данные: (UCI Energy)[https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression_multioutput.html], 768×10
- ML: MultiOutput Ridge/GBR — (пример)[https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression_multioutput.html]	
- DL: MLP с двумя головками — (пример)[https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html]
- Синтетика: Два отклика — (make_regression)[https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html]


## Профиль модальности
- Профиль: P1 Табличный 
- Загрузчик и пример: Чанки CSV/Parquet; пример — типизированная строка
- Генератор синтетики: Гаусс/GMM/бутстрэп, CTGAN для категорий	
- Метрики дрейфа: PSI/KS по признакам, CSI по предсказаниям
- Специфика метрик: accuracy/PR/ROC, MAE/RMSE
