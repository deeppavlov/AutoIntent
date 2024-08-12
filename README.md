# AutoIntent

![](assets/classification_pipeline.png)

Планы:
- RegExp модуль:
    - сгенерить регулярки с помощью ллм для имеющихся датасетов
    - возможно применить какой-нибудь статистический анализ (биграммы триграммы) и самому составить
    - реализовать метрики
        - аккураси и покрытие (эти метрики помогут понять, нужно ли вообще использовать модуль RegExp)
- OOS detection
    - прогнать CrossEncoderWithLogreg на всех датасетах
    - пока что реализован простейший сеттинг: в разметке есть отдельный класс (-1), содержащий все OOS примеры
    - потом подумать над другими сценариями: тулза для генерации OOS семплов
    - посмотреть датасеты где много доменов, и отдельные домены взять как OOS (hwu64, multiwoz, sgd)

backlog:
- multilabel classification
- logging
- добавить скрипт для извлечения лучшего пайплайна из логов оптимизации
- подумать над кешированием запросов к collection (ибо на оптимизации k для knn и dncc можно переиспользовать много запросов)
- подумать над проблемой переобучения: следующие этапы оптимизации должны использовать другие данные нежели предыдущие
- много разных TODO в коде
- medium results caching to file
- dataset expansion (LLM, xeger etc)
- optuna