# AutoIntent

![](assets/classification_pipeline.png)

Планы:
- прогнать CrossEncoderWithLogreg на всех датасетах
- реализовать подбор порога как в статье DNNC
- реализовать метрики для RegExpModule
- подумать над кешированием запросов к collection (ибо на оптимизации k для knn и dncc можно переиспользовать много запросов)
- идея для метрики для RegExp:
    - аккураси и покрытие (эти метрики помогут понять, нужно ли вообще использовать модуль RegExp)
- подумать над проблемой переобучения: следующие этапы оптимизации должны использовать другие данные нежели предыдущие
- много разных TODO в коде

backlog:
- optuna
- medium results caching
- logging
- dataset expansion (LLM, xeger etc)