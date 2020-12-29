=========
Changelog
=========

Version 2.0
===========

- Replaced `DataSource` with `Dataset`
- Vocab creation is now automatically done when executing `Pipeline.train()`
- Introduced `TuneExperiment` class
- Added the *transformers* feature
- Move `Pipeline.explore()` command to its own module
- `Pipeline.train()` modifies the pipeline inplace instead of creating a copy for the training
- `TokenClassification` accepts entities
- Added a `RelationClassification` head
- A LOT if minor and not so minor changes ...

Version 1.0
===========

- Introduce the *pipeline, backbone, head* concept
