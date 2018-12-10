# Issues and ideas

* ```learn``` command: do we need to maintain a copy/paste of the code? or is the same as running allennlp train with --include-package biome?

* CSV Datasets with no header, how they can be mapped in the data reader pipeline? "col_1"?

* Some datasets refer to classes as numbers and provide a classes.txt file for mapping them to labels, what is the best way to map them to labels?

* How to integrate ``dry-run`` command?

* Do we need ``make-vocab`` command?

* Which is the best way to handle full dataset shuffle pre-process step?

* Ìs this useful ``allennlp/data/dataset_readers/multiprocess_dataset_reader.py``, it assumes file_path as a glob so dataset needs to be splitted beforehand (not so cool) ?

* ``"lowercase_tokens": false`` is a property of tokenizer in allennlp, I think it makes more sense to have this more extensible, to preprocess text with pipelines suchas html_strip --> lowercase --> digits --> emails .. Something like ``tf.transform``