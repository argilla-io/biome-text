(window.webpackJsonp=window.webpackJsonp||[]).push([[51],{457:function(e,t,a){"use strict";a.r(t);var n=a(26),s=Object(n.a)({},(function(){var e=this,t=e.$createElement,a=e._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":e.$parent.slotKey}},[a("h1",{attrs:{id:"biome-text-trainer"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#biome-text-trainer"}},[e._v("#")]),e._v(" biome.text.trainer "),a("Badge",{attrs:{text:"Module"}})],1),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"create-dataloader"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#create-dataloader"}},[e._v("#")]),e._v(" create_dataloader "),a("Badge",{attrs:{text:"Function"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("create_dataloader")]),e._v(" ("),e._v("\n  instance_dataset: Union["),a("a",{attrs:{title:"biome.text.dataset.AllennlpDataset",href:"dataset.html#biome.text.dataset.AllennlpDataset"}},[e._v("AllennlpDataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpLazyDataset",href:"dataset.html#biome.text.dataset.AllennlpLazyDataset"}},[e._v("AllennlpLazyDataset")]),e._v("],\n  batch_size: int = 16,\n  num_workers: int = 0,\n)  -> torch.utils.data.dataloader.DataLoader\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Returns a pytorch DataLoader for AllenNLP instances")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("instance_dataset")])])]),e._v(" "),a("dd",[e._v("The dataset of instances for the DataLoader")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("batch_size")])])]),e._v(" "),a("dd",[e._v("Batch size")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("num_workers")])])]),e._v(" "),a("dd",[e._v("How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.\nDefault: 0")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("data_loader")])]),e._v(" "),a("dd",[e._v(" ")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"allennlp-collate"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#allennlp-collate"}},[e._v("#")]),e._v(" allennlp_collate "),a("Badge",{attrs:{text:"Function"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("allennlp_collate")]),e._v("("),a("span",[e._v("instances: List[allennlp.data.instance.Instance]) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]")]),e._v("\n")]),e._v("\n")])])]),e._v(" "),a("dd"),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"trainer"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#trainer"}},[e._v("#")]),e._v(" Trainer "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("Trainer")]),e._v(" ("),e._v("\n    "),a("span",[e._v("pipeline: Pipeline")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("train_dataset: Union["),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[e._v("Dataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpDataset",href:"dataset.html#biome.text.dataset.AllennlpDataset"}},[e._v("AllennlpDataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpLazyDataset",href:"dataset.html#biome.text.dataset.AllennlpLazyDataset"}},[e._v("AllennlpLazyDataset")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("valid_dataset: Union["),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[e._v("Dataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpDataset",href:"dataset.html#biome.text.dataset.AllennlpDataset"}},[e._v("AllennlpDataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpLazyDataset",href:"dataset.html#biome.text.dataset.AllennlpLazyDataset"}},[e._v("AllennlpLazyDataset")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("trainer_config: Union["),a("a",{attrs:{title:"biome.text.configuration.TrainerConfiguration",href:"configuration.html#biome.text.configuration.TrainerConfiguration"}},[e._v("TrainerConfiguration")]),e._v(", NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("vocab_config: Union[str, "),a("a",{attrs:{title:"biome.text.configuration.VocabularyConfiguration",href:"configuration.html#biome.text.configuration.VocabularyConfiguration"}},[e._v("VocabularyConfiguration")]),e._v(", NoneType] = 'default'")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("lazy: bool = False")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Class for training and testing a "),a("code",[e._v("biome.text.Pipeline")]),e._v(".")]),e._v(" "),a("p",[e._v("It is basically a light wrapper around the awesome Pytorch Lightning Trainer to define custom defaults and\nfacilitate the interaction with our pipelines.")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("pipeline")])])]),e._v(" "),a("dd",[e._v("Pipeline to train")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("train_dataset")])])]),e._v(" "),a("dd",[e._v("The training dataset. Default: "),a("code",[e._v("None")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("valid_dataset")])])]),e._v(" "),a("dd",[e._v("The validation dataset. Default: "),a("code",[e._v("None")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("trainer_config")])])]),e._v(" "),a("dd",[e._v("The configuration of the trainer. Default: "),a("code",[e._v("TrainerConfiguration()")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("vocab_config")])])]),e._v(" "),a("dd",[e._v("A "),a("code",[e._v("VocabularyConfiguration")]),e._v(" to create/extend the pipeline's vocabulary.\nIf "),a("code",[e._v('"default"')]),e._v(" (str), we will use the default configuration "),a("code",[e._v("VocabularyConfiguration()")]),e._v(".\nIf None, we will leave the pipeline's vocabulary untouched. Default: "),a("code",[e._v('"default"')]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("lazy")])])]),e._v(" "),a("dd",[e._v("If True, instances are lazily loaded from disk, otherwise they are loaded into memory.\nIgnored when passing in "),a("code",[e._v("InstanceDataset")]),e._v("s. Default: False.")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"fit"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#fit"}},[e._v("#")]),e._v(" fit "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("fit")]),e._v(" ("),e._v("\n  self,\n  output_dir: Union[pathlib.Path, str, NoneType] = 'output',\n  exist_ok: bool = False,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Train the pipeline")]),e._v(" "),a("p",[e._v("At the end of the training the pipeline will load the weights from the best checkpoint.")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("output_dir")])])]),e._v(" "),a("dd",[e._v("If specified, save the trained pipeline to this directory. Default: 'output'.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("exist_ok")])])]),e._v(" "),a("dd",[e._v("If True, overwrite the content of "),a("code",[e._v("output_dir")]),e._v(". Default: False.")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"test"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#test"}},[e._v("#")]),e._v(" test "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("test")]),e._v(" ("),e._v("\n  self,\n  test_dataset: Union["),a("a",{attrs:{title:"biome.text.dataset.Dataset",href:"dataset.html#biome.text.dataset.Dataset"}},[e._v("Dataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpDataset",href:"dataset.html#biome.text.dataset.AllennlpDataset"}},[e._v("AllennlpDataset")]),e._v(", "),a("a",{attrs:{title:"biome.text.dataset.AllennlpLazyDataset",href:"dataset.html#biome.text.dataset.AllennlpLazyDataset"}},[e._v("AllennlpLazyDataset")]),e._v("],\n  batch_size: Union[int, NoneType] = None,\n  output_dir: Union[pathlib.Path, str, NoneType] = None,\n  verbose: bool = True,\n)  -> Dict[str, Any]\n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("Evaluate your model on a test dataset")]),e._v(" "),a("h2",{attrs:{id:"parameters"}},[e._v("Parameters")]),e._v(" "),a("dl",[a("dt",[a("strong",[a("code",[e._v("test_dataset")])])]),e._v(" "),a("dd",[e._v("The test data set.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("batch_size")])])]),e._v(" "),a("dd",[e._v("The batch size. If None (default), we will use the batch size specified in the "),a("code",[e._v("TrainerConfiguration")]),e._v(".")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("output_dir")])])]),e._v(" "),a("dd",[e._v("Save a "),a("code",[e._v("metrics.json")]),e._v(" to this output directory. Default: None.")]),e._v(" "),a("dt",[a("strong",[a("code",[e._v("verbose")])])]),e._v(" "),a("dd",[e._v("If True, prints the test results. Default: True.")])]),e._v(" "),a("h2",{attrs:{id:"returns"}},[e._v("Returns")]),e._v(" "),a("dl",[a("dt",[a("code",[e._v("Dict[str, Any]")])]),e._v(" "),a("dd",[e._v("A dictionary with the metrics")])])])]),e._v(" "),a("div"),e._v(" "),a("pre",{staticClass:"title"},[a("h2",{attrs:{id:"modelcheckpointwithvocab"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#modelcheckpointwithvocab"}},[e._v("#")]),e._v(" ModelCheckpointWithVocab "),a("Badge",{attrs:{text:"Class"}})],1),e._v("\n")]),e._v(" "),a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("class")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("ModelCheckpointWithVocab")]),e._v(" ("),e._v("\n    "),a("span",[e._v("dirpath: Union[str, pathlib.Path, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("filename: Union[str, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("monitor: Union[str, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("verbose: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("save_last: Union[bool, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("save_top_k: Union[int, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("save_weights_only: bool = False")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("mode: str = 'min'")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("auto_insert_metric_name: bool = True")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("every_n_train_steps: Union[int, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("every_n_val_epochs: Union[int, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n    "),a("span",[e._v("period: Union[int, NoneType] = None")]),a("span",[e._v(",")]),e._v("\n"),a("span",[e._v(")")]),e._v("\n")]),e._v("\n")]),e._v(" "),a("p",[e._v("Save the model periodically by monitoring a quantity. Every metric logged with\n:meth:"),a("code",[e._v("~pytorch_lightning.core.lightning.log")]),e._v(" or :meth:"),a("code",[e._v("~pytorch_lightning.core.lightning.log_dict")]),e._v(" in\nLightningModule is a candidate for the monitor key. For more information, see\n:ref:"),a("code",[e._v("weights_loading")]),e._v(".")]),e._v(" "),a("p",[e._v("After training finishes, use :attr:"),a("code",[e._v("best_model_path")]),e._v(" to retrieve the path to the\nbest checkpoint file and :attr:"),a("code",[e._v("best_model_score")]),e._v(" to retrieve its score.")]),e._v(" "),a("p",[e._v("Args:\ndirpath: directory to save the model file.")]),e._v(" "),a("pre",[a("code",[e._v("    Example::\n"),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[e._v("    # custom path\n    # saves a file like: my/path/epoch=0-step=10.ckpt\n    &gt;&gt;&gt; checkpoint_callback = ModelCheckpoint(dirpath='my/path/')\n\nBy default, dirpath is &lt;code&gt;None&lt;/code&gt; and will be set at runtime to the location\nspecified by :class:`~pytorch_lightning.trainer.trainer.Trainer`'s\n:paramref:`~pytorch_lightning.trainer.trainer.Trainer.default_root_dir` or\n:paramref:`~pytorch_lightning.trainer.trainer.Trainer.weights_save_path` arguments,\nand if the Trainer uses a logger, the path will also contain logger name and version.\n")])])]),a("p",[e._v("filename: checkpoint filename. Can contain named formatting options to be auto-filled.")]),e._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[e._v("Example::\n\n    # save any arbitrary metrics like &lt;code&gt;val\\_loss&lt;/code&gt;, etc. in name\n    # saves a file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt\n    &gt;&gt;&gt; checkpoint_callback = ModelCheckpoint(\n    ...     dirpath='my/path',\n    ...     filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'\n    ... )\n\nBy default, filename is &lt;code&gt;None&lt;/code&gt; and will be set to ``'{epoch}-{step}'``.\n")])])]),a("p",[e._v("monitor: quantity to monitor. By default it is <code>None</code> which saves a checkpoint only for the last epoch.\nverbose: verbosity mode. Default: <code>False</code>.\nsave_last: When <code>True</code>, always saves the model at the end of the epoch to\na file <code>last.ckpt</code>. Default: <code>None</code>.\nsave_top_k: if "),a("code",[e._v("save_top_k == k")]),e._v(",\nthe best k models according to\nthe quantity monitored will be saved.\nif "),a("code",[e._v("save_top_k == 0")]),e._v(", no models are saved.\nif "),a("code",[e._v("save_top_k == -1")]),e._v(", all models are saved.\nPlease note that the monitors are checked every <code>period</code> epochs.\nif "),a("code",[e._v("save_top_k &gt;= 2")]),e._v(" and the callback is called multiple\ntimes inside an epoch, the name of the saved file will be\nappended with a version count starting with <code>v1</code>.\nmode: one of {min, max}.\nIf "),a("code",[e._v("save_top_k != 0")]),e._v(", the decision to overwrite the current save file is made\nbased on either the maximization or the minimization of the monitored quantity.\nFor "),a("code",[e._v("'val_acc'")]),e._v(", this should be "),a("code",[e._v("'max'")]),e._v(", for "),a("code",[e._v("'val_loss'")]),e._v(" this should be "),a("code",[e._v("'min'")]),e._v(", etc.\nsave_weights_only: if <code>True</code>, then only the model's weights will be\nsaved (<code>model.save_weights(filepath)</code>), else the full model\nis saved (<code>model.save(filepath)</code>).\nevery_n_train_steps: Number of training steps between checkpoints.\nIf "),a("code",[e._v("every_n_train_steps == None or every_n_train_steps == 0")]),e._v(", we skip saving during training\nTo disable, set "),a("code",[e._v("every_n_train_steps = 0")]),e._v(". This value must be <code>None</code> non-negative.\nThis must be mutually exclusive with <code>every_n_val_epochs</code>.\nevery_n_val_epochs: Number of validation epochs between checkpoints.\nIf "),a("code",[e._v("every_n_val_epochs == None or every_n_val_epochs == 0")]),e._v(", we skip saving on validation end\nTo disable, set "),a("code",[e._v("every_n_val_epochs = 0")]),e._v(". This value must be <code>None</code> or non-negative.\nThis must be mutually exclusive with <code>every_n_train_steps</code>.\nSetting both "),a("code",[e._v("ModelCheckpoint(..., every_n_val_epochs=V)")]),e._v(" and\n"),a("code",[e._v("Trainer(max_epochs=N, check_val_every_n_epoch=M)")]),e._v("\nwill only save checkpoints at epochs 0 < E <= N\nwhere both values for <code>every_n_val_epochs</code> and <code>check_val_every_n_epoch</code> evenly divide E.\nperiod: Interval (number of epochs) between checkpoints.")]),e._v(" "),a("div",{staticClass:"language- extra-class"},[a("pre",[a("code",[e._v('!!! warning "Warning"\n    This argument has been deprecated in v1.3 and will be removed in v1.5.\n\nUse &lt;code&gt;every\\_n\\_val\\_epochs&lt;/code&gt; instead.\n')])])]),a("p")])]),e._v(" "),a("p",[e._v("Note:\nFor extra customization, ModelCheckpoint includes the following attributes:")]),e._v(" "),a("pre",[a("code",[e._v("- "),a("code",[e._v('CHECKPOINT_JOIN_CHAR = "-"')]),a("p"),e._v("\n"),a("ul",[e._v("\n"),a("li",[a("code",[e._v('CHECKPOINT_NAME_LAST = "last"')])]),e._v("\n"),a("li",[a("code",[e._v('FILE_EXTENSION = ".ckpt"')])]),e._v("\n"),a("li",[a("code",[e._v("STARTING_VERSION = 1")])]),e._v("\n")]),e._v("\n"),a("p",[e._v("For example, you can change the default last checkpoint name by doing\n"),a("code",[e._v('checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"')]),e._v("\n")])])]),e._v(" "),a("p",[e._v("Raises:\nMisconfigurationException:\nIf "),a("code",[e._v("save_top_k")]),e._v(" is neither "),a("code",[e._v("None")]),e._v(" nor more than or equal to "),a("code",[e._v("-1")]),e._v(",\nif "),a("code",[e._v("monitor")]),e._v(" is "),a("code",[e._v("None")]),e._v(" and "),a("code",[e._v("save_top_k")]),e._v(" is none of "),a("code",[e._v("None")]),e._v(", "),a("code",[e._v("-1")]),e._v(", and "),a("code",[e._v("0")]),e._v(", or\nif "),a("code",[e._v("mode")]),e._v(" is none of "),a("code",[e._v('"min"')]),e._v(" or "),a("code",[e._v('"max"')]),e._v(".\nValueError:\nIf "),a("code",[e._v("trainer.save_checkpoint")]),e._v(" is "),a("code",[e._v("None")]),e._v(".")]),e._v(" "),a("p",[e._v("Example::")]),e._v(" "),a("pre",[a("code",[e._v(">>> from pytorch_lightning import Trainer\n>>> from pytorch_lightning.callbacks import ModelCheckpoint"),a("p"),e._v("\n"),a("h1",{attrs:{id:"saves-checkpoints-to-my-path-at-every-epoch"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#saves-checkpoints-to-my-path-at-every-epoch"}},[e._v("#")]),e._v(" saves checkpoints to 'my/path/' at every epoch")]),e._v("\n"),a("p",[e._v(">>> checkpoint_callback = ModelCheckpoint(dirpath='my/path/')\n>>> trainer = Trainer(callbacks=[checkpoint_callback])")]),e._v("\n"),a("h1",{attrs:{id:"save-epoch-and-val-loss-in-name"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#save-epoch-and-val-loss-in-name"}},[e._v("#")]),e._v(" save epoch and val_loss in name")]),e._v("\n"),a("h1",{attrs:{id:"saves-a-file-like-my-path-sample-mnist-epoch-02-val-loss-0-32-ckpt"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#saves-a-file-like-my-path-sample-mnist-epoch-02-val-loss-0-32-ckpt"}},[e._v("#")]),e._v(" saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt")]),e._v("\n"),a("p",[e._v(">>> checkpoint_callback = ModelCheckpoint(\n...     monitor='val_loss',\n...     dirpath='my/path/',\n...     filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'\n... )")]),e._v("\n"),a("h1",{attrs:{id:"save-epoch-and-val-loss-in-name-but-specify-the-formatting-yourself-e-g-to-avoid-problems-with-tensorboard"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#save-epoch-and-val-loss-in-name-but-specify-the-formatting-yourself-e-g-to-avoid-problems-with-tensorboard"}},[e._v("#")]),e._v(" save epoch and val_loss in name, but specify the formatting yourself (e.g. to avoid problems with Tensorboard")]),e._v("\n"),a("h1",{attrs:{id:"or-neptune-due-to-the-presence-of-characters-like-or"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#or-neptune-due-to-the-presence-of-characters-like-or"}},[e._v("#")]),e._v(" or Neptune, due to the presence of characters like '=' or '/')")]),e._v("\n"),a("h1",{attrs:{id:"saves-a-file-like-my-path-sample-mnist-epoch02-val-loss0-32-ckpt"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#saves-a-file-like-my-path-sample-mnist-epoch02-val-loss0-32-ckpt"}},[e._v("#")]),e._v(" saves a file like: my/path/sample-mnist-epoch02-val_loss0.32.ckpt")]),e._v("\n"),a("p",[e._v(">>> checkpoint_callback = ModelCheckpoint(\n...     monitor='val/loss',\n...     dirpath='my/path/',\n...     filename='sample-mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}',\n...     auto_insert_metric_name=False\n... )")]),e._v("\n"),a("h1",{attrs:{id:"retrieve-the-best-checkpoint-after-training"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#retrieve-the-best-checkpoint-after-training"}},[e._v("#")]),e._v(" retrieve the best checkpoint after training")]),e._v("\n"),a("p",[e._v("checkpoint_callback = ModelCheckpoint(dirpath='my/path/')\ntrainer = Trainer(callbacks=[checkpoint_callback])\nmodel = ...\ntrainer.fit(model)\ncheckpoint_callback.best_model_path\n")])])]),e._v(" "),a("pre",{staticClass:"title"},[a("p"),e._v("\n"),a("h3",{attrs:{id:"ancestors"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#ancestors"}},[e._v("#")]),e._v(" Ancestors")]),e._v("\n")]),e._v(" "),a("ul",{staticClass:"hlist"},[a("li",[e._v("pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint")]),e._v(" "),a("li",[e._v("pytorch_lightning.callbacks.base.Callback")]),e._v(" "),a("li",[e._v("abc.ABC")])]),e._v(" "),a("dl",[a("pre",{staticClass:"title"},[a("h3",{attrs:{id:"on-pretrain-routine-start"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#on-pretrain-routine-start"}},[e._v("#")]),e._v(" on_pretrain_routine_start "),a("Badge",{attrs:{text:"Method"}})],1),e._v("\n")]),e._v(" "),a("dt",[a("div",{staticClass:"language-python extra-class"},[a("pre",{staticClass:"language-python"},[a("code",[e._v("\n"),a("span",{staticClass:"token keyword"},[e._v("def")]),e._v(" "),a("span",{staticClass:"ident"},[e._v("on_pretrain_routine_start")]),e._v(" ("),e._v("\n  self,\n  trainer,\n  pl_module: PipelineModel,\n) \n")]),e._v("\n")])])]),e._v(" "),a("dd",[a("p",[e._v("When pretrain routine starts we build the ckpt dir on the fly")])])])])}),[],!1,null,null,null);t.default=s.exports}}]);