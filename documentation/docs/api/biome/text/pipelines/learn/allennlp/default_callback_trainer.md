# biome.text.pipelines.learn.allennlp.default_callback_trainer <Badge text="Module"/>
<p>This module includes the default biome callback trainer and some extra functions/classes for this purpose</p>
<dl>
<h2 id="biome.text.pipelines.learn.allennlp.default_callback_trainer.DefaultCallbackTrainer">DefaultCallbackTrainer <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">DefaultCallbackTrainer</span> (</span>
    <span>model: allennlp.models.model.Model</span><span>,</span>
    <span>training_data: Iterable[allennlp.data.instance.Instance]</span><span>,</span>
    <span>iterator: allennlp.data.iterators.data_iterator.DataIterator</span><span>,</span>
    <span>optimizer: torch.optim.optimizer.Optimizer</span><span>,</span>
    <span>num_epochs: int = 20</span><span>,</span>
    <span>shuffle: bool = True</span><span>,</span>
    <span>serialization_dir: Union[str, NoneType] = None</span><span>,</span>
    <span>cuda_device: Union[int, List] = -1</span><span>,</span>
    <span>callbacks: List[allennlp.training.callbacks.callback.Callback] = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>An callback trainer with some extra callbacks already configured</p>
<p>A trainer for doing supervised learning. It just takes a labeled dataset
and a <code>DataIterator</code>, and uses the supplied <code>Optimizer</code> to learn the weights
for your model over some fixed number of epochs. It uses callbacks to handle various
things ancillary to training, like tracking metrics, validation, early stopping,
logging to tensorboard, and so on.</p>
<p>It's easy to create your own callbacks; for example, if you wanted to get a Slack
notification when training finishes. For more complicated variations, you might have
to create your own subclass, in which case make sure to fire off all the training events.</p>
<h2 id="parameters">Parameters</h2>
<p>model : <code>Model</code>, required.
An AllenNLP model to be optimized. Pytorch Modules can also be optimized if
their <code>forward</code> method returns a dictionary with a "loss" key, containing a
scalar tensor representing the loss function to be optimized.</p>
<pre><code>If you are training your model using GPUs, your model should already be
on the correct device. (If you use &lt;code&gt;Trainer.from\_params&lt;/code&gt; this will be
handled for you.)
</code></pre>
<dl>
<dt><strong><code>training_data</code></strong> :&ensp;<code><code>Iterable\[Instance]&lt;/code&gt;, required</code></dt>
<dd>The instances that you want to train your model on.</dd>
<dt><strong><code>iterator</code></strong> :&ensp;<code><code>DataIterator&lt;/code&gt;, required</code></dt>
<dd>The iterator for batching / epoch-ing the instances.</dd>
<dt>optimizer : <code>torch.nn.Optimizer</code>, required.</dt>
<dt>An instance of a Pytorch Optimizer, instantiated with the parameters of the</dt>
<dt>model to be optimized.</dt>
<dt><strong><code>num_epochs</code></strong> :&ensp;<code>int</code>, optional <code>(default=20)</code></dt>
<dd>Number of training epochs.</dd>
<dt><strong><code>shuffle</code></strong> :&ensp;<code>bool</code>, optional <code>(default=True)</code></dt>
<dd>Whether to shuffle the instances each epoch.</dd>
<dt><strong><code>serialization_dir</code></strong> :&ensp;<code>str</code>, optional <code>(default=None)</code></dt>
<dd>Path to directory for saving and loading model files. Models will not be saved if
this parameter is not passed.</dd>
<dt><strong><code>cuda_device</code></strong> :&ensp;<code>Union[int, List[int]]</code>, optional <code>(default=-1)</code></dt>
<dd>An integer or list of integers specifying the CUDA device(s) to use. If -1, the CPU is used.</dd>
<dt><strong><code>callbacks</code></strong> :&ensp;<code>List[Callback]</code>, optional <code>(default=None)</code></dt>
<dd>A list of callbacks that will be called based on training events.</dd>
</dl></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.training.callback_trainer.CallbackTrainer</li>
<li>allennlp.training.trainer_base.TrainerBase</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.pipelines.learn.allennlp.default_callback_trainer.DefaultCallbackTrainer.from_params">from_params <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_params</span> (</span>
   params: allennlp.common.params.Params,
   serialization_dir: str,
   recover: bool = False,
   cache_directory: str = None,
   cache_prefix: str = None,
)  -> allennlp.training.callback_trainer.CallbackTrainer
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>This is the automatic implementation of <code>from_params</code>. Any class that subclasses <code>FromParams</code>
(or <code>Registrable</code>, which itself subclasses <code>FromParams</code>) gets this implementation for free.
If you want your class to be instantiated from params in the "obvious" way &ndash; pop off parameters
and hand them to your constructor with the same names &ndash; this provides that functionality.</p>
<p>If you need more complex logic in your from <code>from_params</code> method, you'll have to implement
your own method that overrides this one.</p></div>
</dd>
</dl>
</dd>
</dl>