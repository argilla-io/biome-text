# biome.text.models.mixins <Badge text="Module"/>
<dl>
<h2 id="biome.text.models.mixins.BiomeClassifierMixin">BiomeClassifierMixin <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">BiomeClassifierMixin</span> (</span>
    <span>vocab</span><span>,</span>
    <span>accuracy: Union[allennlp.training.metrics.categorical_accuracy.CategoricalAccuracy, NoneType] = None</span><span>,</span>
    <span>**kwargs</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>A mixin class for biome classifiers.</p>
<p>Inheriting from this class allows you to use Biome's awesome UIs.
It standardizes the <code>decode</code> and <code>get_metrics</code> methods.
Some stuff to be aware of:
- make sure your forward's output_dict has a "class_probability" key
- use the <code>_biome_classifier_metrics</code> dict in the forward method to record the metrics
- the forward signature must be compatible with the text_to_instance method of your DataReader
- the <code>decode</code> and <code>get_metrics</code> methods override the allennlp.models.model.Model methods</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong></dt>
<dd>Used to initiate the F1 measures for each label. It is also passed on to the model.</dd>
<dt><strong><code>accuracy</code></strong></dt>
<dd>The accuracy you want to use. By default, we choose a categorical top-1 accuracy.</dd>
<dt><strong><code>kwargs</code></strong></dt>
<dd>Passed on to the model class init</dd>
</dl>
<h2 id="examples">Examples</h2>
<p>An example of how to implement an AllenNLP model in biome-text to be able to use Biome's UIs:</p>
<pre><code class="python">&gt;&gt;&gt; from allennlp.models.bert_for_classification import BertForClassification
&gt;&gt;&gt;
&gt;&gt;&gt; @Model.register(&quot;biome_bert_classifier&quot;)
&gt;&gt;&gt; class BiomeBertClassifier(BiomeClassifierMixin, BertForClassification):
&gt;&gt;&gt;     def __init__(self, vocab, bert_model, num_labels, index, label_namespace,
&gt;&gt;&gt;                  trainable, initializer, regularizer, accuracy):
&gt;&gt;&gt;         super().__init__(accuracy=accuracy, vocab=vocab, bert_model=bert_model, num_labels=num_labels,
&gt;&gt;&gt;                          index=index, label_namespace=label_namespace, trainable=trainable,
&gt;&gt;&gt;                          initializer=initializer, regularizer=regularizer)
&gt;&gt;&gt;
&gt;&gt;&gt;     @overrides
&gt;&gt;&gt;     def forward(self, tokens, label = None):
&gt;&gt;&gt;         output_dict = super().forward(tokens=tokens, label=label)
&gt;&gt;&gt;         output_dict[&quot;class_probabilities&quot;] = output_dict.pop(&quot;probs&quot;)
&gt;&gt;&gt;         if label is not None:
&gt;&gt;&gt;             for metric in self._biome_classifier_metrics.values():
&gt;&gt;&gt;                 metric(logits, label)
&gt;&gt;&gt;         return output_dict
</code></pre></div>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.models.biome_bimpm.BiomeBiMpm" href="biome_bimpm.html#biome.text.models.biome_bimpm.BiomeBiMpm">BiomeBiMpm</a></li>
<li><a title="biome.text.models.multifield_bimpm.MultifieldBiMpm" href="multifield_bimpm.html#biome.text.models.multifield_bimpm.MultifieldBiMpm">MultifieldBiMpm</a></li>
<li><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase">SequenceClassifierBase</a></li>
</ul>
<dl>
<h3 id="biome.text.models.mixins.BiomeClassifierMixin.decode">decode <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">decode</span> (</span>
   self,
   output_dict: Dict[str, torch.Tensor],
)  -> Dict[str, torch.Tensor]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Does a simple position-wise argmax over each token, converts indices to string labels, and
adds a <code>"tags"</code> key to the dictionary with the result.</p></div>
</dd>
<h3 id="biome.text.models.mixins.BiomeClassifierMixin.get_metrics">get_metrics <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_metrics</span> (</span>
   self,
   reset: bool = False,
)  -> Dict[str, float]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Get the metrics of our classifier, see :func:<code>~allennlp_2.models.Model.get_metrics</code>.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>reset</code></strong></dt>
<dd>Reset the metrics after obtaining them?</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>A dictionary with all metric names and values.</p></div>
</dd>
</dl>
</dd>
</dl>