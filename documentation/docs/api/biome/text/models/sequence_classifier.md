# biome.text.models.sequence_classifier <Badge text="Module"/>
<dl>
<h2 id="biome.text.models.sequence_classifier.SequenceClassifier">SequenceClassifier <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">SequenceClassifier</span> (</span>
    <span>vocab: allennlp.data.vocabulary.Vocabulary</span><span>,</span>
    <span>text_field_embedder: allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder</span><span>,</span>
    <span>seq2vec_encoder: allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder</span><span>,</span>
    <span>seq2seq_encoder: Union[allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, NoneType] = None</span><span>,</span>
    <span>multifield_seq2seq_encoder: Union[allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder, NoneType] = None</span><span>,</span>
    <span>multifield_seq2vec_encoder: Union[allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder, NoneType] = None</span><span>,</span>
    <span>feed_forward: Union[allennlp.modules.feedforward.FeedForward, NoneType] = None</span><span>,</span>
    <span>dropout: Union[float, NoneType] = None</span><span>,</span>
    <span>multifield_dropout: Union[float, NoneType] = None</span><span>,</span>
    <span>initializer: Union[allennlp.nn.initializers.InitializerApplicator, NoneType] = None</span><span>,</span>
    <span>regularizer: Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType] = None</span><span>,</span>
    <span>accuracy: Union[allennlp.training.metrics.categorical_accuracy.CategoricalAccuracy, NoneType] = None</span><span>,</span>
    <span>loss_weights: Dict[str, float] = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>In the most simple form this <code>BaseModelClassifier</code> encodes a sequence with a <code>Seq2VecEncoder</code>, then
predicts a label for the sequence.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong></dt>
<dd>A Vocabulary, required in order to compute sizes for input/output projections
and passed on to the :class:<code>~allennlp.models.model.Model</code> class.</dd>
<dt><strong><code>text_field_embedder</code></strong></dt>
<dd>Used to embed the input text into a <code>TextField</code></dd>
<dt><strong><code>seq2seq_encoder</code></strong></dt>
<dd>Optional Seq2Seq encoder layer for the input text.</dd>
<dt><strong><code>seq2vec_encoder</code></strong></dt>
<dd>Required Seq2Vec encoder layer. If <code>seq2seq_encoder</code> is provided, this encoder
will pool its output. Otherwise, this encoder will operate directly on the output
of the <code>text_field_embedder</code>.</dd>
<dt><strong><code>dropout</code></strong></dt>
<dd>Dropout percentage to use on the output of the Seq2VecEncoder</dd>
<dt><strong><code>multifield_seq2seq_encoder</code></strong></dt>
<dd>Optional Seq2Seq encoder layer for the encoded fields.</dd>
<dt><strong><code>multifield_seq2vec_encoder</code></strong></dt>
<dd>If we use <code>ListField</code>s, this Seq2Vec encoder is required.
If <code>multifield_seq2seq_encoder</code> is provided, this encoder will pool its output.
Otherwise, this encoder will operate directly on the output of the <code>seq2vec_encoder</code>.</dd>
<dt><strong><code>multifield_dropout</code></strong></dt>
<dd>Dropout percentage to use on the output of the doc Seq2VecEncoder</dd>
<dt><strong><code>feed_forward</code></strong></dt>
<dd>A feed forward layer applied to the encoded inputs.</dd>
<dt><strong><code>initializer</code></strong></dt>
<dd>Used to initialize the model parameters.</dd>
<dt><strong><code>regularizer</code></strong></dt>
<dd>Used to regularize the model. Passed on to :class:<code>~allennlp.models.model.Model</code>.</dd>
<dt><strong><code>accuracy</code></strong></dt>
<dd>The accuracy you want to use. By default, we choose a categorical top-1 accuracy.</dd>
<dt><strong><code>loss_weights</code></strong></dt>
<dd>A dict with the labels and the corresponding weights.
These weights will be used in the CrossEntropyLoss function.</dd>
</dl>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase">SequenceClassifierBase</a></li>
<li><a title="biome.text.models.mixins.BiomeClassifierMixin" href="mixins.html#biome.text.models.mixins.BiomeClassifierMixin">BiomeClassifierMixin</a></li>
<li>allennlp.models.model.Model</li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.models.sequence_classifier.SequenceClassifier.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   tokens: Dict[str, torch.Tensor],
   label: torch.Tensor = None,
)  -> Dict[str, torch.Tensor]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>tokens</code></strong></dt>
<dd>The input tokens.
The dictionary is the output of a <code>TextField.as_array()</code>. It gives names to the tensors created by
the <code>TokenIndexer</code>s.
In its most basic form, using a <code>SingleIdTokenIndexer</code>, the dictionary is composed of:
<code>{"tokens": Tensor(batch_size, num_tokens)}</code>.
The keys of the dictionary are defined in the <code>model.yml</code> input.
The dictionary is designed to be passed on directly to a <code>TextFieldEmbedder</code>, that has a
<code>TokenEmbedder</code> for each key in the dictionary (except you set <code>allow_unmatched_keys</code> in the
<code>TextFieldEmbedder</code> to False) and knows how to combine different word/character representations into a
single vector per token in your input.</dd>
<dt><strong><code>label</code></strong></dt>
<dd>A torch tensor representing the sequence of integer gold class label of shape
<code>(batch_size, num_classes)</code>.</dd>
</dl></div>
</dd>
</dl>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase">SequenceClassifierBase</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.decode" href="mixins.html#biome.text.models.mixins.BiomeClassifierMixin.decode">decode</a></code></li>
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.extend_labels" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase.extend_labels">extend_labels</a></code></li>
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.forward_tokens" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase.forward_tokens">forward_tokens</a></code></li>
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.get_metrics" href="mixins.html#biome.text.models.mixins.BiomeClassifierMixin.get_metrics">get_metrics</a></code></li>
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.label_for_index" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase.label_for_index">label_for_index</a></code></li>
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.n_inputs" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase.n_inputs">n_inputs</a></code></li>
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.num_classes" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase.num_classes">num_classes</a></code></li>
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.output_classes" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase.output_classes">output_classes</a></code></li>
<li><code><a title="biome.text.models.sequence_classifier_base.SequenceClassifierBase.output_layer" href="sequence_classifier_base.html#biome.text.models.sequence_classifier_base.SequenceClassifierBase.output_layer">output_layer</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>