# biome.text.models.sequence_pair_classifier <Badge text="Module"/>
<dl>
<h2 id="biome.text.models.sequence_pair_classifier.SequencePairClassifier">SequencePairClassifier <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">SequencePairClassifier</span> (</span>
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
<div class="desc"><p>This <code><a title="biome.text.models.sequence_pair_classifier.SequencePairClassifier" href="#biome.text.models.sequence_pair_classifier.SequencePairClassifier">SequencePairClassifier</a></code> uses a siamese network architecture to perform a classification task between a pair
of records or documents.</p>
<p>The classifier can be configured to take into account the hierarchical structure of documents
and multi-field records.</p>
<p>A record/document can be (1) single-field (single sentence): composed of a sequence of
tokens, or (2) multi-field (multi-sentence): a sequence of fields with each of the fields containing a sequence of
tokens. In the case of multi-field a doc_seq2vec_encoder and optionally a doc_seq2seq_encoder should be configured,
for encoding each of the fields into a single vector encoding the full record/doc must be configured.</p>
<p>The sequences are encoded into two single vectors, the resulting vectors are concatenated and fed to a
linear classification layer.</p>
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
<h3 id="biome.text.models.sequence_pair_classifier.SequencePairClassifier.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   record1: Dict[str, torch.Tensor],
   record2: Dict[str, torch.Tensor],
   label: torch.Tensor = None,
)  -> Dict[str, torch.Tensor]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>record1</code></strong></dt>
<dd>The first input tokens.
The dictionary is the output of a <code>TextField.as_array()</code>. It gives names to the tensors created by
the <code>TokenIndexer</code>s.
In its most basic form, using a <code>SingleIdTokenIndexer</code>, the dictionary is composed of:
<code>{"tokens": Tensor(batch_size, num_tokens)}</code>.
The keys of the dictionary are defined in the <code>model.yml</code> input.
The dictionary is designed to be passed on directly to a <code>TextFieldEmbedder</code>, that has a
<code>TokenEmbedder</code> for each key in the dictionary (except you set <code>allow_unmatched_keys</code> in the
<code>TextFieldEmbedder</code> to False) and knows how to combine different word/character representations into a
single vector per token in your input.</dd>
<dt><strong><code>record2</code></strong></dt>
<dd>The second input tokens.</dd>
<dt><strong><code>label</code></strong> :&ensp;<code>torch.LongTensor</code>, optional <code>(default = None)</code></dt>
<dd>A torch tensor representing the sequence of integer gold class label of shape
<code>(batch_size, num_classes)</code>.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>An output dictionary consisting of:</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>logits</code></strong> :&ensp;<code>torch.FloatTensor</code></dt>
<dd>A tensor of shape <code>(batch_size, num_tokens, tag_vocab_size)</code> representing
unnormalised log probabilities of the tag classes.</dd>
<dt><strong><code>class_probabilities</code></strong> :&ensp;<code>torch.FloatTensor</code></dt>
<dd>A tensor of shape <code>(batch_size, num_tokens, tag_vocab_size)</code> representing
a distribution of the tag classes per word.</dd>
<dt><strong><code>loss</code></strong> :&ensp;<code>torch.FloatTensor</code>, optional</dt>
<dd>A scalar loss to be optimised.</dd>
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