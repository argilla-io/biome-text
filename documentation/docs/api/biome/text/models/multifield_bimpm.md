# biome.text.models.multifield_bimpm <Badge text="Module"/>
<dl>
<h2 id="biome.text.models.multifield_bimpm.MultifieldBiMpm">MultifieldBiMpm <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">MultifieldBiMpm</span> (</span>
    <span>vocab: allennlp.data.vocabulary.Vocabulary</span><span>,</span>
    <span>text_field_embedder: allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder</span><span>,</span>
    <span>matcher_word: allennlp.modules.bimpm_matching.BiMpmMatching</span><span>,</span>
    <span>encoder: allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder</span><span>,</span>
    <span>matcher_forward: allennlp.modules.bimpm_matching.BiMpmMatching</span><span>,</span>
    <span>aggregator: allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder</span><span>,</span>
    <span>classifier_feedforward: allennlp.modules.feedforward.FeedForward</span><span>,</span>
    <span>matcher_backward: allennlp.modules.bimpm_matching.BiMpmMatching = None</span><span>,</span>
    <span>encoder2: allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder = None</span><span>,</span>
    <span>matcher2_forward: allennlp.modules.bimpm_matching.BiMpmMatching = None</span><span>,</span>
    <span>matcher2_backward: allennlp.modules.bimpm_matching.BiMpmMatching = None</span><span>,</span>
    <span>dropout: float = 0.1</span><span>,</span>
    <span>multifield: bool = True</span><span>,</span>
    <span>initializer: allennlp.nn.initializers.InitializerApplicator = &lt;allennlp.nn.initializers.InitializerApplicator object&gt;</span><span>,</span>
    <span>regularizer: Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType] = None</span><span>,</span>
    <span>accuracy: Union[allennlp.training.metrics.categorical_accuracy.CategoricalAccuracy, NoneType] = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This <code>Model</code> is a version of AllenNLPs implementation of the BiMPM model described in
<code>Bilateral Multi-Perspective Matching for Natural Language Sentences &lt;https://arxiv.org/abs/1702.03814&gt;</code>_
by Zhiguo Wang et al., 2017.</p>
<p>This version adds the feature of being compatible with multiple inputs for the two records.
The matching will be done for all possible combinations between the two records, that is:
(r1_1, r2_1), (r1_1, r2_2), &hellip;, (r1_2, r2_1), (r1_2, r2_2), &hellip;</p>
<p>This version also allows you to apply only one encoder, and to leave out the backward matching.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;<code>Vocabulary</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>text_field_embedder</code></strong> :&ensp;<code>TextFieldEmbedder</code></dt>
<dd>Used to embed the <code>record1</code> and <code>record2</code> <code>TextFields</code> we get as input to the
model.</dd>
<dt><strong><code>matcher_word</code></strong> :&ensp;<code>BiMpmMatching</code></dt>
<dd>BiMPM matching on the output of word embeddings of record1 and record2.</dd>
<dt><strong><code>encoder</code></strong> :&ensp;<code>Seq2SeqEncoder</code></dt>
<dd>Encoder layer for record1 and record2</dd>
<dt><strong><code>matcher_forward</code></strong> :&ensp;<code>BiMPMMatching</code></dt>
<dd>BiMPM matching for the forward output of the encoder layer</dd>
<dt><strong><code>aggregator</code></strong> :&ensp;<code>Seq2VecEncoder</code></dt>
<dd>Aggregator of all BiMPM matching vectors</dd>
<dt><strong><code>classifier_feedforward</code></strong> :&ensp;<code>FeedForward</code></dt>
<dd>Fully connected layers for classification.
A linear output layer with the number of labels at the end will be added automatically!!!</dd>
<dt><strong><code>matcher_backward</code></strong> :&ensp;<code>BiMPMMatching</code>, optional</dt>
<dd>BiMPM matching for the backward output of the encoder layer</dd>
<dt><strong><code>encoder2</code></strong> :&ensp;<code>Seq2SeqEncoder</code>, optional</dt>
<dd>Encoder layer for encoded record1 and encoded record2</dd>
<dt><strong><code>matcher2_forward</code></strong> :&ensp;<code>BiMPMMatching</code>, optional</dt>
<dd>BiMPM matching for the forward output of the second encoder layer</dd>
<dt><strong><code>matcher2_backward</code></strong> :&ensp;<code>BiMPMMatching</code>, optional</dt>
<dd>BiMPM matching for the backward output of the second encoder layer</dd>
<dt><strong><code>dropout</code></strong> :&ensp;<code>float</code>, optional <code>(default=0.1)</code></dt>
<dd>Dropout percentage to use.</dd>
<dt><strong><code>multifield</code></strong> :&ensp;<code>bool</code>, optional <code>(default=False)</code></dt>
<dd>Are there multiple inputs for each record, that is do the inputs come from <code>ListField</code>s?</dd>
<dt><strong><code>initializer</code></strong> :&ensp;<code>InitializerApplicator</code>, optional <code>(default=``InitializerApplicator()``)</code></dt>
<dd>If provided, will be used to initialize the model parameters.</dd>
<dt><strong><code>regularizer</code></strong> :&ensp;<code>RegularizerApplicator</code>, optional <code>(default=``None``)</code></dt>
<dd>If provided, will be used to calculate the regularization penalty during training.</dd>
<dt><strong><code>accuracy</code></strong></dt>
<dd>The accuracy you want to use. By default, we choose a categorical top-1 accuracy.</dd>
</dl>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.models.mixins.BiomeClassifierMixin" href="mixins.html#biome.text.models.mixins.BiomeClassifierMixin">BiomeClassifierMixin</a></li>
<li>allennlp.models.model.Model</li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.models.multifield_bimpm.MultifieldBiMpm.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   record1: Dict[str, torch.LongTensor],
   record2: Dict[str, torch.LongTensor],
   label: torch.LongTensor = None,
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
<li><code><b><a title="biome.text.models.mixins.BiomeClassifierMixin" href="mixins.html#biome.text.models.mixins.BiomeClassifierMixin">BiomeClassifierMixin</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.models.mixins.BiomeClassifierMixin.decode" href="mixins.html#biome.text.models.mixins.BiomeClassifierMixin.decode">decode</a></code></li>
<li><code><a title="biome.text.models.mixins.BiomeClassifierMixin.get_metrics" href="mixins.html#biome.text.models.mixins.BiomeClassifierMixin.get_metrics">get_metrics</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>