# biome.text.models.biome_bimpm <Badge text="Module"/>
<dl>
<h2 id="biome.text.models.biome_bimpm.BiomeBiMpm">BiomeBiMpm <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">BiomeBiMpm</span> (</span>
    <span>vocab: allennlp.data.vocabulary.Vocabulary</span><span>,</span>
    <span>text_field_embedder: allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder</span><span>,</span>
    <span>matcher_word: allennlp.modules.bimpm_matching.BiMpmMatching</span><span>,</span>
    <span>encoder1: allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder</span><span>,</span>
    <span>matcher_forward1: allennlp.modules.bimpm_matching.BiMpmMatching</span><span>,</span>
    <span>matcher_backward1: allennlp.modules.bimpm_matching.BiMpmMatching</span><span>,</span>
    <span>encoder2: allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder</span><span>,</span>
    <span>matcher_forward2: allennlp.modules.bimpm_matching.BiMpmMatching</span><span>,</span>
    <span>matcher_backward2: allennlp.modules.bimpm_matching.BiMpmMatching</span><span>,</span>
    <span>aggregator: allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder</span><span>,</span>
    <span>classifier_feedforward: allennlp.modules.feedforward.FeedForward</span><span>,</span>
    <span>dropout: float = 0.1</span><span>,</span>
    <span>initializer: allennlp.nn.initializers.InitializerApplicator = &lt;allennlp.nn.initializers.InitializerApplicator object&gt;</span><span>,</span>
    <span>regularizer: Union[allennlp.nn.regularizers.regularizer_applicator.RegularizerApplicator, NoneType] = None</span><span>,</span>
    <span>accuracy: Union[allennlp.training.metrics.categorical_accuracy.CategoricalAccuracy, NoneType] = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This <code>Model</code> implements BiMPM model described in <code>Bilateral Multi-Perspective Matching
for Natural Language Sentences &lt;https://arxiv.org/abs/1702.03814&gt;</code><em> by Zhiguo Wang et al., 2017.
Also please refer to the <code>TensorFlow implementation &lt;https://github.com/zhiguowang/BiMPM/&gt;</code></em> and
<code>PyTorch implementation &lt;https://github.com/galsang/BIMPM-pytorch&gt;</code>_.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>vocab</code></strong> :&ensp;<code>Vocabulary</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>text_field_embedder</code></strong> :&ensp;<code>TextFieldEmbedder</code></dt>
<dd>Used to embed the <code>premise</code> and <code>hypothesis</code> <code>TextFields</code> we get as input to the
model.</dd>
<dt><strong><code>matcher_word</code></strong> :&ensp;<code>BiMpmMatching</code></dt>
<dd>BiMPM matching on the output of word embeddings of premise and hypothesis.</dd>
<dt><strong><code>encoder1</code></strong> :&ensp;<code>Seq2SeqEncoder</code></dt>
<dd>First encoder layer for the premise and hypothesis</dd>
<dt><strong><code>matcher_forward1</code></strong> :&ensp;<code>BiMPMMatching</code></dt>
<dd>BiMPM matching for the forward output of first encoder layer</dd>
<dt><strong><code>matcher_backward1</code></strong> :&ensp;<code>BiMPMMatching</code></dt>
<dd>BiMPM matching for the backward output of first encoder layer</dd>
<dt><strong><code>encoder2</code></strong> :&ensp;<code>Seq2SeqEncoder</code></dt>
<dd>Second encoder layer for the premise and hypothesis</dd>
<dt><strong><code>matcher_forward2</code></strong> :&ensp;<code>BiMPMMatching</code></dt>
<dd>BiMPM matching for the forward output of second encoder layer</dd>
<dt><strong><code>matcher_backward2</code></strong> :&ensp;<code>BiMPMMatching</code></dt>
<dd>BiMPM matching for the backward output of second encoder layer</dd>
<dt><strong><code>aggregator</code></strong> :&ensp;<code>Seq2VecEncoder</code></dt>
<dd>Aggregator of all BiMPM matching vectors</dd>
<dt><strong><code>classifier_feedforward</code></strong> :&ensp;<code>FeedForward</code></dt>
<dd>Fully connected layers for classification.</dd>
<dt><strong><code>dropout</code></strong> :&ensp;<code>float</code>, optional <code>(default=0.1)</code></dt>
<dd>Dropout percentage to use.</dd>
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
<li>allennlp.models.bimpm.BiMpm</li>
<li>allennlp.models.model.Model</li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.models.biome_bimpm.BiomeBiMpm.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   record1: Dict[str, torch.LongTensor],
   record2: Dict[str, torch.LongTensor],
   label: torch.Tensor = None,
)  -> Dict[str, torch.Tensor]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>premise</code></strong> :&ensp;<code>Dict[str, torch.LongTensor]</code></dt>
<dd>The premise from a <code>TextField</code></dd>
<dt><strong><code>hypothesis</code></strong> :&ensp;<code>Dict[str, torch.LongTensor]</code></dt>
<dd>The hypothesis from a <code>TextField</code></dd>
<dt><strong><code>label</code></strong> :&ensp;<code>torch.LongTensor</code>, optional <code>(default = None)</code></dt>
<dd>The label for the pair of the premise and the hypothesis</dd>
<dt><strong><code>metadata</code></strong> :&ensp;<code>List[Dict[str, Any]]</code>, optional<code>, (default = None)</code></dt>
<dd>Additional information about the pair</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>An output dictionary consisting of:</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>logits</code></strong> :&ensp;<code>torch.FloatTensor</code></dt>
<dd>A tensor of shape <code>(batch_size, num_labels)</code> representing unnormalised log
probabilities of the entailment label.</dd>
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