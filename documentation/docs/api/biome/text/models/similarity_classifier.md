# biome.text.models.similarity_classifier <Badge text="Module"/>
<dl>
<h2 id="biome.text.models.similarity_classifier.SimilarityClassifier">SimilarityClassifier <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">SimilarityClassifier</span> (</span>
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
    <span>margin: float = 0.5</span><span>,</span>
    <span>verification_weight: float = 2.0</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This <code><a title="biome.text.models.similarity_classifier.SimilarityClassifier" href="#biome.text.models.similarity_classifier.SimilarityClassifier">SimilarityClassifier</a></code> uses a siamese network architecture to perform a binary classification task:
are two inputs similar or not?
The two input sequences are encoded with two single vectors, the resulting vectors are concatenated and fed to a
linear classification layer.</p>
<p>Apart from the CrossEntropy loss, this model includes a CosineEmbedding loss
(<a href="https://pytorch.org/docs/stable/nn.html#cosineembeddingloss">https://pytorch.org/docs/stable/nn.html#cosineembeddingloss</a>) that will drive the network to create
vector clusters for each "class" in the data.
Make sure that the label "same" is indexed as 0, and the label "different" as 1!!!
Make sure that the dropout of the last Seq2Vec or the last FeedForward layer is set to 0!!!
(Deep Learning Face Representation by Joint Identification-Verification, <a href="https://arxiv.org/pdf/1406.4773.pdf">https://arxiv.org/pdf/1406.4773.pdf</a>)</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>kwargs</code></strong></dt>
<dd>See the <code>BaseModelClassifier</code> for a description of the parameters.</dd>
<dt><strong><code>margin</code></strong></dt>
<dd>This parameter is passed on to the CosineEmbedding loss. It provides a margin,
at which dissimilar vectors are not driven further apart.
Can be between -1 (always drive apart) and 1 (never drive apart).</dd>
<dt><strong><code>verification_weight</code></strong></dt>
<dd>Defines the weight of the verification loss in the final loss sum:
loss = CrossEntropy + w * CosineEmbedding</dd>
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
<h3 id="biome.text.models.similarity_classifier.SimilarityClassifier.forward">forward <Badge text="Method"/></h3>
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
<div class="desc"><p>The architecture is basically:
Embedding -&gt; Seq2Seq -&gt; Seq2Vec -&gt; Dropout -&gt; (Optional: MultiField stuff) -&gt; FeedForward
-&gt; Concatenation -&gt; Classification layer</p>
<h2 id="parameters">Parameters</h2>
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
<h2 id="biome.text.models.similarity_classifier.ContrastiveLoss">ContrastiveLoss <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">ContrastiveLoss</span> ()</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Computes a contrastive loss given a distance.</p>
<p>We do not use it at the moment, i leave it here just in case.</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>torch.nn.modules.module.Module</li>
</ul>
<dl>
<h3 id="biome.text.models.similarity_classifier.ContrastiveLoss.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   distance,
   label,
   margin,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Compute the loss.</p>
<p>Important: Make sure label = 0 corresponds to the same case, label = 1 to the different case!</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>distance</code></strong></dt>
<dd>Distance between the two input vectors</dd>
<dt><strong><code>label</code></strong></dt>
<dd>Label if the two input vectors belong to the same or different class.</dd>
<dt><strong><code>margin</code></strong></dt>
<dd>If the distance is larger than the margin, the distance of different class vectors
does not contribute to the loss.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>loss</code></dt>
<dd>&nbsp;</dd>
</dl></div>
</dd>
</dl>
</dd>
</dl>