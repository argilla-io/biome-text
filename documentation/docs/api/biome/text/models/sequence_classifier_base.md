# biome.text.models.sequence_classifier_base <Badge text="Module"/>
<dl>
<h2 id="biome.text.models.sequence_classifier_base.SequenceClassifierBase">SequenceClassifierBase <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">SequenceClassifierBase</span> (</span>
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
<li><a title="biome.text.models.mixins.BiomeClassifierMixin" href="mixins.html#biome.text.models.mixins.BiomeClassifierMixin">BiomeClassifierMixin</a></li>
<li>allennlp.models.model.Model</li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.models.sequence_classifier.SequenceClassifier" href="sequence_classifier.html#biome.text.models.sequence_classifier.SequenceClassifier">SequenceClassifier</a></li>
<li><a title="biome.text.models.sequence_pair_classifier.SequencePairClassifier" href="sequence_pair_classifier.html#biome.text.models.sequence_pair_classifier.SequencePairClassifier">SequencePairClassifier</a></li>
<li><a title="biome.text.models.similarity_classifier.SimilarityClassifier" href="similarity_classifier.html#biome.text.models.similarity_classifier.SimilarityClassifier">SimilarityClassifier</a></li>
</ul>
<h3>Instance variables</h3>
<dl>
<dt id="biome.text.models.sequence_classifier_base.SequenceClassifierBase.n_inputs"><code class="name">var <span class="ident">n_inputs</span></code></dt>
<dd>
<div class="desc"><p>This value is used for calculate the output layer dimension. Default value is 1</p></div>
</dd>
<dt id="biome.text.models.sequence_classifier_base.SequenceClassifierBase.num_classes"><code class="name">var <span class="ident">num_classes</span></code></dt>
<dd>
<div class="desc"><p>Number of output classes</p></div>
</dd>
<dt id="biome.text.models.sequence_classifier_base.SequenceClassifierBase.output_classes"><code class="name">var <span class="ident">output_classes</span> : List[str]</code></dt>
<dd>
<div class="desc"><p>The output token classes</p></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.models.sequence_classifier_base.SequenceClassifierBase.label_for_index">label_for_index <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">label_for_index</span> (</span>
   self,
   idx,
)  -> str
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Token label for label index</p></div>
</dd>
<h3 id="biome.text.models.sequence_classifier_base.SequenceClassifierBase.extend_labels">extend_labels <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">extend_labels</span> (</span>
   self,
   labels: List[str],
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Extends the number of output labels</p></div>
</dd>
<h3 id="biome.text.models.sequence_classifier_base.SequenceClassifierBase.forward_tokens">forward_tokens <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward_tokens</span> (</span>
   self,
   tokens: Dict[str, torch.Tensor],
)  -> torch.Tensor
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Apply the whole forward chain but last layer (output)</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>tokens</code></strong></dt>
<dd>The tokens tensor</dd>
</dl>
<h2 id="returns">Returns</h2>
<p>A <code>Tensor</code></p></div>
</dd>
<h3 id="biome.text.models.sequence_classifier_base.SequenceClassifierBase.output_layer">output_layer <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">output_layer</span> (</span>
   self,
   encoded_text: torch.Tensor,
   label,
)  -> Dict[str, torch.Tensor]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><h2 id="returns">Returns</h2>
<dl>
<dt><code>An output dictionary consisting of:</code></dt>
<dd>&nbsp;</dd>
<dt><strong><code>logits</code></strong> :&ensp;<code>:class:</code>~torch.Tensor``</dt>
<dd>A tensor of shape <code>(batch_size, num_classes)</code> representing
the logits of the classifier model.</dd>
<dt><strong><code>class_probabilities</code></strong> :&ensp;<code>:class:</code>~torch.Tensor``</dt>
<dd>A tensor of shape <code>(batch_size, num_classes)</code> representing
the softmax probabilities of the classes.</dd>
<dt><strong><code>loss</code></strong> :&ensp;<code>:class:</code>~torch.Tensor``, optional</dt>
<dd>A scalar loss to be optimised.</dd>
</dl></div>
</dd>
<h3 id="biome.text.models.sequence_classifier_base.SequenceClassifierBase.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   *inputs,
)  -> Dict[str, torch.Tensor]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Defines the forward pass of the model. In addition, to facilitate easy training,
this method is designed to compute a loss function defined by a user.</p>
<p>The input is comprised of everything required to perform a
training update, <code>including</code> labels - you define the signature here!
It is down to the user to ensure that inference can be performed
without the presence of these labels. Hence, any inputs not available at
inference time should only be used inside a conditional block.</p>
<p>The intended sketch of this method is as follows::</p>
<pre><code>def forward(self, input1, input2, targets=None):
    ....
    ....
    output1 = self.layer1(input1)
    output2 = self.layer2(input2)
    output_dict = {"output1": output1, "output2": output2}
    if targets is not None:
        # Function returning a scalar torch.Tensor, defined by the user.
        loss = self._compute_loss(output1, output2, targets)
        output_dict["loss"] = loss
    return output_dict
</code></pre>
<h2 id="parameters">Parameters</h2>
<p>inputs:
Tensors comprising everything needed to perform a training update, <code>including</code> labels,
which should be optional (i.e have a default value of <code>None</code>).
At inference time,
simply pass the relevant inputs, not including the labels.</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>output_dict</code></strong> :&ensp;<code>Dict[str, torch.Tensor]</code></dt>
<dd>The outputs from the model. In order to train a model using the
:class:<code>~allennlp.training.Trainer</code> api, you must provide a "loss" key pointing to a
scalar <code>torch.Tensor</code> representing the loss to be optimized.</dd>
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