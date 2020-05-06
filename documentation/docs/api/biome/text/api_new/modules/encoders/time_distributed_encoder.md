# biome.text.api_new.modules.encoders.time_distributed_encoder <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.modules.encoders.time_distributed_encoder.TimeDistributedEncoder">TimeDistributedEncoder <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TimeDistributedEncoder</span> (encoder: allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Wraps a Seq2SeqEncoder into a TimeDistributed module and implements the Seq2SeqEncoder API</p>
<p>Initializes internal Module state, shared by both nn.Module and ScriptModule.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder</li>
<li>allennlp.modules.encoder_base._EncoderBase</li>
<li>torch.nn.modules.module.Module</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.api_new.modules.encoders.time_distributed_encoder.TimeDistributedEncoder.forward">forward <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">forward</span> (</span>
   self,
   *input,
   **inputs,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Defines the computation performed at every call.</p>
<p>Should be overridden by all subclasses.</p>
<div class="admonition note">
<p class="admonition-title">Note</p>
<p>Although the recipe for forward pass needs to be defined within
this function, one should call the :class:<code>Module</code> instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.</p>
</div></div>
</dd>
<h3 id="biome.text.api_new.modules.encoders.time_distributed_encoder.TimeDistributedEncoder.is_bidirectional">is_bidirectional <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">is_bidirectional</span></span>(<span>self) -> bool</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Returns <code>True</code> if this encoder is bidirectional.
If so, we assume the forward direction
of the encoder is the first half of the final dimension, and the backward direction is the
second half.</p></div>
</dd>
<h3 id="biome.text.api_new.modules.encoders.time_distributed_encoder.TimeDistributedEncoder.get_output_dim">get_output_dim <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_output_dim</span></span>(<span>self) -> int</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Returns the dimension of each vector in the sequence output by this <code>Seq2SeqEncoder</code>.
This is <code>not</code> the shape of the returned tensor, but the last element of that shape.</p></div>
</dd>
<h3 id="biome.text.api_new.modules.encoders.time_distributed_encoder.TimeDistributedEncoder.get_input_dim">get_input_dim <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_input_dim</span></span>(<span>self)</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Returns the dimension of the vector input for each element in the sequence input
to a <code>Seq2SeqEncoder</code>. This is <code>not</code> the shape of the input tensor, but the
last element of that shape.</p></div>
</dd>
</dl>
</dd>
</dl>