# biome.text.interpreters.integrated_gradient <Badge text="Module"/>
<dl>
<h2 id="biome.text.interpreters.integrated_gradient.IntegratedGradient">IntegratedGradient <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">IntegratedGradient</span> (predictor: allennlp.predictors.predictor.Predictor)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Interprets the prediction using Integrated Gradients (<a href="https://arxiv.org/abs/1703.01365">https://arxiv.org/abs/1703.01365</a>)</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.interpret.saliency_interpreters.integrated_gradient.IntegratedGradient</li>
<li>allennlp.interpret.saliency_interpreters.saliency_interpreter.SaliencyInterpreter</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.interpreters.integrated_gradient.IntegratedGradient.saliency_interpret_from_json">saliency_interpret_from_json <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">saliency_interpret_from_json</span> (</span>
   self,
   inputs: Dict[str, Any],
)  -> Dict[str, Any]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>This function finds a modification to the input text that would change the model's
prediction in some desired manner (e.g., an adversarial attack).</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>inputs</code></strong> :&ensp;<code>JsonDict</code></dt>
<dd>The input you want to interpret (the same as the argument to a Predictor, e.g., predict_json()).</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><strong><code>interpretation</code></strong> :&ensp;<code>JsonDict</code></dt>
<dd>Contains the normalized saliency values for each input token. The dict has entries for
each instance in the inputs JsonDict, e.g., <code>{instance_1: ..., instance_2:, ... }</code>.
Each one of those entries has entries for the saliency of the inputs, e.g.,
<code>{grad_input_1: ..., grad_input_2: ... }</code>.</dd>
</dl></div>
</dd>
</dl>
</dd>
</dl>