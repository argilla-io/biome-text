# biome.text.pipelines.learn.allennlp.callbacks.evaluate <Badge text="Module"/>
<dl>
<h2 id="biome.text.pipelines.learn.allennlp.callbacks.evaluate.EvaluateCallback">EvaluateCallback <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">EvaluateCallback</span> (serialization_dir: str)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This callback allows to a callback trainer evaluate the model against a test dataset</p>
<h2 id="attributes">Attributes</h2>
<p>serialization_dir:str
The experiment folder</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.training.callbacks.callback.Callback</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.pipelines.learn.allennlp.callbacks.evaluate.EvaluateCallback.evaluate_dataset">evaluate_dataset <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">evaluate_dataset</span> (</span>
   self,
   trainer: allennlp.training.callback_trainer.CallbackTrainer,
)  -> NoneType
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>This method launches an test dataset (if defined) evaluation when the training ends
and adds the test metrics to trainer metrics before they are processed (thanks to priority argument)</p>
<h2 id="parameters">Parameters</h2>
<p>trainer:CallbackTrainer
The main callback trainer</p></div>
</dd>
</dl>
</dd>
</dl>