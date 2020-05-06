# biome.text.predictors.default_predictor <Badge text="Module"/>
<dl>
<h2 id="biome.text.predictors.default_predictor.DefaultBasePredictor">DefaultBasePredictor <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">DefaultBasePredictor</span> (model: allennlp.models.model.Model, dataset_reader: <a title="biome.text.dataset_readers.datasource_reader.DataSourceReader" href="../dataset_readers/datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader">DataSourceReader</a>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>a <code>Predictor</code> is a thin wrapper around an AllenNLP model that handles JSON -&gt; JSON predictions
that can be used for serving models through the web API or making predictions in bulk.</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.predictors.predictor.Predictor</li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<h3 id="biome.text.predictors.default_predictor.DefaultBasePredictor.predict_json">predict_json <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predict_json</span> (</span>
   self,
   inputs: Dict[str, Any],
)  -> Dict[str, Any]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.predictors.default_predictor.DefaultBasePredictor.predict_batch_json">predict_batch_json <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">predict_batch_json</span> (</span>
   self,
   inputs: List[Dict[str, Any]],
)  -> List[Dict[str, Any]]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
</dd>
</dl>