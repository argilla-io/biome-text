# biome.text.pipelines.learn.allennlp.learn <Badge text="Module"/>
<dl>
<h3 id="biome.text.pipelines.learn.allennlp.learn.learn">learn <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">learn</span> (</span>
   output: str,
   model_spec: Union[str, NoneType] = None,
   model_binary: Union[str, NoneType] = None,
   vocab: Union[str, NoneType] = None,
   trainer_path: str = '',
   train_cfg: str = '',
   validation_cfg: Union[str, NoneType] = None,
   test_cfg: Union[str, NoneType] = None,
   verbose: bool = False,
)  -> allennlp.models.model.Model
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.pipelines.learn.allennlp.learn.recover_output_folder">recover_output_folder <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">recover_output_folder</span> (</span>
   output: str,
   params: allennlp.common.params.Params,
)  -> bool
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>If output folder already exists, we automatically recover the generated vocab in this folder.</p>
<p>Allows reuse the generated vocab if something went wrong in previous executions</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>output</code></strong></dt>
<dd>Path to the output folder</dd>
<dt><strong><code>params</code></strong></dt>
<dd>Parameters for the train command</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>is_recovered</code></dt>
<dd>True if existing output folder is recovered, False if output folder does not exist.</dd>
</dl></div>
</dd>
</dl>