# biome.text.pipelines.learn.allennlp.defs <Badge text="Module"/>
<dl>
<h2 id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig">BiomeConfig <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">BiomeConfig</span> (</span>
    <span>model_path: str = None</span><span>,</span>
    <span>trainer_path: str = None</span><span>,</span>
    <span>vocab_path: str = None</span><span>,</span>
    <span>train_path: str = None</span><span>,</span>
    <span>validation_path: str = None</span><span>,</span>
    <span>test_path: str = None</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This class contains biome config parameters usually necessary to run the biome commands.</p>
<p>It also allows a transformation of these parameters to AllenNLP parameters.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>model_path</code></strong></dt>
<dd>Path to the model yaml file</dd>
<dt><strong><code>trainer_path</code></strong></dt>
<dd>Path to the trainer yaml file</dd>
<dt><strong><code>vocab_path</code></strong></dt>
<dd>Path to the vocab yaml file</dd>
<dt><strong><code>train_path</code></strong></dt>
<dd>Path to the data source yaml file of the training set</dd>
<dt><strong><code>validation_path</code></strong></dt>
<dd>Path to the data source yaml file of the validation set</dd>
<dt><strong><code>test_path</code></strong></dt>
<dd>Path to the data source yaml file of the test set</dd>
</dl></div>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.CUDA_DEVICE_FIELD"><code class="name">var <span class="ident">CUDA_DEVICE_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.MODEL_FIELD"><code class="name">var <span class="ident">MODEL_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.TRAINER_FIELD"><code class="name">var <span class="ident">TRAINER_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.TRAIN_DATA_FIELD"><code class="name">var <span class="ident">TRAIN_DATA_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.VALIDATION_DATA_FIELD"><code class="name">var <span class="ident">VALIDATION_DATA_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.TEST_DATA_FIELD"><code class="name">var <span class="ident">TEST_DATA_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.EVALUATE_ON_TEST_FIELD"><code class="name">var <span class="ident">EVALUATE_ON_TEST_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.DATASET_READER_FIELD"><code class="name">var <span class="ident">DATASET_READER_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.TYPE_FIELD"><code class="name">var <span class="ident">TYPE_FIELD</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.DEFAULT_CALLBACK_TRAINER"><code class="name">var <span class="ident">DEFAULT_CALLBACK_TRAINER</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.yaml_to_dict">yaml_to_dict <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">yaml_to_dict</span></span>(<span>path: str) -> Dict[str, Any]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Reads a yaml file and returns a dict.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>Path to the yaml file</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>dict</code></dt>
<dd>If no path is specified, returns an empty dict</dd>
</dl></div>
</dd>
<h3 id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.get_cuda_device">get_cuda_device <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_cuda_device</span></span>(<span>) -> int</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Gets the cuda device from an environment variable.</p>
<p>This is necessary to activate a GPU if available</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>cuda_device</code></dt>
<dd>The integer number of the CUDA device</dd>
</dl></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.pipelines.learn.allennlp.defs.BiomeConfig.to_allennlp_params">to_allennlp_params <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_allennlp_params</span></span>(<span>self) -> Dict</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Transforms the cfg to AllenNLP parameters by basically joining all biome configurations.</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>params</code></dt>
<dd>A dict in the right format containing the AllenNLP parameters</dd>
</dl></div>
</dd>
</dl>
</dd>
</dl>