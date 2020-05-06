# biome.text.api_new.helpers <Badge text="Module"/>
<dl>
<h3 id="biome.text.api_new.helpers.yaml_to_dict">yaml_to_dict <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">yaml_to_dict</span></span>(<span>filepath: str) -> Dict[str, Any]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Loads a yaml file into a data dictionary</p></div>
</dd>
<h3 id="biome.text.api_new.helpers.get_compatible_doc_type">get_compatible_doc_type <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_compatible_doc_type</span></span>(<span>client: elasticsearch.client.Elasticsearch) -> str</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Find a compatible name for doc type by checking the cluster info
Parameters</p>
<hr>
<dl>
<dt><strong><code>client</code></strong></dt>
<dd>The elasticsearch client</dd>
</dl>
<h2 id="returns">Returns</h2>
<pre><code>A compatible name for doc type in function of cluster version
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.helpers.get_env_cuda_device">get_env_cuda_device <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_env_cuda_device</span></span>(<span>) -> int</span>
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
<h3 id="biome.text.api_new.helpers.update_method_signature">update_method_signature <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">update_method_signature</span> (</span>
   signature: inspect.Signature,
   to_method,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Updates signature to method</p></div>
</dd>
<h3 id="biome.text.api_new.helpers.isgeneric">isgeneric <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">isgeneric</span></span>(<span>class_type: Type) -> bool</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Checks if a class type is a generic type (List[str] or Union[str, int]</p></div>
</dd>
<h3 id="biome.text.api_new.helpers.is_running_on_notebook">is_running_on_notebook <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">is_running_on_notebook</span></span>(<span>) -> bool</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Checks if code is running inside a jupyter notebook</p></div>
</dd>
<h3 id="biome.text.api_new.helpers.split_signature_params_by_predicate">split_signature_params_by_predicate <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">split_signature_params_by_predicate</span> (</span>
   signature_function: Callable,
   predicate: Callable,
)  -> Tuple[List[inspect.Parameter], List[inspect.Parameter]]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Splits parameters signature by defined boolean predicate function</p></div>
</dd>
</dl>