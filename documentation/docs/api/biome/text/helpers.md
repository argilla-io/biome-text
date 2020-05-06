# biome.text.helpers <Badge text="Module"/>
<dl>
<h3 id="biome.text.helpers.get_compatible_doc_type">get_compatible_doc_type <Badge text="Function"/></h3>
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
</dl>