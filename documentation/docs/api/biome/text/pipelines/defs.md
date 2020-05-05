# biome.text.pipelines.defs <Badge text="Module"/>
<dl>
<h2 id="biome.text.pipelines.defs.ElasticsearchConfig">ElasticsearchConfig <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">ElasticsearchConfig</span> (es_host: str, es_index: str)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Elasticsearch configuration data class</p></div>
</dd>
<h2 id="biome.text.pipelines.defs.ExploreConfig">ExploreConfig <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">ExploreConfig</span> (</span>
    <span>batch_size: int = 500</span><span>,</span>
    <span>prediction_cache_size: int = 0</span><span>,</span>
    <span>interpret: bool = False</span><span>,</span>
    <span>force_delete: bool = True</span><span>,</span>
    <span>**metadata</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Explore configuration data class</p></div>
</dd>
</dl>