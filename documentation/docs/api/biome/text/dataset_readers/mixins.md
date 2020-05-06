# biome.text.dataset_readers.mixins <Badge text="Module"/>
<dl>
<h2 id="biome.text.dataset_readers.mixins.CacheableMixin">CacheableMixin <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">CacheableMixin</span> ()</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This <code><a title="biome.text.dataset_readers.mixins.CacheableMixin" href="#biome.text.dataset_readers.mixins.CacheableMixin">CacheableMixin</a></code> allow in memory cache mechanism</p></div>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.dataset_readers.datasource_reader.DataSourceReader" href="datasource_reader.html#biome.text.dataset_readers.datasource_reader.DataSourceReader">DataSourceReader</a></li>
</ul>
<dl>
<h3 id="biome.text.dataset_readers.mixins.CacheableMixin.get">get <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get</span></span>(<span>key) -> Union[Any, NoneType]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Get a value from cache by key</p></div>
</dd>
<h3 id="biome.text.dataset_readers.mixins.CacheableMixin.set">set <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">set</span> (</span>
   key,
   data,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Set an cache entry</p></div>
</dd>
</dl>
</dd>
</dl>