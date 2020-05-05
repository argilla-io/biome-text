# biome.text.models.archival <Badge text="Module"/>
<dl>
<h3 id="biome.text.models.archival.load_archive">load_archive <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">load_archive</span> (</span>
   archive_file: str,
   cuda_device: int = -1,
   overrides: str = '',
   weights_file: str = None,
)  -> allennlp.models.archival.Archive
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.models.archival.to_local_archive">to_local_archive <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_local_archive</span></span>(<span>archive_file: str) -> str</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Wraps archive download to support remote locations (s3, hdfs,&hellip;)</p></div>
</dd>
</dl>