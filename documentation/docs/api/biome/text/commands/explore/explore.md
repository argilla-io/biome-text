# biome.text.commands.explore.explore <Badge text="Module"/>
<dl>
<h3 id="biome.text.commands.explore.explore.explore_with_args">explore_with_args <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">explore_with_args</span></span>(<span>args: argparse.Namespace) -> NoneType</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.commands.explore.explore.explore">explore <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">explore</span> (</span>
   binary: str,
   source_path: str,
   es_host: str,
   es_index: str,
   batch_size: int = 500,
   prediction_cache_size: int = 0,
   interpret: bool = False,
   force_delete: bool = True,
   **prediction_metadata,
)  -> NoneType
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.commands.explore.explore.register_biome_prediction">register_biome_prediction <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">register_biome_prediction</span> (</span>
   name: str,
   created_index: str,
   es_hosts: str,
   pipeline: <a title="biome.text.pipelines.pipeline.Pipeline" href="../../pipelines/pipeline.html#biome.text.pipelines.pipeline.Pipeline">Pipeline</a>,
   **extra_args: dict,
)  -> NoneType
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h2 id="biome.text.commands.explore.explore.BiomeExplore">BiomeExplore <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">BiomeExplore</span> ()</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>An abstract class representing subcommands for allennlp.run.
If you wanted to (for example) create your own custom <code>special-evaluate</code> command to use like</p>
<p><code>allennlp special-evaluate ...</code></p>
<p>you would create a <code>Subcommand</code> subclass and then pass it as an override to
:func:<code>~allennlp.commands.main</code> .</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.commands.subcommand.Subcommand</li>
</ul>
<dl>
<h3 id="biome.text.commands.explore.explore.BiomeExplore.add_subparser">add_subparser <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">add_subparser</span> (</span>
   self,
   name: str,
   parser: argparse._SubParsersAction,
)  -> argparse.ArgumentParser
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