# biome.text.commands.ui.ui <Badge text="Module"/>
<dl>
<h3 id="biome.text.commands.ui.ui.launch_ui_from_args">launch_ui_from_args <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">launch_ui_from_args</span></span>(<span>args: argparse.Namespace) -> NoneType</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.commands.ui.ui.launch_ui">launch_ui <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">launch_ui</span> (</span>
   es_host: str,
   port: int = 9000,
)  -> NoneType
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.commands.ui.ui.temporal_static_path">temporal_static_path <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">temporal_static_path</span> (</span>
   explore_view: str,
   basedir: Union[str, NoneType] = None,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h2 id="biome.text.commands.ui.ui.BiomeUI">BiomeUI <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">BiomeUI</span> ()</span>
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
<h3 id="biome.text.commands.ui.ui.BiomeUI.add_subparser">add_subparser <Badge text="Method"/></h3>
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