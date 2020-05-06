# biome.text.commands.serve.serve <Badge text="Module"/>
<dl>
<h3 id="biome.text.commands.serve.serve.serve">serve <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">serve</span> (</span>
   binary: str,
   port: int = 8000,
   output: str = None,
)  -> NoneType
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.commands.serve.serve.make_app">make_app <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">make_app</span> (</span>
   binary: str,
   output: str = None,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>This function allows to serve a model from a gunicorn server. For example:</p>
<blockquote>
<blockquote>
<blockquote>
<p>gunicorn 'biome.allennlp.commands.serve.serve:make_app("/path/to/model.tar.gz")'</p>
</blockquote>
</blockquote>
</blockquote>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>binary</code></strong></dt>
<dd>Path to the <em>model.tar.gz</em> file</dd>
<dt><strong><code>output</code></strong></dt>
<dd>Path to the output folder, in which to store the predictions.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>app</code></dt>
<dd>A Flask app used by gunicorn server</dd>
</dl></div>
</dd>
</dl>
<dl>
<h2 id="biome.text.commands.serve.serve.BiomeRestAPI">BiomeRestAPI <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">BiomeRestAPI</span> ()</span>
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
<h3 id="biome.text.commands.serve.serve.BiomeRestAPI.add_subparser">add_subparser <Badge text="Method"/></h3>
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