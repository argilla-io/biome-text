# biome.text.commands.learn.learn <Badge text="Module"/>
<p>The <code>train</code> subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.</p>
<p>.. code-block:: bash</p>
<p>$ python -m allennlp.run train &ndash;help
usage: run [command] train [-h] -s SERIALIZATION_DIR param_path</p>
<p>Train the specified model on the specified dataset.</p>
<p>positional arguments:
param_path
path to parameter file describing the model to be trained</p>
<p>optional arguments:
-h, &ndash;help
show this help message and exit
-s SERIALIZATION_DIR, &ndash;serialization-dir SERIALIZATION_DIR
directory in which to save the model and its logs</p>
<dl>
<h3 id="biome.text.commands.learn.learn.learn_from_args">learn_from_args <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">learn_from_args</span></span>(<span>args: argparse.Namespace)</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Launches a pipeline learn action with input command line arguments</p></div>
</dd>
</dl>
<dl>
<h2 id="biome.text.commands.learn.learn.BiomeLearn">BiomeLearn <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">BiomeLearn</span> ()</span>
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
<h3 id="biome.text.commands.learn.learn.BiomeLearn.description">description <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">description</span></span>(<span>) -> str</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.commands.learn.learn.BiomeLearn.command_handler">command_handler <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">command_handler</span></span>(<span>) -> Callable</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.commands.learn.learn.BiomeLearn.add_subparser">add_subparser <Badge text="Method"/></h3>
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