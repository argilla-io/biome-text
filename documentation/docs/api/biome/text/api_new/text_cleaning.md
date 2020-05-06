# biome.text.api_new.text_cleaning <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.text_cleaning.TextCleaning">TextCleaning <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TextCleaning</span> ()</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Base class for text cleaning processors</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.api_new.text_cleaning.TextCleaning.default_implementation"><code class="name">var <span class="ident">default_implementation</span> : str</code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.text_cleaning.TextCleaningRule">TextCleaningRule <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TextCleaningRule</span> (func: Callable[[str], str])</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Registers a function as a rule for the default text cleaning implementation</p>
<p>Use the decorator <code>@TextCleaningRule</code> for creating custom text cleaning and pre-processing rules.</p>
<p>An example function to strip spaces (already included in the default <code><a title="biome.text.api_new.text_cleaning.TextCleaning" href="#biome.text.api_new.text_cleaning.TextCleaning">TextCleaning</a></code> processor):</p>
<pre><code class="python">@TextCleaningRule
def strip_spaces(text: str) -&gt; str:
    return text.strip()
</code></pre>
<h1 id="parameters">Parameters</h1>
<pre><code>func: &lt;code&gt;Callable\[\[str]&lt;/code&gt;
    The function to register
</code></pre></div>
<dl>
<h3 id="biome.text.api_new.text_cleaning.TextCleaningRule.registered_rules">registered_rules <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">registered_rules</span></span>(<span>) -> Dict[str, Callable[[str], str]]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Registered rules dictionary</p></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.text_cleaning.DefaultTextCleaning">DefaultTextCleaning <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">DefaultTextCleaning</span> (rules: List[str] = None)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Defines rules that can be applied to the text before it gets tokenized.</p>
<p>Each rule is a simple python function that receives and returns a <code>str</code>.</p>
<h1 id="parameters">Parameters</h1>
<pre><code>rules: &lt;code&gt;List\[str]&lt;/code&gt;
    A list of registered rule method names to be applied to text inputs
</code></pre></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
</dd>
</dl>