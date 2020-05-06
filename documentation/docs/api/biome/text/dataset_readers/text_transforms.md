# biome.text.dataset_readers.text_transforms <Badge text="Module"/>
<dl>
<h2 id="biome.text.dataset_readers.text_transforms.TextTransforms">TextTransforms <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">TextTransforms</span> (rules: List[str] = None)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This class defines some rules that can be applied to the text before it gets embedded in a <code>TextField</code>.</p>
<p>Each rule is a simple python class method that receives and returns a str.
It will be applied when an instance of this class is called.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>rules</code></strong></dt>
<dd>A list of class method names to be applied on calling the instance.</dd>
</dl>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>DEFAULT_RULES</code></strong></dt>
<dd>The default rules if the <code>rules</code> parameter is not provided.</dd>
</dl></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.dataset_readers.text_transforms.RmSpacesTransforms" href="#biome.text.dataset_readers.text_transforms.RmSpacesTransforms">RmSpacesTransforms</a></li>
</ul>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.dataset_readers.text_transforms.TextTransforms.default_implementation"><code class="name">var <span class="ident">default_implementation</span> : str</code></dt>
<dd>
<div class="desc"></div>
</dd>
<dt id="biome.text.dataset_readers.text_transforms.TextTransforms.DEFAULT_RULES"><code class="name">var <span class="ident">DEFAULT_RULES</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.dataset_readers.text_transforms.RmSpacesTransforms">RmSpacesTransforms <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">RmSpacesTransforms</span> (rules: List[str] = None)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This class defines some rules that can be applied to the text before it gets embedded in a <code>TextField</code>.</p>
<p>Each rule is a simple python class method that receives and returns a str.
It will be applied when an instance of this class is called.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>rules</code></strong></dt>
<dd>A list of class method names to be applied on calling the instance.</dd>
</dl>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>DEFAULT_RULES</code></strong></dt>
<dd>The default rules if the <code>rules</code> parameter is not provided.</dd>
</dl></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.dataset_readers.text_transforms.TextTransforms" href="#biome.text.dataset_readers.text_transforms.TextTransforms">TextTransforms</a></li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.dataset_readers.text_transforms.Html2TextTransforms" href="#biome.text.dataset_readers.text_transforms.Html2TextTransforms">Html2TextTransforms</a></li>
</ul>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.dataset_readers.text_transforms.RmSpacesTransforms.DEFAULT_RULES"><code class="name">var <span class="ident">DEFAULT_RULES</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.dataset_readers.text_transforms.RmSpacesTransforms.strip_spaces">strip_spaces <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">strip_spaces</span></span>(<span>text: str) -> str</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Strip leading and trailing spaces/new lines</p></div>
</dd>
<h3 id="biome.text.dataset_readers.text_transforms.RmSpacesTransforms.rm_useless_spaces">rm_useless_spaces <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">rm_useless_spaces</span></span>(<span>text: str) -> str</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Remove multiple spaces in <code>text</code></p></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.dataset_readers.text_transforms.Html2TextTransforms">Html2TextTransforms <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">Html2TextTransforms</span> (rules: List[str] = None)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This class defines some rules that can be applied to the text before it gets embedded in a <code>TextField</code>.</p>
<p>Each rule is a simple python class method that receives and returns a str.
It will be applied when an instance of this class is called.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>rules</code></strong></dt>
<dd>A list of class method names to be applied on calling the instance.</dd>
</dl>
<h2 id="attributes">Attributes</h2>
<dl>
<dt><strong><code>DEFAULT_RULES</code></strong></dt>
<dd>The default rules if the <code>rules</code> parameter is not provided.</dd>
</dl></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.dataset_readers.text_transforms.RmSpacesTransforms" href="#biome.text.dataset_readers.text_transforms.RmSpacesTransforms">RmSpacesTransforms</a></li>
<li><a title="biome.text.dataset_readers.text_transforms.TextTransforms" href="#biome.text.dataset_readers.text_transforms.TextTransforms">TextTransforms</a></li>
<li>allennlp.common.registrable.Registrable</li>
<li>allennlp.common.from_params.FromParams</li>
</ul>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.dataset_readers.text_transforms.Html2TextTransforms.DEFAULT_RULES"><code class="name">var <span class="ident">DEFAULT_RULES</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.dataset_readers.text_transforms.Html2TextTransforms.fix_html">fix_html <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">fix_html</span></span>(<span>text: str) -> str</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>list of replacements in html code.
I leave a link to the fastai version here as a reference:
<a href="https://docs.fast.ai/text.transform.html#fix_html">https://docs.fast.ai/text.transform.html#fix_html</a></p></div>
</dd>
<h3 id="biome.text.dataset_readers.text_transforms.Html2TextTransforms.html_to_text">html_to_text <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">html_to_text</span></span>(<span>text: str) -> str</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Extract text from a html doc with BeautifulSoup4</p></div>
</dd>
</dl>
<h3>Inherited members</h3>
<ul class="hlist">
<li><code><b><a title="biome.text.dataset_readers.text_transforms.RmSpacesTransforms" href="#biome.text.dataset_readers.text_transforms.RmSpacesTransforms">RmSpacesTransforms</a></b></code>:
<ul class="hlist">
<li><code><a title="biome.text.dataset_readers.text_transforms.RmSpacesTransforms.rm_useless_spaces" href="#biome.text.dataset_readers.text_transforms.RmSpacesTransforms.rm_useless_spaces">rm_useless_spaces</a></code></li>
<li><code><a title="biome.text.dataset_readers.text_transforms.RmSpacesTransforms.strip_spaces" href="#biome.text.dataset_readers.text_transforms.RmSpacesTransforms.strip_spaces">strip_spaces</a></code></li>
</ul>
</li>
</ul>
</dd>
</dl>