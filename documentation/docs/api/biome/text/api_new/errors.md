# biome.text.api_new.errors <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.errors.BaseError">BaseError <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">BaseError</span> (...)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Base error. This class could include common error attributes or methods</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.errors.ValidationError" href="#biome.text.api_new.errors.ValidationError">ValidationError</a></li>
</ul>
</dd>
<h2 id="biome.text.api_new.errors.ValidationError">ValidationError <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">ValidationError</span> (...)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Base error for data validation</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.errors.BaseError" href="#biome.text.api_new.errors.BaseError">BaseError</a></li>
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.errors.MissingArgumentError" href="#biome.text.api_new.errors.MissingArgumentError">MissingArgumentError</a></li>
</ul>
</dd>
<h2 id="biome.text.api_new.errors.MissingArgumentError">MissingArgumentError <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">MissingArgumentError</span> (arg_name: str)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Error related with input params</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.errors.ValidationError" href="#biome.text.api_new.errors.ValidationError">ValidationError</a></li>
<li><a title="biome.text.api_new.errors.BaseError" href="#biome.text.api_new.errors.BaseError">BaseError</a></li>
<li>builtins.Exception</li>
<li>builtins.BaseException</li>
</ul>
</dd>
<h2 id="biome.text.api_new.errors.http_error_handling">http_error_handling <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">http_error_handling</span> ()</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Error handling for http error transcription</p></div>
</dd>
</dl>