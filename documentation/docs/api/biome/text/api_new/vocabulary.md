# biome.text.api_new.vocabulary <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.vocabulary.vocabulary">vocabulary <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">vocabulary</span> ()</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Manages vocabulary tasks and fetches vocabulary information</p>
<p>Provides utilities for getting information from a given vocabulary.</p>
<p>Provides management actions such as extending the labels, setting new labels or creating an "empty" vocab.</p></div>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.api_new.vocabulary.vocabulary.LABELS_NAMESPACE"><code class="name">var <span class="ident">LABELS_NAMESPACE</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.vocabulary.vocabulary.num_labels">num_labels <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">num_labels</span></span>(<span>vocab: allennlp.data.vocabulary.Vocabulary) -> int</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Gives the number of labels in the vocabulary</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>num_labels: &lt;code&gt;int&lt;/code&gt;
    The number of labels in the vocabulary
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.get_labels">get_labels <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_labels</span></span>(<span>vocab: allennlp.data.vocabulary.Vocabulary) -> List[str]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Gets list of labels in the vocabulary</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>labels: &lt;code&gt;List\[str]&lt;/code&gt;
    A list of label strings
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.label_for_index">label_for_index <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">label_for_index</span> (</span>
   vocab: allennlp.data.vocabulary.Vocabulary,
   idx: int,
)  -> str
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Gets label string for a label <code>int</code> id</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>label: &lt;code&gt;str&lt;/code&gt;
   The string for a label id
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.index_for_label">index_for_label <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">index_for_label</span> (</span>
   vocab: allennlp.data.vocabulary.Vocabulary,
   label: str,
)  -> int
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Gets the label <code>int</code> id for label string</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>label_idx: &lt;code&gt;int&lt;/code&gt;
    The label id for label string
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.get_index_to_labels_dictionary">get_index_to_labels_dictionary <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_index_to_labels_dictionary</span></span>(<span>vocab: allennlp.data.vocabulary.Vocabulary) -> Dict[int, str]</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Gets a dictionary for turning label <code>int</code> ids into label strings</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>labels: &lt;code&gt;Dict\[int, str]&lt;/code&gt;
    A dictionary to get fetch label strings from ids
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.vocab_size">vocab_size <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">vocab_size</span> (</span>
   vocab: allennlp.data.vocabulary.Vocabulary,
   namespace: str,
)  -> int
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Fetches the vocabulary size of a given namespace</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
namespace: &lt;code&gt;str&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>size: &lt;code&gt;int&lt;/code&gt;
    The vocabulary size for a given namespace
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.words_vocab_size">words_vocab_size <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">words_vocab_size</span></span>(<span>vocab: allennlp.data.vocabulary.Vocabulary) -> int</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Fetches the vocabulary size for the <code>words</code> namespace</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>size: &lt;code&gt;int&lt;/code&gt;
    The vocabulary size for the words namespace
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.extend_labels">extend_labels <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">extend_labels</span> (</span>
   vocab: allennlp.data.vocabulary.Vocabulary,
   labels: List[str],
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Adds a list of label strings to the vocabulary</p>
<p>Use this to add new labels to your vocabulary (e.g., useful for reusing the weights of an existing classifier)</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
labels: &lt;code&gt;List\[str]&lt;/code&gt;
    A list of strings containing the labels to add to an existing vocabulary
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.empty_vocab">empty_vocab <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">empty_vocab</span> (</span>
   featurizer: <a title="biome.text.api_new.featurizer.InputFeaturizer" href="featurizer.html#biome.text.api_new.featurizer.InputFeaturizer">InputFeaturizer</a>,
   labels: List[str] = None,
)  -> allennlp.data.vocabulary.Vocabulary
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Generates a "mock" empty vocabulary for a given <code>InputFeaturizer</code></p>
<p>This method generate a mock vocabulary for the featurized namespaces.
TODO: Clarify? &ndash;&gt; If default model use another tokens indexer key name, the pipeline model won't be loaded from configuration</p>
<h1 id="parameters">Parameters</h1>
<pre><code>featurizer: &lt;code&gt;InputFeaturizer&lt;/code&gt;
    A featurizer for which to create the vocabulary
labels: &lt;code&gt;List\[str]&lt;/code&gt;
    The label strings to add to the vocabulary
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>vocabulary: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
    The instantiated vocabulary
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.vocabulary.vocabulary.set_labels">set_labels <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">set_labels</span> (</span>
   vocab: allennlp.data.vocabulary.Vocabulary,
   new_labels: List[str],
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Resets the labels in the vocabulary with a given labels string list</p>
<h1 id="parameters">Parameters</h1>
<pre><code>vocab: &lt;code&gt;allennlp.data.Vocabulary&lt;/code&gt;
new_labels: &lt;code&gt;List\[str]&lt;/code&gt;
    The label strings to add to the vocabulary
</code></pre></div>
</dd>
</dl>
</dd>
</dl>