# biome.text.tokenizer <Badge text="Module"/>
## Tokenizer <Badge text="Class"/>
<pre class="language-python">
            <code>
              <span class="token keyword">class</span> <span class="ident">Tokenizer</span> (</span>
                  <span>lang: str = 'en'</span><span>,</span>
                  <span>skip_empty_tokens: bool = False</span><span>,</span>
                  <span>max_sequence_length: int = None</span><span>,</span>
                  <span>max_nr_of_sentences: int = None</span><span>,</span>
                  <span>text_cleaning: Union[biome.text.text_cleaning.TextCleaning, NoneType] = None</span><span>,</span>
                  <span>segment_sentences: Union[bool, allennlp.data.tokenizers.sentence_splitter.SentenceSplitter] = False</span><span>,</span>
                  <span>start_tokens: Union[List[str], NoneType] = None</span><span>,</span>
                  <span>end_tokens: Union[List[str], NoneType] = None</span><span>,</span>
              <span>)</span>
            </code>
          </pre>
<p>Pre-processes and tokenizes input text</p>
<p>Transforms inputs (e.g., a text, a list of texts, etc.) into structures containing <code>allennlp.data.Token</code> objects.</p>
<p>Use its arguments to configure the first stage of the pipeline (i.e., pre-processing a given set of text inputs.)</p>
<p>Use methods for tokenizing depending on the shape of inputs (e.g., records with multiple fields, sentences lists).</p>
<h1 id="parameters">Parameters</h1>
<p>lang: <code>str</code>
The <code>spaCy</code> language to be used by the tokenizer (default is <code>en</code>)
skip_empty_tokens: <code>bool</code>
max_sequence_length: <code>int</code>
Maximum length in characters for input texts truncated with <code>[:max_sequence_length]</code> after <code>TextCleaning</code>.
max_nr_of_sentences: <code>int</code>
Maximum number of sentences to keep when using <code>segment_sentences</code> truncated with <code>[:max_sequence_length]</code>.
text_cleaning: <code>Optional[TextCleaning]</code>
A <code>TextCleaning</code> configuration with pre-processing rules for cleaning up and transforming raw input text.
segment_sentences:
<code>Union[bool, SentenceSplitter]</code>
Whether to segment input texts in to sentences using the default <code>SentenceSplitter</code> or a given splitter.
start_tokens: <code>Optional[List[str]]</code>
A list of token strings to the sequence before tokenized input text.
end_tokens: <code>Optional[List[str]]</code>
A list of token strings to the sequence after tokenized input text.</p>
<span style="white-space, word-break">
&#160;
&#xA0;
&NonBreakingSpace;
### Subclasses
</span>
<pre>


### Ancestors
</pre>
<ul class="hlist">
<li>allennlp.common.from_params.FromParams</li>
</ul>
<dl>
<pre>

### tokenize_text <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
          <code>
          <span class="token keyword">def</span> <span class="ident">tokenize_text</span> (</span>
            self,
            text: str,
          )  -> List[allennlp.data.tokenizers.token.Token]
          </code>
        </pre>
</div>
</dt>
<dd>
<p>Tokenizes a text string</p>
<p>Use this for the simplest case where your input is just a <code>str</code></p>
<h1 id="parameters">Parameters</h1>
<pre><code>text: &lt;code&gt;str&lt;/code&gt;
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>tokens: &lt;code&gt;List\[Token]&lt;/code&gt;
</code></pre>
</dd>
<pre>

### tokenize_document <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
          <code>
          <span class="token keyword">def</span> <span class="ident">tokenize_document</span> (</span>
            self,
            document: List[str],
          )  -> List[List[allennlp.data.tokenizers.token.Token]]
          </code>
        </pre>
</div>
</dt>
<dd>
<p>Tokenizes a document-like structure containing lists of text inputs</p>
<p>Use this to account for hierarchical text structures (e.g., a paragraph is a list of sentences)</p>
<h1 id="parameters">Parameters</h1>
<pre><code>document: &lt;code&gt;List\[str]&lt;/code&gt;
A &lt;code&gt;List&lt;/code&gt; with text inputs, e.g., sentences
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>tokens: &lt;code&gt;List\[List\[Token]]&lt;/code&gt;
</code></pre>
</dd>
<pre>

### tokenize_record <Badge text="Method"/>
</pre>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
          <code>
          <span class="token keyword">def</span> <span class="ident">tokenize_record</span> (</span>
            self,
            record: Dict[str, Any],
          )  -> Dict[str, Tuple[List[allennlp.data.tokenizers.token.Token], List[allennlp.data.tokenizers.token.Token]]]
          </code>
        </pre>
</div>
</dt>
<dd>
<p>Tokenizes a record-like structure containing text inputs</p>
<p>Use this to keep information about the record-like data structure as input features to the model.</p>
<h1 id="parameters">Parameters</h1>
<pre><code>record: &lt;code&gt;Dict\[str, Any]&lt;/code&gt;
A &lt;code&gt;Dict&lt;/code&gt; with arbitrary "fields" containing text.
</code></pre>
<h1 id="returns">Returns</h1>
<pre><code>tokens: &lt;code&gt;Dict\[str, Tuple\[List\[Token], List\[Token]]]&lt;/code&gt;
    A dictionary with two lists of &lt;code&gt;Token&lt;/code&gt;'s for each record entry: &lt;code&gt;key&lt;/code&gt; and &lt;code&gt;value&lt;/code&gt; tokens.
</code></pre>
</dd>
</dl>