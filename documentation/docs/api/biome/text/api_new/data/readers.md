# biome.text.api_new.data.readers <Badge text="Module"/>
<dl>
<h3 id="biome.text.api_new.data.readers.from_csv">from_csv <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_csv</span> (</span>
   path: Union[str, List[str]],
   **params,
)  -> dask.dataframe.core.DataFrame
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a <code>dask.dataframe.DataFrame</code> from one or several csv files.
Includes a "path column".</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>Path to files</dd>
<dt><strong><code>params</code></strong></dt>
<dd>Extra arguments passed on to <code>dask.dataframe.read_csv</code></dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>df</code></dt>
<dd>A <code>dask.DataFrame</code></dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.readers.from_json">from_json <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_json</span> (</span>
   path: Union[str, List[str]],
   flatten: bool = True,
   **params,
)  -> dask.dataframe.core.DataFrame
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a <code>dask.dataframe.DataFrame</code> from one or several json files.
Includes a "path column".</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>Path to files</dd>
<dt><strong><code>flatten</code></strong></dt>
<dd>If true (default false), flatten json nested data</dd>
<dt><strong><code>params</code></strong></dt>
<dd>Extra arguments passed on to <code>pandas.read_json</code></dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>df</code></dt>
<dd>A <code>dask.DataFrame</code></dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.readers.from_parquet">from_parquet <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_parquet</span> (</span>
   path: Union[str, List[str]],
   **params,
)  -> dask.dataframe.core.DataFrame
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a <code>dask.dataframe.DataFrame</code> from one or several parquet files.
Includes a "path column".</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>Path to files</dd>
<dt><strong><code>params</code></strong></dt>
<dd>Extra arguments passed on to <code>pandas.read_parquet</code></dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>df</code></dt>
<dd>A <code>dask.dataframe.DataFrame</code></dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.readers.from_excel">from_excel <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_excel</span> (</span>
   path: Union[str, List[str]],
   **params,
)  -> dask.dataframe.core.DataFrame
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a <code>dask.dataframe.DataFrame</code> from one or several excel files.
Includes a "path column".</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>Path to files</dd>
<dt><strong><code>params</code></strong></dt>
<dd>Extra arguments passed on to <code>pandas.read_excel</code></dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>df</code></dt>
<dd>A <code>dask.dataframe.DataFrame</code></dd>
</dl></div>
</dd>
</dl>
<dl>
<h2 id="biome.text.api_new.data.readers.DataFrameReader">DataFrameReader <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">DataFrameReader</span> ()</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>A base class for read :class:dask.dataframe.DataFrame</p></div>
<h3>Subclasses</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.data.readers.ElasticsearchDataFrameReader" href="#biome.text.api_new.data.readers.ElasticsearchDataFrameReader">ElasticsearchDataFrameReader</a></li>
</ul>
<dl>
<h3 id="biome.text.api_new.data.readers.DataFrameReader.read">read <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">read</span> (</span>
   source: Union[str, List[str]],
   **kwargs,
)  -> dask.dataframe.core.DataFrame
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Base class method for read the DataSources as a :class:dask.dataframe.DataFrame</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt>source: The source information.</dt>
<dt><strong><code>kwargs</code></strong> :&ensp;<code>extra arguments passed to read method. Each reader should declare needed arguments</code></dt>
<dd>&nbsp;</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>A :class:dask.dataframe.DataFrame read from source</code></dt>
<dd>&nbsp;</dd>
</dl></div>
</dd>
</dl>
</dd>
<h2 id="biome.text.api_new.data.readers.ElasticsearchDataFrameReader">ElasticsearchDataFrameReader <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">ElasticsearchDataFrameReader</span> ()</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>Read a :class:dask.dataframe.DataFrame from a elasticsearch index</p></div>
<h3>Ancestors</h3>
<ul class="hlist">
<li><a title="biome.text.api_new.data.readers.DataFrameReader" href="#biome.text.api_new.data.readers.DataFrameReader">DataFrameReader</a></li>
</ul>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.api_new.data.readers.ElasticsearchDataFrameReader.SOURCE_TYPE"><code class="name">var <span class="ident">SOURCE_TYPE</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.data.readers.ElasticsearchDataFrameReader.read">read <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">read</span> (</span>
   source: Union[str, List[str]],
   index: str,
   doc_type: str = '_doc',
   query: Union[dict, NoneType] = None,
   es_host: str = 'http://localhost:9200',
   flatten_content: bool = False,
   **kwargs,
)  -> dask.dataframe.core.DataFrame
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a :class:dask.dataframe.DataFrame from a elasticsearch indexes</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>source</code></strong></dt>
<dd>The source param must match with :class:ElasticsearchDataFrameReader.SOURCE_TYPE</dd>
<dt><strong><code>es_host</code></strong></dt>
<dd>The elasticsearch host url (default to "http://localhost:9200")</dd>
<dt><strong><code>index</code></strong></dt>
<dd>The elasticsearch index</dd>
<dt><strong><code>doc_type</code></strong></dt>
<dd>The elasticsearch document type (default to "_doc")</dd>
<dt><strong><code>query</code></strong></dt>
<dd>The index query applied for extract the data</dd>
<dt><strong><code>flatten_content</code></strong></dt>
<dd>If True, applies a flatten to all nested data. It may take time to apply this flatten, so
is deactivate by default.</dd>
<dt><strong><code>kwargs</code></strong></dt>
<dd>Extra arguments passed to base search method</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>A :class:dask.dataframe.DataFrame with index query results</code></dt>
<dd>&nbsp;</dd>
</dl></div>
</dd>
</dl>
</dd>
</dl>