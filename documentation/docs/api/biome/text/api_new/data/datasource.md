# biome.text.api_new.data.datasource <Badge text="Module"/>
<dl>
<h2 id="biome.text.api_new.data.datasource.DataSource">DataSource <Badge text="Class"/></h2>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
    <code>
<span class="token keyword">class</span> <span class="ident">DataSource</span> (</span>
    <span>source: Union[str, List[str], NoneType] = None</span><span>,</span>
    <span>attributes: Union[Dict[str, Any], NoneType] = None</span><span>,</span>
    <span>mapping: Union[Dict[str, Union[List[str], str]], NoneType] = None</span><span>,</span>
    <span>format: Union[str, NoneType] = None</span><span>,</span>
    <span>**kwargs</span><span>,</span>
<span>)</span>
    </code></pre></div>
</dt>
<dd>
<div class="desc"><p>This class takes care of reading the data source, usually specified in a yaml file.</p>
<p>It uses the <em>source readers</em> to extract a dask DataFrame.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>source</code></strong></dt>
<dd>The data source. Could be a list of filesystem path, or a key name indicating the source backend (elasticsearch)</dd>
<dt><strong><code>attributes</code></strong></dt>
<dd>Attributes needed for extract data from source</dd>
<dt><strong><code>format</code></strong></dt>
<dd>The data format. Optional. If found, overwrite the format extracted from source.
Supported formats are listed as keys in the <code>SUPPORTED_FORMATS</code> dict of this class.</dd>
<dt><strong><code>mapping</code></strong></dt>
<dd>Used to map the features (columns) of the data source
to the parameters of the DataSourceReader's <code>text_to_instance</code> method.</dd>
<dt><strong><code>kwargs</code></strong></dt>
<dd>Additional kwargs are passed on to the <em>source readers</em> that depend on the format.
@Deprecated. Use <code>attributes</code> instead</dd>
</dl></div>
<h3>Class variables</h3>
<dl>
<dt id="biome.text.api_new.data.datasource.DataSource.SUPPORTED_FORMATS"><code class="name">var <span class="ident">SUPPORTED_FORMATS</span></code></dt>
<dd>
<div class="desc"></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.data.datasource.DataSource.add_supported_format">add_supported_format <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">add_supported_format</span> (</span>
   format_key: str,
   parser: Callable,
   default_params: Dict[str, Any] = None,
)  -> NoneType
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Add a new format and reader to the data source readers.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>format_key</code></strong></dt>
<dd>The new format key</dd>
<dt><strong><code>parser</code></strong></dt>
<dd>The parser function</dd>
<dt><strong><code>default_params</code></strong></dt>
<dd>Default parameters for the parser function</dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.datasource.DataSource.from_yaml">from_yaml <Badge text="Static method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">from_yaml</span> (</span>
   file_path: str,
   default_mapping: Dict[str, str] = None,
)  -> <a title="biome.text.api_new.data.datasource.DataSource" href="#biome.text.api_new.data.datasource.DataSource">DataSource</a>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Create a data source from a yaml file.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>file_path</code></strong></dt>
<dd>The path to the yaml file.</dd>
<dt><strong><code>default_mapping</code></strong></dt>
<dd>A mapping configuration when no defined in yaml file</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>cls</code></dt>
<dd>&nbsp;</dd>
</dl></div>
</dd>
</dl>
<dl>
<h3 id="biome.text.api_new.data.datasource.DataSource.to_dataframe">to_dataframe <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_dataframe</span></span>(<span>self) -> dask.dataframe.core.DataFrame</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Returns the underlying DataFrame of the data source</p></div>
</dd>
<h3 id="biome.text.api_new.data.datasource.DataSource.to_mapped_dataframe">to_mapped_dataframe <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_mapped_dataframe</span></span>(<span>self) -> dask.dataframe.core.DataFrame</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>The columns of this DataFrame are named after the mapping keys, which in turn should match
the parameter names in the DatasetReader's <code>text_to_instance</code> method.
The content of these columns is specified in the mapping dictionary.</p>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>mapped_dataframe</code></dt>
<dd>Contains columns corresponding to the parameter names of the DatasetReader's <code>text_to_instance</code> method.</dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.datasource.DataSource.to_yaml">to_yaml <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">to_yaml</span> (</span>
   self,
   path: str,
   make_source_path_absolute: bool = False,
)  -> str
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Create a yaml config file for this data source.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>Path to the yaml file to be written.</dd>
<dt><strong><code>make_source_path_absolute</code></strong></dt>
<dd>If true, writes the source of the DataSource as an absolute path.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>path</code></dt>
<dd>&nbsp;</dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.datasource.DataSource.head">head <Badge text="Method"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">head</span> (</span>
   self,
   n: int = 10,
)  -> 'pandas.DataFrame'
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Allows for a peek into the data source showing the first n rows.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>n</code></strong></dt>
<dd>Number of lines</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>df</code></dt>
<dd>The first n lines as a <code>pandas.DataFrame</code></dd>
</dl></div>
</dd>
</dl>
</dd>
</dl>