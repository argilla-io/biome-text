# biome.text.api_new.data.utils <Badge text="Module"/>
<dl>
<h3 id="biome.text.api_new.data.utils.get_nested_property_from_data">get_nested_property_from_data <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">get_nested_property_from_data</span> (</span>
   data: Dict,
   property_key: str,
)  -> Union[Any, NoneType]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Search an deep property key in a data dictionary.</p>
<p>For example, having the data dictionary {"a": {"b": "the value"}}, the call</p>
<blockquote>
<blockquote>
<p>self.get_nested_property_from_data( {"a": {"b": "the value"}}, "a.b")</p>
</blockquote>
</blockquote>
<p>is equivalent to:</p>
<blockquote>
<blockquote>
<p>data["a"]["b"]</p>
</blockquote>
</blockquote>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>data</code></strong></dt>
<dd>The data dictionary</dd>
<dt><strong><code>property_key</code></strong></dt>
<dd>The (deep) property key</dd>
</dl>
<h2 id="returns">Returns</h2>
<pre><code>The property value if found, None otherwise
</code></pre></div>
</dd>
<h3 id="biome.text.api_new.data.utils.configure_dask_cluster">configure_dask_cluster <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">configure_dask_cluster</span> (</span>
   address: str = 'local',
   n_workers: int = 1,
   worker_memory: Union[str, int] = '1GB',
)  -> Union[distributed.client.Client, NoneType]
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a dask client (with a LocalCluster if needed)</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>address</code></strong></dt>
<dd>The cluster address. If "local" try to connect to a local cluster listening the 8786 port.
If no cluster listening, creates a new LocalCluster</dd>
<dt><strong><code>n_workers</code></strong></dt>
<dd>The number of cluster workers (only a new "local" cluster generation)</dd>
<dt><strong><code>worker_memory</code></strong></dt>
<dd>The memory reserved for local workers</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>A new dask Client</code></dt>
<dd>&nbsp;</dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.utils.close_dask_client">close_dask_client <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">close_dask_client</span></span>(<span>)</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.api_new.data.utils.extension_from_path">extension_from_path <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">extension_from_path</span></span>(<span>path: Union[str, List[str]]) -> str</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Helper method to get file extension</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>path</code></strong></dt>
<dd>A string or a list of strings.
If it is a list, the first entry is taken.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>extension</code></dt>
<dd>File extension</dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.utils.make_paths_relative">make_paths_relative <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">make_paths_relative</span> (</span>
   yaml_dirname: str,
   cfg_dict: Dict,
   path_keys: List[str] = None,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Helper method to convert file system paths relative to the yaml config file,
to paths relative to the current path.</p>
<p>It will recursively cycle through <code>cfg_dict</code> if it is nested.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>yaml_dirname</code></strong></dt>
<dd>Dirname to the yaml config file (as obtained by <code>os.path.dirname</code>.</dd>
<dt><strong><code>cfg_dict</code></strong></dt>
<dd>The config dictionary extracted from the yaml file.</dd>
<dt><strong><code>path_keys</code></strong></dt>
<dd>If not None, it will only try to modify the <code>cfg_dict</code> values corresponding to the <code>path_keys</code>.</dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.utils.is_relative_file_system_path">is_relative_file_system_path <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">is_relative_file_system_path</span></span>(<span>string: str) -> bool</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Helper method to check if a string is a relative file system path.</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>string</code></strong></dt>
<dd>The string to be checked.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>bool</code></dt>
<dd>Whether the string is a relative file system path or not.
If string is not type(str), return False.</dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.utils.flatten_dask_dataframe">flatten_dask_dataframe <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">flatten_dask_dataframe</span></span>(<span>data_frame: dask.dataframe.core.DataFrame) -> dask.dataframe.core.DataFrame</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Flatten an dataframe adding nested values as new columns
and dropping the old ones
Parameters</p>
<hr>
<dl>
<dt><strong><code>data_frame</code></strong></dt>
<dd>The original dask DataFrame</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>A new Dataframe with flatten content</code></dt>
<dd>&nbsp;</dd>
</dl></div>
</dd>
<h3 id="biome.text.api_new.data.utils.flatten_dataframe">flatten_dataframe <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">flatten_dataframe</span></span>(<span>data_frame: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame</span>
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"></div>
</dd>
<h3 id="biome.text.api_new.data.utils.save_dict_as_yaml">save_dict_as_yaml <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">save_dict_as_yaml</span> (</span>
   dictionary: dict,
   path: str,
   create_dirs: bool = True,
)  -> str
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Save a cfg dict to path as yaml</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>dictionary</code></strong></dt>
<dd>Dictionary to be saved</dd>
<dt><strong><code>path</code></strong></dt>
<dd>Filesystem location where the yaml file will be saved</dd>
<dt><strong><code>create_dirs</code></strong></dt>
<dd>If true, create directories in path.
If false, throw exception if directories in path do not exist.</dd>
</dl>
<h2 id="returns">Returns</h2>
<dl>
<dt><code>path</code></dt>
<dd>Location of the yaml file</dd>
</dl></div>
</dd>
</dl>