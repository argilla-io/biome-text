# biome.text.pipelines.explore <Badge text="Module"/>
<dl>
<h3 id="biome.text.pipelines.explore.pipeline_predictions">pipeline_predictions <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">pipeline_predictions</span> (</span>
   pipeline: <a title="biome.text.pipelines.pipeline.Pipeline" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline">Pipeline</a>,
   source_path: str,
   config: <a title="biome.text.pipelines.defs.ExploreConfig" href="defs.html#biome.text.pipelines.defs.ExploreConfig">ExploreConfig</a>,
   es_config: <a title="biome.text.pipelines.defs.ElasticsearchConfig" href="defs.html#biome.text.pipelines.defs.ElasticsearchConfig">ElasticsearchConfig</a>,
)  -> dask.dataframe.core.DataFrame
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Read a data source and tries to apply a model predictions to the whole data source. The
results will be persisted into an elasticsearch index for further data exploration</p></div>
</dd>
<h3 id="biome.text.pipelines.explore.register_biome_prediction">register_biome_prediction <Badge text="Function"/></h3>
<dt>
<div class="language-python extra-class">
<pre class="language-python">
<code>
<span class="token keyword">def</span> <span class="ident">register_biome_prediction</span> (</span>
   name: str,
   pipeline: <a title="biome.text.pipelines.pipeline.Pipeline" href="pipeline.html#biome.text.pipelines.pipeline.Pipeline">Pipeline</a>,
   es_config: <a title="biome.text.pipelines.defs.ElasticsearchConfig" href="defs.html#biome.text.pipelines.defs.ElasticsearchConfig">ElasticsearchConfig</a>,
   **kwargs,
) 
</code>
        </pre>
</div>
</dt>
<dd>
<div class="desc"><p>Creates a new metadata entry for the incoming prediction</p>
<h2 id="parameters">Parameters</h2>
<dl>
<dt><strong><code>name</code></strong></dt>
<dd>A descriptive prediction name</dd>
<dt><strong><code>pipeline</code></strong></dt>
<dd>The pipeline used for the prediction batch</dd>
<dt>es_config:</dt>
<dt>The Elasticsearch configuration data</dt>
<dt><strong><code>kwargs</code></strong></dt>
<dd>Extra arguments passed as extra metadata info</dd>
</dl></div>
</dd>
</dl>