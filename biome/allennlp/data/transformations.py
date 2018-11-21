from typing import Dict

from biome.spec import ModelConnect
from biome.spec.data_source import DataSource
from biome.spec.utils import to_biome_class


def biome_datasource_spec_to_dataset_config(dataset_config: Dict) -> Dict:
    settings = dataset_config.get('settings', {})

    datasource = to_biome_class(data=dataset_config.get('datasource', {}), klass=DataSource)
    model_connect = to_biome_class(data=settings.get('modelConnect', {}), klass=ModelConnect)

    if datasource.type == 'FileSystem':
        params = datasource.params
        format = datasource.format
        format_params = format.params
        delimiter = format_params.get('delimiter', None)
        config = dict(
            path=__extrat_ds_path(params),
            format=format.type,
            encoding=format.charset,
            transformations=model_connect
        )
        if delimiter:
            config['sep'] = delimiter
        return config
    else:
        # TODO include another datasource types
        return {}


def __extrat_ds_path(params):
    # TODO Define a api constant
    ds_location = params.get('recogn.ai/location')
    ds_files = params.get('recogn.ai/files')
    if ds_files and len(ds_files) == 1:
        return ds_files[0]
    return '' if not ds_location \
        else '{0}*'.format(ds_location) if ds_location.endswith('/') \
        else "{0}/*".format(ds_location)
