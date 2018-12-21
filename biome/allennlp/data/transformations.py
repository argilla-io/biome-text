from typing import Dict

from biome.spec import ModelForward
from biome.spec.data_source import DataSource
from biome.spec.utils import to_biome_class

__RECOGNAI_LOCATION = 'recogn.ai/location'
__RECOGNAI_FILES = 'recogn.ai/files'

__DS_LOCATION = 'location'
__DS_PATH = 'path'

def biome_datasource_spec_to_dataset_config(dataset_config: Dict) -> Dict:
    settings = dataset_config.get('settings', {})

    datasource = to_biome_class(data=dataset_config.get('datasource', {}), klass=DataSource)
    model_connect = to_biome_class(data=settings.get('forward', {}), klass=ModelForward)

    if datasource.type == 'FileSystem':
        params = datasource.params
        format = datasource.format
        format_params = format.params
        delimiter = format_params.pop('delimiter', None)
        config = dict(
            path=__extrat_ds_path(params),
            format=format.type,
            encoding=format.charset,
            transformations=model_connect,
            **format_params # allow pass through format params to dataset reader
        )
        return config
    else:
        # TODO include another datasource types
        return dataset_config


def __extrat_ds_path(params):
    # TODO Define a api constant
    ds_location = params.get(__DS_LOCATION, params.get(__RECOGNAI_LOCATION))
    ds_files = params.get(__DS_PATH, params.get(__RECOGNAI_FILES))
    if ds_files and len(ds_files) == 1:
        return ds_files[0]
    return '' if not ds_location \
        else '{0}*'.format(ds_location) if ds_location.endswith('/') \
        else "{0}/*".format(ds_location)
