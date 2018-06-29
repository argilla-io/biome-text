from typing import Dict


def is_biome_datasource_spec(dataset_config: Dict) -> bool:
    return 'datasource' in dataset_config and 'settings' in dataset_config


def biome_datasource_spec_to_dataset_config(dataset_config) -> Dict:
    datasource = dataset_config.pop('datasource', {})
    settings = dataset_config.pop('settings', {})

    if datasource['type'] == 'FileSystem':
        params = datasource.pop('params', {})
        format = datasource.pop('format', {})
        format_params = format.pop('params', {})
        delimiter = format_params.pop('delimiter', None)
        config = {
            'path': __extrat_ds_path(params),
            'format': format['type'],
            'encoding': format['charset'],
            'transformations': settings
        }
        if delimiter:
            config['sep'] = delimiter
        return config
    else:
        # TODO include another datasource types
        return {}


def __extrat_ds_path(params):
    ds_location = params.pop('recogn.ai/location')
    ds_files = params.pop('recogn.ai/files')
    if ds_files and len(ds_files) == 1:
        return ds_files[0]
    return '' if not ds_location \
        else '{0}*'.format(ds_location) if ds_location.endswith('/') \
        else "{0}/*".format(ds_location)
