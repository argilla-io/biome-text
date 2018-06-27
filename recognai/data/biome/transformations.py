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
            'path': "{0}*".format(params['recogn.ai/location']),
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
