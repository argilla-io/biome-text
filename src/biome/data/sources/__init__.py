from .datasource import DataSource
from .example_preparator import SOURCE_FIELD, RESERVED_FIELD_PREFIX
from .readers import (
    XlsDataSourceReader,
    CsvDataSourceReader,
    JsonDataSourceReader,
    ElasticsearchDataSourceReader,
)
from .utils import ID, RESOURCE

# Define common data source readers in new form
# TODO unify duplicated format keys
DataSource.add_supported_format("xls", XlsDataSourceReader())
DataSource.add_supported_format("xlsx", XlsDataSourceReader())
DataSource.add_supported_format("csv", CsvDataSourceReader())
DataSource.add_supported_format("json", JsonDataSourceReader())
DataSource.add_supported_format("jsonl", JsonDataSourceReader())
DataSource.add_supported_format("json-l", JsonDataSourceReader())
DataSource.add_supported_format("elastic", ElasticsearchDataSourceReader())
DataSource.add_supported_format("elasticsearch", ElasticsearchDataSourceReader())
DataSource.add_supported_format("es", ElasticsearchDataSourceReader())
