import axios from 'axios';
import flatten from 'flat';
import defaultData from './mockData';

const publicPath = process.env.BASE_URL || '/';
const elasticsearch = axios.create({ baseURL: `${publicPath}elastic` });

let versionNumber = '7';
elasticsearch.get('/').then((response) => {
  [versionNumber] = response.data.version.number.split('.');
});

function mapPrediction2ProjectInfo(prediction) {
  const info = prediction._source;

  return {
    ...info,
    id: prediction._id,
    projectName: info.project || 'default',
    dataSource: info.datasource || 'unknown',
    model: info.model || 'none',
    kind: info.kind || 'explore',
    usePrediction: info.use_prediction,
    exploreName: (info.explore_name || info.name).replace(/_/g, ' '),
    metrics: {
      f1: NaN,
      recall: NaN,
      precision: NaN,
    },
    inputs: info.inputs || ['tokens'],
    createdAt: info.created_at,
  };
}

class ESClient {
  constructor(index) {
    this.index = index || '';
  }

  async search(query) {
    try {
      let esQuery = query;
      if (versionNumber === '7') {
        esQuery = { track_total_hits: true, ...query };
      }
      const results = await elasticsearch.post(`${this.index}/_search`, esQuery);
      const totalInfo = results.data.hits.total;
      results.data.hits.total = totalInfo.value === undefined ? totalInfo : totalInfo.value;
      return results.data;
    } catch (error) {
      return defaultData;
    }
  }

  async getTotalRecords() {
    try {
      const response = await this.search({ size: 0, query: { match_all: {} } });
      // ES 6.x and 7.x
      return response.hits.total.value || response.hits.total;
    } catch (error) {
      return 0;
    }
  }

  async feedback(id, feedbackStatus) {
    const feedback = (status) => {
      if (!status) {
        return null;
      }
      return {
        status,
        prediction: {
          max_class: status,
          max_class_prob: 1.0,
        },
      };
    };
    const type = '_doc';
    const data = {
      doc: {
        label: feedbackStatus,
        'biome.feedback': feedback(feedbackStatus),
      },
    };
    return elasticsearch.post(`${this.index}/${type}/${id}/_update`, data);
  }

  // eslint-disable-next-line class-methods-use-this
  static async fetchPredictions() {
    // TODO: This method should be filtered by project
    try {
      const results = await elasticsearch.post('.biome/_search', { size: 1000 });
      // eslint-disable-next-line no-underscore-dangle
      return results.data.hits.hits.map(mapPrediction2ProjectInfo);
    } catch (error) {
      console.warn(error);
      return [];
    }
  }

  // eslint-disable-next-line class-methods-use-this
  static async deletePrediction(predictionId) {
    const indexDeleted = await elasticsearch.delete(predictionId);
    const metadatEntryDeleted = await elasticsearch.delete(`.biome/_doc/${predictionId}`);
    return { ...indexDeleted.data, ...metadatEntryDeleted.data };
  }

  // eslint-disable-next-line class-methods-use-this
  static async fetchPrediction(predictionId) {
    try {
      const results = await elasticsearch.get(`.biome/_doc/${predictionId}`);
      const indexInfo = results.data._source;
      const response = await elasticsearch.get(`${predictionId}/_mappings`);
      // Check elasticsearch fersion for mapping extraction properly (_doc)
      let propMappings;
      if (versionNumber === '7') {
        propMappings = response.data[predictionId].mappings.properties;
      } else {
        propMappings = response.data[predictionId].mappings._doc.properties;
      }
      const props = flatten(propMappings);
      const toFieldNames = (arrayList, originalInputs) => arrayList.map(key => key.replace('.type', ''))
        // Input field is a primitive type ('tokens' or 'record1')
        // or a compound object ('tokens.firstName', or 'record1.lastName')
        .filter(key => originalInputs.find(input => key === input || key.startsWith(`${input}.`)))
        .map(name => name.replace(/\.properties/g, ''));

      const queryFields = Object.keys(props).filter(key => key.endsWith('.type') && props[key] === 'text');
      const modelInputs = Object.keys(props).filter(key => key.endsWith('.type') && props[key] !== 'keyword');

      indexInfo.searchableFields = toFieldNames(queryFields, indexInfo.inputs);
      indexInfo.readableFields = toFieldNames(modelInputs, indexInfo.inputs);

      return indexInfo;
    } catch (error) {
      console.warn(error);
      return {};
    }
  }
}

export default ESClient;
