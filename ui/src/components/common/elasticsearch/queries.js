/* eslint-disable no-shadow */
const defaultMappingConfig = {
  predicted: 'prediction.max_class.keyword',
  gold: 'label.keyword',
  confidence: 'prediction.max_class_prob',
  feedbackStatus: 'biome.feedback.status.keyword',
};

function defaultConfiguration(mappingConfig, enableGold, enableFeedback) {
  function goldAggregation() {
    if (!enableGold) {
      return {};
    }
    return {
      gold: {
        terms: {
          field: mappingConfig.gold,
          size: 100,
          order: {
            _count: 'desc',
          },
        },
      },
    };
  }
  function confusionMatrixAggregation(enableGold) {
    if (!enableGold) {
      return {};
    }
    return {
      confusion_matrix: {
        terms: {
          field: mappingConfig.gold,
          size: 100,
        },
        aggs: {
          predicted: {
            terms: {
              field: mappingConfig.predicted,
              size: 100,
            },
          },
        },
      },
    };
  }

  function predictedAggregation() {
    return {
      predicted: {
        terms: {
          field: mappingConfig.predicted,
          size: 100,
          order: {
            _count: 'desc',
          },
        },
      },
    };
  }
  function feedbackAggregation() {
    if (!enableFeedback) {
      return {};
    }
    return {
      feedbackStatus: {
        terms: {
          field: mappingConfig.feedbackStatus,
          size: 100,
          order: {
            _count: 'desc',
          },
        },
      },
    };
  }

  function confidenceAggregation() {
    const ranges = (from, to, interval) => {
      function range(from, to, step) {
        // eslint-disable-next-line no-bitwise
        const range = Array(~~((to - from) / step) + 1) // '~~' is Alternative for Math.floor()
          .fill()
          .map((v, i) => from + i * step);
        range.splice(-1, 1);
        return range;
      }

      function zip(rows) {
        return rows[0].map((_, c) => rows.map(row => row[c]));
      }

      const calculatedRanges = zip([
        range(from, to, interval),
        range(from + interval, to + interval, interval),
      ]).map(range => ({
        from: range[0],
        to: range[1],
      }));

      // calculatedRanges.push({ from: to - interval });
      return calculatedRanges;
    };

    return {
      confidence: {
        range: {
          field: mappingConfig.confidence,
          ranges: ranges(0.0, 1.0, 0.05),
        },
      },
    };
  }


  return {
    _source: true,
    queryAggs: {
      ...goldAggregation(enableGold),
      ...confusionMatrixAggregation(enableGold),
      ...predictedAggregation(),
      ...feedbackAggregation(enableFeedback),
      ...confidenceAggregation(),
    },
  };
}

function getFilterRange(filtersStatus = {}, searchFieldsMap = {}) {
  const confidence = filtersStatus.confidence[0];
  const min = confidence[0] * 0.01;
  const max = confidence[1] * 0.01;
  const filterRange = { range: {} };
  filterRange.range[searchFieldsMap.confidence] = {
    gte: min,
    lte: max,
  };
  return filterRange;
}

function getFiltermTerm(id, filtersStatus = {}, searchFieldsMap = {}) {
  let termId = searchFieldsMap[id];
  let filterValue = filtersStatus[id];
  if (!termId) {
    termId = id;
    [filterValue] = filterValue;
  }
  const termFilter = {};
  if (termId.endsWith('.keyword')) {
    termFilter[termId] = filterValue;
    return {
      terms: termFilter,
    };
  }
  termFilter[termId] = filterValue;
  return {
    match_phrase: termFilter,
  };
}

function configureSort(sortBy, sortOrder, mappingConfig) {
  const sort = [];

  if (sortBy !== undefined) {
    const sortByConfig = {};
    const sortByField = mappingConfig[sortBy];

    sortByConfig[sortByField] = { order: sortOrder };
    sort.push(sortByConfig);
  }

  sort.push({ _id: { order: 'asc' } });
  sort.push({ _score: { order: 'desc' } });

  return sort;
}

class ElasticManager {
  // eslint-disable-next-line class-methods-use-this
  feedbackResultsQuery() {
    // TODO: parameterize feedback status mapping
    return {
      size: 0,
      from: 0,
      aggs: {
        feedback: {
          terms: {
            field: defaultMappingConfig.feedbackStatus,
            size: 100,
            order: {
              _count: 'desc',
            },
          },
        },
      },
    };
  }

  toESQuery(
    {
      keyword,
      filtersStatus = {},
      esOptions = {},
      queryFields,
      sortOptions,
      hasGold = false,
      hasFeedback = false,
      showAll = false,
    },
  ) {
    const mappingConfig = { queryFields, ...defaultMappingConfig };
    const searchOptions = {
      filterTerms: Object.keys(filtersStatus).map((key) => {
        if (key === 'confidence') {
          return getFilterRange(filtersStatus, mappingConfig);
        }
        return getFiltermTerm(key, filtersStatus, mappingConfig);
      }),
      ...defaultConfiguration(mappingConfig, hasGold, hasFeedback),
      ...esOptions,
    };
    const query = {
      bool: {},
    };
    let queryMatch = {};
    if (keyword.length <= 0) {
      queryMatch = { match_all: {} };
    } else {
      queryMatch.multi_match = {
        fields: mappingConfig.queryFields,
        type: 'phrase_prefix',
        query: keyword,
        boost: 20,
      };
    }
    if (this.filterTermsEnabled(searchOptions)) {
      query.bool = {
        minimum_should_match: searchOptions.filterTerms.length,
        must: queryMatch,
        should: [searchOptions.filterTerms],
      };
    } else {
      query.bool.must = queryMatch;
    }
    if (!showAll) {
      if (!Object.keys(filtersStatus).includes('feedbackStatus')) {
        query.bool.must_not = {
          exists: {
            field: mappingConfig.feedbackStatus,
          },
        };
      }
    }
    const { sortBy, sortOrder } = sortOptions;
    return {
      query,
      // eslint-disable-next-line no-underscore-dangle
      _source: searchOptions._source,
      aggs: searchOptions.queryAggs,
      from: searchOptions.from,
      size: searchOptions.size,
      sort: configureSort(sortBy, sortOrder, mappingConfig),
    };
  }

  // eslint-disable-next-line class-methods-use-this
  filterTermsEnabled(searchOptions) {
    return searchOptions.filterTerms && searchOptions.filterTerms.length > 0;
  }
}
const manager = new ElasticManager();
export default manager;
