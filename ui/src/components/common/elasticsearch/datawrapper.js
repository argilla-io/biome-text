function confusionMatrixFilterFromAggregation(matrixAggregations) {
  if (!matrixAggregations) {
    return undefined;
  }
  return {
    name: 'Confusion matrix',
    id: 'confusionMatrix',
    type: 'confusion-matrix',
    values: matrixAggregations.buckets.flatMap(goldBucket => goldBucket.predicted.buckets.map(subBck => ({
      predicted: subBck.key,
      gold: goldBucket.key,
      count: subBck.doc_count,
      id: subBck.key + goldBucket.key,
    }))),
  };
}

function termFilterFromAggregation(name, aggregation) {
  if (!aggregation) {
    return undefined;
  }
  return {
    name,
    id: name,
    type: 'dropdown',
    values: aggregation.buckets.map(bck => ({
      name: bck.key,
      id: bck.key,
      count: bck.doc_count,
    })),
  };
}

function confidenceFilterFromAggregation(confidenceAggregation) {
  if (!confidenceAggregation) {
    return undefined;
  }
  const data = confidenceAggregation.buckets.map(bucket => ({
    key: bucket.key,
    count: bucket.doc_count,
  }));
  return {
    name: 'Confidence',
    id: 'confidence',
    type: 'confidence-range',
    config: {
      confidence: [0.0, 1.0],
      selected: false,
    },
    values: data,
  };
}

function getBaseMetrics() {
  return {
    fp: 0,
    tp: 0,
    fn: 0,
  };
}

function calculateMetrics(metrics) {
  const sanitizeIsNan = (n) => {
    if (Number.isNaN(n)) {
      return 0;
    } return n;
  };
  const getMetricForClass = (key, metric) => {
    const metricClass = {};
    metricClass[key] = sanitizeIsNan(metric).toFixed(2);
    return metricClass;
  };

  const metricsPerClass = {
    Precision: [],
    Recall: [],
    F1: [],
  };
  Object.keys(metrics).forEach((key) => {
    const metric = metrics[key];
    const precision = (metric.tp / (metric.tp + metric.fp)) * 100;
    const recall = (metric.tp / (metric.tp + metric.fn)) * 100;
    const f1 = 2 * ((precision * recall) / (precision + recall));
    metricsPerClass.Precision.push(getMetricForClass(key, precision));
    metricsPerClass.Recall.push(getMetricForClass(key, recall));
    metricsPerClass.F1.push(getMetricForClass(key, f1));
  });
  return metricsPerClass;
}

export default {
  mapESDocument2Record(document, inputs, outputField) {
    function getNestedValue(object, dottedKey) {
      if (object === undefined) {
        return object;
      }
      if (object[dottedKey] !== undefined) {
        return object[dottedKey];
      }
      const [head, ...tail] = dottedKey.split('.');
      return getNestedValue(object[head], tail.join('.'));
    }
    const { _id } = document;
    // eslint-disable-next-line no-param-reassign, no-underscore-dangle
    document = document._source;

    const prediction = document.prediction || document.annotation || {
      classes: [],
      max_class_prob: 0.0,
    };

    const doc = {
      _id,
      gold: document[outputField || 'label'],
      inputs: (inputs || []).reduce((acc, k) => {
        acc[k] = getNestedValue(document, k);
        return acc;
      }, {}),
      predicted: prediction.label || prediction.max_class,
      confidence: prediction.prob || prediction.max_class_prob,
      classes: prediction.classes,
      activerecord: true,
      _source: Object.keys(document).reduce((acc, key) => {
        // Filtering non searchable values
        // TODO think about the nested values (maybe flatten ???)
        if (typeof (document[key]) !== 'object') {
          acc[key] = document[key];
        }
        return acc;
      }, {}),
      feedback: document['biome.feedback'],
    };

    doc.interpretations = document.interpretations;

    if (prediction.explain) {
      doc.interpretations = Object.keys(prediction.explain).reduce((acc, k) => ({
        ...acc,
        [k]: prediction.explain[k].map((element) => {
          if (Array.isArray(element)) {
            return element.map((tokenInfo) => {
              const { token, attribution } = tokenInfo;
              return { token, attribution, grad: attribution };
            });
          }
          const { token, attribution } = element;
          return { token, attribution, grad: attribution };
        }),
      }), {});
    }
    doc.metadata = doc._source;
    if (document.metadata !== undefined) {
      doc.metadata = Object.keys(document.metadata || {}).reduce((acc, e) => ({ ...acc, [`metadata.${e}`]: document.metadata[e] }), {});
    }
    doc.predictedOk = doc.gold === doc.predicted;

    return doc;
  },

  filtersFromAggregations(aggregations) {
    return [
      termFilterFromAggregation('gold', aggregations.gold),
      termFilterFromAggregation('predicted', aggregations.predicted),
      termFilterFromAggregation('feedbackStatus', aggregations.feedbackStatus),
      confidenceFilterFromAggregation(aggregations.confidence),
      confusionMatrixFilterFromAggregation(aggregations.confusion_matrix),
    ]
      .filter(filter => filter !== undefined)
      .reduce((filtersMap, filter) => ({ ...filtersMap, [filter.id]: filter }), {});
  },

  metricsFromAggregation(aggregations) {
    const matrixAggregations = aggregations.confusion_matrix;
    if (!matrixAggregations) {
      return undefined;
    }
    const metrics = {};
    matrixAggregations.buckets.forEach((goldBucket) => {
      const goldValue = goldBucket.key;
      if (!metrics[goldValue]) {
        metrics[goldValue] = getBaseMetrics();
      }
      goldBucket.predicted.buckets.forEach((subBck) => {
        const predictedValue = subBck.key;
        const recordsCount = subBck.doc_count;
        if (!metrics[predictedValue]) {
          metrics[predictedValue] = getBaseMetrics();
        }
        if (goldValue === predictedValue) {
          metrics[goldValue].tp += recordsCount;
        }
        if (goldValue !== predictedValue) {
          metrics[goldValue].fn += recordsCount;
          metrics[predictedValue].fp += recordsCount;
        }
      });
    });

    return calculateMetrics(metrics);
  },

  feedbackMetricsFromAggregation(aggregations) {
    const metrics = {};
    if (aggregations.feedback) {
      aggregations.feedback.buckets.forEach((bucket) => {
        metrics[bucket.key] = bucket.doc_count;
      });
    }
    return metrics;
  },
};
