'use strict';

const FACTORS = ['importance', 'sourceReliability', 'freshness', 'foundational', 'usage'];
const NON_CONTENT_WORDS = new Set(['a', 'an', 'the', 'what', 'who', 'when', 'where', 'which', 'why', 'how', 'is', 'are', 'was', 'were', 'do', 'does', 'did', 'can', 'to', 'of', 'in', 'on', 'at', 'for', 'with', 'from', 'and', 'or']);

function triggerWeight(text) {
  const folded = text.normalize('NFKC').toLocaleLowerCase();
  if (/[\u3400-\u9fff]/.test(folded)) return 1;
  const words = folded.match(/[a-z0-9+#.]+/g) || [];
  const contentWords = words.filter(word => !NON_CONTENT_WORDS.has(word));
  const count = contentWords.length || words.length;
  if (count <= 1) return 0.4;
  if (count === 2) return 0.55;
  return 1;
}

function suggestionsToKnowledgeItems(records, existingIds = []) {
  if (!Array.isArray(records)) throw new Error('promotion_records_must_be_an_array');
  const seen = new Set(existingIds);
  return records.map((record, index) => {
    const row = index + 1;
    if (!record || typeof record !== 'object' || Array.isArray(record)) throw new Error('promotion_record_' + row + '_must_be_an_object');
    const id = typeof record.id === 'string' ? record.id.trim() : '';
    if (!/^[a-z0-9][a-z0-9_-]*$/i.test(id)) throw new Error('promotion_record_' + row + '_has_invalid_id');
    if (seen.has(id)) throw new Error('promotion_duplicate_id_' + id);
    seen.add(id);
    if (typeof record.content !== 'string' || !record.content.trim()) throw new Error('promotion_record_' + row + '_requires_content');
    if (typeof record.type !== 'string' || !record.type.trim()) throw new Error('promotion_record_' + row + '_requires_type');
    if (typeof record.theme !== 'string' || !record.theme.trim()) throw new Error('promotion_record_' + row + '_requires_theme');
    const weight = Number.isFinite(record.weight) ? record.weight : record.weightSuggestion;
    if (!Number.isFinite(weight) || weight < 0 || weight > 1) throw new Error('promotion_record_' + row + '_requires_weight_0_to_1');
    if (!Array.isArray(record.positiveTriggers) || record.positiveTriggers.length === 0 || record.positiveTriggers.some(value => typeof value !== 'string' || !value.trim())) {
      throw new Error('promotion_record_' + row + '_requires_positive_triggers');
    }
    const negativeTriggers = record.negativeTriggers === undefined ? [] : record.negativeTriggers;
    if (!Array.isArray(negativeTriggers) || negativeTriggers.some(value => typeof value !== 'string' || !value.trim())) {
      throw new Error('promotion_record_' + row + '_has_invalid_negative_triggers');
    }
    const links = record.relatedItemIds === undefined ? [] : record.relatedItemIds;
    if (!Array.isArray(links) || links.some(value => typeof value !== 'string')) throw new Error('promotion_record_' + row + '_has_invalid_links');
    const features = {};
    for (const factor of FACTORS) {
      const value = record.features && record.features[factor];
      if (value !== undefined) {
        if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error('promotion_record_' + row + '_has_invalid_feature_' + factor);
        features[factor] = value;
      }
    }
    const item = {
      id,
      title: record.theme.trim().slice(0, 100) || record.content.trim().slice(0, 100),
      summary: record.content.trim(),
      type: record.type.trim(),
      theme: record.theme.trim(),
      weight,
      triggers: [...new Set(record.positiveTriggers.map(value => value.trim()))].map(text => ({
        text,
        // Short or question-word-heavy phrases should not outrank a more specific matching trigger.
        weight: triggerWeight(text)
      })),
      negativeTriggers: [...new Set(negativeTriggers.map(value => value.trim()))],
      tags: Array.isArray(record.labels) ? [...new Set(record.labels.filter(value => typeof value === 'string' && value.trim()).map(value => value.trim()))] : [],
      links: [...new Set(links)],
      features,
      reviewed: record.reviewed === true
    };
    if (record.source && typeof record.source === 'object' && !Array.isArray(record.source)) item.source = record.source;
    if (typeof record.sourceEvidence === 'string') item.sourceEvidence = record.sourceEvidence;
    if (Number.isFinite(record.confidence)) item.confidence = record.confidence;
    if (Array.isArray(record.validationWarnings)) item.validationWarnings = record.validationWarnings;
    return item;
  });
}

module.exports = { suggestionsToKnowledgeItems, triggerWeight };
