'use strict';

const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_CONFIG = path.join(__dirname, '..', 'config.example.json');
const FACTORS = ['importance', 'sourceReliability', 'freshness', 'foundational', 'usage'];

function isWeight(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
}

function validateTerms(terms, label) {
  if (terms === undefined) return;
  if (!Array.isArray(terms) || terms.some(term => {
    if (typeof term === 'string') return false;
    return !term || typeof term !== 'object' || Array.isArray(term) || typeof term.text !== 'string' ||
      (term.weight !== undefined && !isWeight(term.weight));
  })) throw new Error(`config_invalid_${label}`);
}

function validateConfig(config) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    throw new Error('config_must_be_an_object');
  }
  if (config.knowledgeItems !== undefined && !Array.isArray(config.knowledgeItems)) {
    throw new Error('config_knowledgeItems_must_be_an_array');
  }
  if (config.labels !== undefined && !Array.isArray(config.labels)) {
    throw new Error('config_labels_must_be_an_array');
  }
  const itemIds = new Set();
  for (const item of config.knowledgeItems || []) {
    if (!item || typeof item.id !== 'string' || !/^[a-z0-9][a-z0-9_-]*$/i.test(item.id)) {
      throw new Error('config_invalid_knowledge_item_id');
    }
    if (itemIds.has(item.id)) throw new Error('config_duplicate_knowledge_item_id');
    itemIds.add(item.id);
    if (!isWeight(item.weight)) throw new Error('config_knowledge_item_weight_must_be_0_to_1');
    for (const key of ['title', 'summary', 'type', 'theme']) {
      if (item[key] !== undefined && typeof item[key] !== 'string') throw new Error(`config_invalid_knowledge_item_${key}`);
    }
    if (!Array.isArray(item.triggers) || item.triggers.length === 0) {
      throw new Error('config_knowledge_item_requires_triggers');
    }
    validateTerms(item.triggers, 'triggers');
    validateTerms(item.negativeTriggers || [], 'negative_triggers');
    for (const key of ['tags', 'links']) {
      if (item[key] !== undefined && (!Array.isArray(item[key]) || item[key].some(value => typeof value !== 'string'))) {
        throw new Error(`config_invalid_knowledge_item_${key}`);
      }
    }
    if (item.features !== undefined) {
      if (!item.features || typeof item.features !== 'object' || Array.isArray(item.features)) {
        throw new Error('config_invalid_knowledge_item_features');
      }
      for (const [factor, value] of Object.entries(item.features)) {
        if (!FACTORS.includes(factor) || !isWeight(value)) throw new Error(`config_invalid_knowledge_item_feature_${factor}`);
      }
    }
  }
  const labelIds = new Set();
  for (const label of config.labels || []) {
    if (!label || typeof label.id !== 'string' || !/^[a-z0-9][a-z0-9_-]*$/i.test(label.id)) {
      throw new Error('config_invalid_label_id');
    }
    if (labelIds.has(label.id)) throw new Error('config_duplicate_label_id');
    labelIds.add(label.id);
    if (label.description !== undefined && typeof label.description !== 'string') throw new Error('config_invalid_label_description');
    validateTerms(label.keywords || [], 'label_keywords');
  }
  if (config.scoring !== undefined) {
    const scoring = config.scoring;
    if (!scoring || typeof scoring !== 'object' || Array.isArray(scoring)) throw new Error('config_invalid_scoring');
    if (scoring.minScore !== undefined && !isWeight(scoring.minScore)) throw new Error('config_invalid_minScore');
    if (scoring.negativePenalty !== undefined && !isWeight(scoring.negativePenalty)) throw new Error('config_invalid_negativePenalty');
    if (scoring.topK !== undefined && (!Number.isInteger(scoring.topK) || scoring.topK < 1 || scoring.topK > 100)) {
      throw new Error('config_invalid_topK');
    }
    if (scoring.semanticCatalogLimit !== undefined && (!Number.isInteger(scoring.semanticCatalogLimit) || scoring.semanticCatalogLimit < 1 || scoring.semanticCatalogLimit > 500)) {
      throw new Error('config_invalid_semanticCatalogLimit');
    }
    if (scoring.linkedLimit !== undefined && (!Number.isInteger(scoring.linkedLimit) || scoring.linkedLimit < 0 || scoring.linkedLimit > 100)) {
      throw new Error('config_invalid_linkedLimit');
    }
  }
  if (config.weightPolicy !== undefined) {
    const policy = config.weightPolicy;
    if (!policy || typeof policy !== 'object' || Array.isArray(policy)) throw new Error('config_invalid_weightPolicy');
    const profiles = policy.profiles === undefined ? {} : policy.profiles;
    if (!profiles || typeof profiles !== 'object' || Array.isArray(profiles)) throw new Error('config_invalid_weightPolicy_profiles');
    for (const [profileId, profile] of Object.entries(profiles)) {
      if (!/^[a-z0-9][a-z0-9_-]*$/i.test(profileId) || !profile || typeof profile !== 'object' || Array.isArray(profile)) {
        throw new Error('config_invalid_weight_profile');
      }
      if (profile.label !== undefined && typeof profile.label !== 'string') throw new Error('config_invalid_weight_profile_label');
      if (profile.relevanceFloor !== undefined && !isWeight(profile.relevanceFloor)) throw new Error('config_invalid_weight_profile_relevanceFloor');
      if (!profile.weights || typeof profile.weights !== 'object' || Array.isArray(profile.weights)) throw new Error('config_invalid_weight_profile_weights');
      let total = 0;
      for (const [factor, value] of Object.entries(profile.weights)) {
        if (!FACTORS.includes(factor) || !Number.isFinite(value) || value < 0) throw new Error(`config_invalid_weight_profile_factor_${factor}`);
        total += value;
      }
      if (total <= 0) throw new Error('config_weight_profile_requires_positive_weight');
    }
    if (policy.defaultProfile !== undefined && typeof policy.defaultProfile !== 'string') throw new Error('config_invalid_weightPolicy_defaultProfile');
    if (policy.defaultProfile && !profiles[policy.defaultProfile] && !['balanced', 'authoritative', 'current', 'foundational'].includes(policy.defaultProfile)) {
      throw new Error('config_weightPolicy_default_profile_not_found');
    }
  }
  return config;
}

function loadConfig(filePath = DEFAULT_CONFIG) {
  const absolute = path.resolve(filePath);
  let parsed;
  try {
    parsed = JSON.parse(fs.readFileSync(absolute, 'utf8'));
  } catch (error) {
    if (error && error.code === 'ENOENT') throw new Error('config_file_not_found');
    throw new Error('config_file_is_not_valid_json');
  }
  return validateConfig(parsed);
}

function resolveProvider(env = process.env) {
  const kind = String(env.HENGINE_PROVIDER || '').trim().toLowerCase();
  if (!kind) return null;
  if (!['ollama', 'openai-compatible'].includes(kind)) throw new Error('unsupported_provider');
  const model = String(env.HENGINE_MODEL || '').trim();
  if (!model) throw new Error('provider_model_required');
  const baseUrl = String(env.HENGINE_BASE_URL || (kind === 'ollama' ? 'http://127.0.0.1:11434' : 'https://api.openai.com/v1')).trim();
  let parsedUrl;
  try { parsedUrl = new URL(baseUrl); } catch (_) { throw new Error('provider_base_url_invalid'); }
  if (!['http:', 'https:'].includes(parsedUrl.protocol)) throw new Error('provider_base_url_invalid');
  const timeoutValue = Number.parseInt(env.HENGINE_TIMEOUT_MS || '12000', 10);
  const timeoutMs = Number.isFinite(timeoutValue) ? Math.min(120000, Math.max(500, timeoutValue)) : 12000;
  return {
    kind,
    model,
    baseUrl: baseUrl.replace(/\/+$/, ''),
    apiKey: String(env.HENGINE_API_KEY || '').trim(),
    timeoutMs,
    redactBeforeModel: String(env.HENGINE_REDACT || 'true').toLowerCase() !== 'false'
  };
}

function resolveAnnotationProvider(env = process.env) {
  const kind = String(env.HENGINE_ANNOTATION_PROVIDER || '').trim().toLowerCase();
  if (!kind) return null;
  if (!['openai-compatible', 'anthropic'].includes(kind)) throw new Error('annotation_provider_must_be_cloud_provider');
  const model = String(env.HENGINE_ANNOTATION_MODEL || '').trim();
  if (!model) throw new Error('annotation_model_required');
  const baseUrl = String(env.HENGINE_ANNOTATION_BASE_URL || (kind === 'anthropic' ? 'https://api.anthropic.com/v1' : '')).trim();
  if (!baseUrl) throw new Error('annotation_base_url_required');
  let parsedUrl;
  try { parsedUrl = new URL(baseUrl); } catch (_) { throw new Error('annotation_base_url_invalid'); }
  if (!['http:', 'https:'].includes(parsedUrl.protocol)) throw new Error('annotation_base_url_invalid');
  const apiKey = String(env.HENGINE_ANNOTATION_API_KEY || '').trim();
  if (!apiKey) throw new Error('annotation_api_key_required');
  const timeoutValue = Number.parseInt(env.HENGINE_ANNOTATION_TIMEOUT_MS || '120000', 10);
  const timeoutMs = Number.isFinite(timeoutValue) ? Math.min(300000, Math.max(500, timeoutValue)) : 120000;
  return {
    kind,
    model,
    baseUrl: baseUrl.replace(/\/+$/, ''),
    apiKey,
    timeoutMs,
    redactBeforeModel: String(env.HENGINE_ANNOTATION_REDACT || 'true').toLowerCase() !== 'false'
  };
}

module.exports = { DEFAULT_CONFIG, loadConfig, resolveProvider, resolveAnnotationProvider, validateConfig };
