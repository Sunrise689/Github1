'use strict';

const FACTORS = ['importance', 'sourceReliability', 'freshness', 'foundational', 'usage'];

const BUILTIN_PROFILES = {
  balanced: {
    label: 'Balanced',
    relevanceFloor: 0.55,
    weights: { importance: 0.35, sourceReliability: 0.2, freshness: 0.15, foundational: 0.15, usage: 0.15 }
  },
  authoritative: {
    label: 'Source-first',
    relevanceFloor: 0.55,
    weights: { importance: 0.25, sourceReliability: 0.4, freshness: 0.1, foundational: 0.15, usage: 0.05 }
  },
  current: {
    label: 'Current information',
    relevanceFloor: 0.55,
    weights: { importance: 0.2, sourceReliability: 0.15, freshness: 0.4, foundational: 0.1, usage: 0.1 }
  },
  foundational: {
    label: 'Foundational knowledge',
    relevanceFloor: 0.55,
    weights: { importance: 0.25, sourceReliability: 0.15, freshness: 0.1, foundational: 0.4, usage: 0.1 }
  }
};

function resolveProfile(config = {}, requestedId) {
  const policy = config.weightPolicy || {};
  const custom = policy.profiles || {};
  const profileId = requestedId || policy.defaultProfile || 'balanced';
  const profile = custom[profileId] || BUILTIN_PROFILES[profileId];
  if (!profile) throw new Error('weight_profile_not_found');
  return { id: profileId, ...profile };
}

function itemFeatures(item) {
  const features = item.features || {};
  return {
    importance: Number.isFinite(features.importance) ? features.importance : (Number.isFinite(item.weight) ? item.weight : 0.5),
    sourceReliability: Number.isFinite(features.sourceReliability) ? features.sourceReliability : 0.5,
    freshness: Number.isFinite(features.freshness) ? features.freshness : 0.5,
    foundational: Number.isFinite(features.foundational) ? features.foundational : 0.5,
    usage: Number.isFinite(features.usage) ? features.usage : 0.5
  };
}

function preferenceScore(item, profile) {
  const values = itemFeatures(item);
  const weights = profile.weights || {};
  const totalWeight = FACTORS.reduce((sum, factor) => sum + (Number.isFinite(weights[factor]) && weights[factor] > 0 ? weights[factor] : 0), 0);
  if (!totalWeight) throw new Error('weight_profile_requires_positive_weight');
  const contributions = {};
  const score = FACTORS.reduce((sum, factor) => {
    const rawWeight = Number.isFinite(weights[factor]) && weights[factor] > 0 ? weights[factor] : 0;
    const normalizedWeight = rawWeight / totalWeight;
    contributions[factor] = Number((normalizedWeight * values[factor]).toFixed(4));
    return sum + normalizedWeight * values[factor];
  }, 0);
  return { score, values, contributions };
}

function scoreWithProfile(relevance, item, config, profile) {
  const pref = preferenceScore(item, profile);
  const baseline = preferenceScore(item, BUILTIN_PROFILES.balanced);
  const floor = Number.isFinite(profile.relevanceFloor) ? profile.relevanceFloor : 0.55;
  const negatives = Number.isFinite(config.scoring && config.scoring.negativePenalty)
    ? config.scoring.negativePenalty : 0.25;
  // Compare a user's profile with the generic balanced baseline. Balanced leaves
  // the initial weight unchanged; other profiles move it within a bounded range.
  const ratio = baseline.score > 0 ? pref.score / baseline.score : 1;
  const multiplier = Math.max(floor, Math.min(2 - floor, ratio));
  return {
    preference: pref.score,
    baselinePreference: baseline.score,
    preferenceFactors: pref.contributions,
    profileMultiplier: multiplier,
    score: relevance * multiplier,
    negativePenalty: negatives
  };
}

module.exports = { FACTORS, BUILTIN_PROFILES, resolveProfile, itemFeatures, preferenceScore, scoreWithProfile };
