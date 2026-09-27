'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { loadConfig, validateConfig, resolveAnnotationProvider } = require('../src/config');
const { scan, scanWithOptionalModel, matchesTerm } = require('../src/engine');
const { annotateByRules, annotateRecord } = require('../src/annotate');
const { suggestionsToKnowledgeItems, triggerWeight } = require('../src/promote');
const { redact } = require('../src/redact');
const { chunkMarkdown, atomizeChunk, evidenceOverlap, resolveEvidence, deduplicateAtoms } = require('../src/ingest');

const config = loadConfig();
const provider = { kind: 'openai-compatible', model: 'example-model', baseUrl: 'https://example.invalid/v1', apiKey: '', timeoutMs: 1000, redactBeforeModel: true };

test('weighted knowledge items rank by query activation and configured importance', () => {
  const result = scan('知识权重能让知识及时触发', config);
  assert.equal(result.hits[0].id, 'weighted-activation');
  assert.ok(result.hits[0].score > 0);
  assert.equal(result.hits[0].selectionSource, 'weighted-rules');
  assert.equal(Object.hasOwn(result, 'query'), false);
});

test('phrase matching respects word boundaries and common English plurals', () => {
  assert.equal(matchesTerm('SQLite is embedded', 'SQL'), false);
  assert.equal(matchesTerm('relational databases', 'relational database'), true);
  assert.equal(matchesTerm('methods for storage', 'method'), true);
  assert.equal(matchesTerm('HTTP request response in a client server model', 'request–response'), true);
  assert.equal(matchesTerm('hop by hop headers', 'hop-by-hop'), true);
});

test('negative triggers suppress a candidate and topK bounds context size', () => {
  const small = {
    scoring: { minScore: 0.1, topK: 1, negativePenalty: 0.8 },
    knowledgeItems: [
      { id: 'high', title: 'High', summary: '', weight: 0.9, triggers: ['database'], negativeTriggers: ['urgent'] },
      { id: 'low', title: 'Low', summary: '', weight: 0.4, triggers: ['database'] }
    ]
  };
  const normal = scan('database', small);
  assert.equal(normal.count, 1);
  assert.equal(normal.hits[0].id, 'high');
  const excluded = scan('database urgent', small);
  assert.equal(excluded.hits.length, 1);
  assert.equal(excluded.hits[0].id, 'low');
});

test('an activated knowledge item exposes its linked items separately', async () => {
  const result = await scanWithOptionalModel('知识权重', config, { rulesOnly: true });
  assert.equal(result.hits[0].id, 'weighted-activation');
  assert.ok(result.related.some(item => item.id === 'knowledge-links'));
  assert.equal(result.related[0].relation, 'linked');
});

test('external retrieval can constrain activation to candidate IDs', () => {
  const result = scan('知识权重', config, { candidateIds: ['source-provenance'] });
  assert.deepEqual(result.hits, []);
  assert.equal(result.candidateCount, 0);
});

test('preference profiles alter ordering while preserving query relevance as the gate', () => {
  const profiled = {
    scoring: { minScore: 0.01, topK: 2 },
    knowledgeItems: [
      { id: 'fresh', weight: 0.8, triggers: ['database'], features: { importance: 0.7, sourceReliability: 0.5, freshness: 1, foundational: 0.1, usage: 0.4 } },
      { id: 'foundational', weight: 0.8, triggers: ['database'], features: { importance: 0.7, sourceReliability: 0.5, freshness: 0.1, foundational: 1, usage: 0.4 } }
    ]
  };
  const current = scan('database', profiled, { profileId: 'current' });
  const foundational = scan('database', profiled, { profileId: 'foundational' });
  assert.equal(current.hits[0].id, 'fresh');
  assert.equal(foundational.hits[0].id, 'foundational');
  assert.equal(scan('unrelated', profiled, { profileId: 'current' }).count, 0);
  assert.equal(current.hits[0].profileId, 'current');
  assert.ok(current.hits[0].relevanceScore > 0);
});

test('markdown ingestion chunks by headings and paragraphs and retains source locations', () => {
  const chunks = chunkMarkdown('# Intro\n\nFirst fact.\n\n## Details\n\nSecond fact.\n\n```js\nconst x = 1;\n\nconst y = 2;\n```', 'sample.md');
  assert.equal(chunks.length, 3);
  assert.equal(chunks[0].source.heading, 'Intro');
  assert.equal(chunks[1].source.heading, 'Intro > Details');
  assert.match(chunks[2].text, /const x = 1;[\s\S]*const y = 2;/);
  assert.equal(chunks[1].source.file, 'sample.md');
});

test('source-prefixed ingestion IDs remain unique and exact duplicate atoms merge', () => {
  const left = chunkMarkdown('# A\n\nA fact.', 'a.md', 1800, 'a-12345678');
  const right = chunkMarkdown('# B\n\nB fact.', 'b.md', 1800, 'b-87654321');
  assert.notEqual(left[0].id, right[0].id);
  const merged = deduplicateAtoms([
    { id: 'a1', content: 'A useful fact.', positiveTriggers: ['fact'], sourceEvidence: 'A useful fact.' },
    { id: 'a2', content: 'A useful fact!', positiveTriggers: ['useful fact'], sourceEvidence: 'A useful fact!' }
  ]);
  assert.equal(merged.length, 1);
  assert.deepEqual(merged[0].positiveTriggers, ['fact', 'useful fact']);
  assert.equal(merged[0].duplicateSources.length, 1);
});

test('atomization validates evidence and drops unknown catalog links', async () => {
  const result = await atomizeChunk({ id: 'doc-0001', text: 'SQLite is an embedded database.', source: { file: 'notes.md', heading: 'Databases', block: 1 } }, config, {
    provider,
    sourceFeatures: { sourceReliability: 0.8 },
    fetchImpl: async () => ({ ok: true, json: async () => ({ choices: [{ message: { content: JSON.stringify({ atoms: [{
      content: 'SQLite is an embedded database.', sourceEvidence: 'SQLite is an embedded database.',
      type: 'label-2', labels: ['label-2'], theme: 'databases', positiveTriggers: ['SQLite'], negativeTriggers: [],
      relatedItemIds: ['not-a-catalog-item'], features: { importance: 0.7, foundational: 0.5 }, confidence: 0.8
    }] }) } }] }) })
  });
  assert.equal(result[0].type, 'fact');
  assert.deepEqual(result[0].relatedItemIds, []);
  assert.deepEqual(result[0].validationWarnings, ['unknown_related_item_ids_dropped']);
  assert.equal(result[0].reviewed, false);
});

test('evidence support warns when an exact quote is unrelated to the suggested fact', () => {
  assert.ok(evidenceOverlap('SQLite implements most of the SQL standard and relational model.', 'It omits materialized views and triggers.') < 0.35);
  assert.ok(evidenceOverlap('SQLite is an embedded database.', 'SQLite is embedded in applications.') >= 0.35);
  assert.equal(resolveEvidence('SQLite is an embedded database.', 'SQLite embedded database', 'SQLite is an embedded database.').match, 'matched_source_sentence');
});

test('model can veto every weighted candidate and final count follows selected items', async () => {
  const result = await scanWithOptionalModel('知识权重和知识关联', config, {
    provider,
    fetchImpl: async () => ({ ok: true, json: async () => ({ choices: [{ message: { content: '{"selectedIds":[]}' } }] }) })
  });
  assert.ok(result.candidateCount >= 2);
  assert.equal(result.count, 0);
  assert.deepEqual(result.hits, []);
  assert.equal(result.decision.mode, 'weighted-model-rerank');
});

test('zero lexical matches can use a compact knowledge index, not full document contents', async () => {
  let submitted;
  const result = await scanWithOptionalModel('What can help an unrelated idea?', config, {
    provider,
    fetchImpl: async (_url, init) => {
      submitted = JSON.parse(init.body);
      return { ok: true, json: async () => ({ choices: [{ message: { content: '{"selectedIds":["candidate-2"]}' } }] }) };
    }
  });
  assert.equal(result.candidateCount, 0);
  assert.equal(result.hits[0].id, 'atomic-knowledge');
  assert.equal(result.hits[0].selectionSource, 'model-catalog');
  assert.equal(result.decision.mode, 'model-catalog-selection');
  assert.match(submitted.messages[1].content, /"summary"/);
  assert.doesNotMatch(submitted.messages[1].content, /"text"\s*:/);
  assert.doesNotMatch(submitted.messages[1].content, /atomic-knowledge|weighted-activation/);
});

test('provider errors preserve weighted rule results and model-bound text is redacted', async () => {
  let sent = '';
  const result = await scanWithOptionalModel('知识权重、知识关联；联系 alex@example.com，key sk-12345678901234567890', config, {
    provider,
    fetchImpl: async (_url, init) => {
      sent = init.body;
      return { ok: false, json: async () => ({}) };
    }
  });
  assert.match(sent, /\[EMAIL\]/);
  assert.doesNotMatch(sent, /alex@example\.com|sk-12345678901234567890/);
  assert.ok(result.hits.length > 0);
  assert.equal(result.decision.mode, 'weighted-rules');
  assert.equal(result.decision.modelStatus, 'provider_http_error');
});

test('offline annotation returns classification suggestions without claiming review', () => {
  const result = annotateByRules('A method: how to connect related knowledge items.', config);
  assert.deepEqual(result.labels, ['method']);
  assert.equal(result.weightSuggestion, null);
  assert.equal(result.reviewed, false);
});

test('model annotation proposes knowledge metadata and validates catalog links', async () => {
  const result = await annotateRecord({ id: 'record-1', text: 'A method for knowledge weighting. Contact alex@example.com' }, config, {
    withModel: true,
    provider: { ...provider, kind: 'ollama', baseUrl: 'http://127.0.0.1:11434' },
    fetchImpl: async (_url, init) => {
      const body = JSON.parse(init.body);
      assert.match(body.messages[1].content, /\[EMAIL\]/);
      assert.doesNotMatch(body.messages[1].content, /alex@example\.com/);
      assert.doesNotMatch(body.messages[1].content, /weighted-activation|atomic-knowledge/);
      return { ok: true, json: async () => ({ message: { content: JSON.stringify({
        labels: ['label-1'], theme: 'knowledge-retrieval', positiveTriggers: ['knowledge weighting'],
        negativeTriggers: ['load all records'], relatedItemIds: ['item-1'],
        weightSuggestion: 0.91, confidence: 0.84
      }) } }) };
    }
  });
  assert.deepEqual(result.labels, ['method']);
  assert.deepEqual(result.relatedItemIds, ['weighted-activation']);
  assert.equal(result.weightSuggestion, 0.91);
  assert.equal(result.reviewed, false);
  assert.doesNotMatch(JSON.stringify(result), /alex@example\.com/);
});

test('redaction masks common contact data, tokens, and user home paths', () => {
  const value = redact('alex@example.com 13812345678 sk-12345678901234567890 C:\\Users\\Alice\\notes.txt');
  assert.doesNotMatch(value, /alex@example\.com|13812345678|sk-123|C:\\Users\\Alice/);
  assert.match(value, /\[EMAIL\]/);
  assert.match(value, /\[PHONE\]/);
  assert.match(value, /\[LOCAL_PATH\]/);
});

test('configuration rejects duplicate knowledge IDs and non-normalized weights', () => {
  assert.throws(() => validateConfig({ knowledgeItems: [
    { id: 'same', weight: 0.5, triggers: ['first'] }, { id: 'same', weight: 0.5, triggers: ['second'] }
  ] }), /config_duplicate_knowledge_item_id/);
  assert.throws(() => validateConfig({ knowledgeItems: [
    { id: 'invalid-weight', weight: 5, triggers: ['trigger'] }
  ] }), /config_knowledge_item_weight_must_be_0_to_1/);
});

test('annotation provider is cloud OpenAI-compatible and isolated from local retrieval settings', () => {
  assert.equal(resolveAnnotationProvider({ HENGINE_PROVIDER: 'ollama', HENGINE_MODEL: 'qwen3' }), null);
  assert.throws(() => resolveAnnotationProvider({ HENGINE_ANNOTATION_PROVIDER: 'ollama' }), /annotation_provider_must_be_cloud_provider/);
  const providerConfig = resolveAnnotationProvider({
    HENGINE_PROVIDER: 'ollama',
    HENGINE_ANNOTATION_PROVIDER: 'openai-compatible',
    HENGINE_ANNOTATION_MODEL: 'deepseek-chat',
    HENGINE_ANNOTATION_BASE_URL: 'https://api.deepseek.com/v1',
    HENGINE_ANNOTATION_API_KEY: 'test-only-key'
  });
  assert.equal(providerConfig.kind, 'openai-compatible');
  assert.equal(providerConfig.baseUrl, 'https://api.deepseek.com/v1');
  assert.equal(providerConfig.timeoutMs, 120000);
  const anthropic = resolveAnnotationProvider({
    HENGINE_ANNOTATION_PROVIDER: 'anthropic',
    HENGINE_ANNOTATION_MODEL: 'claude-example',
    HENGINE_ANNOTATION_API_KEY: 'test-only-key'
  });
  assert.equal(anthropic.baseUrl, 'https://api.anthropic.com/v1');
  assert.equal(anthropic.kind, 'anthropic');
});

test('Anthropic requests use the native messages API and parse JSON text blocks', async () => {
  let submitted;
  const { requestJson, endpointFor } = require('../src/provider');
  const anthropic = { kind: 'anthropic', model: 'claude-example', baseUrl: 'https://api.anthropic.com/v1', apiKey: 'test-only-key', timeoutMs: 1000 };
  const result = await requestJson(anthropic, [
    { role: 'system', content: 'Return JSON.' },
    { role: 'user', content: 'Label this.' }
  ], async (url, init) => {
    submitted = { url, init, body: JSON.parse(init.body) };
    return { ok: true, json: async () => ({ content: [{ type: 'text', text: '{"ok":true}' }] }) };
  });
  assert.equal(endpointFor(anthropic), 'https://api.anthropic.com/v1/messages');
  assert.equal(submitted.init.headers['x-api-key'], 'test-only-key');
  assert.equal(submitted.init.headers['anthropic-version'], '2023-06-01');
  assert.equal(submitted.body.system, 'Return JSON.');
  assert.deepEqual(result, { ok: true });
});

test('model suggestions promote to an activatable item while preserving provenance', () => {
  const record = {
    id: 'sqlite-fact-1', content: 'SQLite is an embedded database library.', type: 'fact', theme: 'database',
    weight: 0.8, positiveTriggers: ['SQLite', 'embedded database'], negativeTriggers: [],
    relatedItemIds: [], features: { importance: 0.9, sourceReliability: 0.8, freshness: 0.5, foundational: 0.7, usage: 0.5 },
    source: { url: 'https://example.org/source' }, sourceEvidence: 'SQLite is an embedded database library.', reviewed: false
  };
  const items = suggestionsToKnowledgeItems([record]);
  assert.equal(items[0].summary, record.content);
  assert.equal(items[0].weight, 0.8);
  assert.deepEqual(items[0].triggers, [{ text: 'SQLite', weight: 0.4 }, { text: 'embedded database', weight: 0.55 }]);
  assert.equal(items[0].sourceEvidence, record.sourceEvidence);
  assert.equal(items[0].reviewed, false);
  assert.equal(validateConfig({ knowledgeItems: items }).knowledgeItems.length, 1);
  assert.throws(() => suggestionsToKnowledgeItems([record, record]), /promotion_duplicate_id_sqlite-fact-1/);
  assert.equal(triggerWeight('what is HTTP'), 0.4);
  assert.equal(triggerWeight('HTTP caching'), 0.55);
  assert.equal(triggerWeight('how does HTTP work with clients'), 1);
  assert.equal(triggerWeight('关系数据库'), 1);
});

test('model ingest fails visibly when every cloud request is rejected', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'h-engine-ingest-failure-'));
  try {
    const input = path.join(root, 'input.md');
    const output = path.join(root, 'output.jsonl');
    fs.writeFileSync(input, '# Example\n\nA short public knowledge statement.', 'utf8');
    const env = {
      ...process.env,
      HENGINE_ANNOTATION_PROVIDER: 'openai-compatible',
      HENGINE_ANNOTATION_MODEL: 'test-model',
      HENGINE_ANNOTATION_BASE_URL: 'http://127.0.0.1:1/v1',
      HENGINE_ANNOTATION_API_KEY: 'test-only-key'
    };
    const cli = path.join(__dirname, '..', 'src', 'cli.js');
    const result = spawnSync(process.execPath, [cli, 'ingest', '--input', input, '--output', output, '--model'], {
      cwd: path.join(__dirname, '..'), env, encoding: 'utf8', timeout: 10000
    });
    assert.equal(result.status, 1, result.stderr);
    assert.match(result.stderr, /ingest_no_atoms_generated/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});
