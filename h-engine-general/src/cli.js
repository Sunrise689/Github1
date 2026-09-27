#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { loadConfig, resolveProvider, resolveAnnotationProvider, validateConfig, DEFAULT_CONFIG } = require('./config');
const { scanWithOptionalModel } = require('./engine');
const { annotateRecord } = require('./annotate');
const { ingestFile } = require('./ingest');
const { suggestionsToKnowledgeItems } = require('./promote');

function takeOption(args, name) {
  const index = args.indexOf(name);
  if (index < 0) return null;
  if (!args[index + 1] || args[index + 1].startsWith('--')) throw new Error(`missing_value_for_${name.slice(2)}`);
  const value = args[index + 1];
  args.splice(index, 2);
  return value;
}

function readStdin() {
  return new Promise((resolve, reject) => {
    let text = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', chunk => { text += chunk; });
    process.stdin.on('end', () => resolve(text));
    process.stdin.on('error', reject);
  });
}

async function runScan(args) {
  const configPath = takeOption(args, '--config');
  const profileId = takeOption(args, '--profile');
  const rulesOnly = args.includes('--rules-only');
  if (rulesOnly) args.splice(args.indexOf('--rules-only'), 1);
  const text = args.length ? args.join(' ') : await readStdin();
  const config = loadConfig(configPath || undefined);
  const provider = rulesOnly ? null : resolveProvider();
  const result = await scanWithOptionalModel(text.trim(), config, { provider, rulesOnly, profileId: profileId || undefined });
  process.stdout.write(`${JSON.stringify(result)}\n`);
}

async function runIngest(args) {
  const configPath = takeOption(args, '--config');
  const inputPath = takeOption(args, '--input');
  const outputPath = takeOption(args, '--output');
  const sourceUrl = takeOption(args, '--source-url');
  const sourceReliability = takeOption(args, '--source-reliability');
  const freshness = takeOption(args, '--freshness');
  const withModel = args.includes('--model');
  if (withModel) args.splice(args.indexOf('--model'), 1);
  if (args.length) throw new Error('unknown_ingest_option');
  if (!inputPath || !outputPath) throw new Error('ingest_requires_input_and_output');
  const inputAbsolute = path.resolve(inputPath);
  const outputAbsolute = path.resolve(outputPath);
  if (inputAbsolute === outputAbsolute) throw new Error('input_and_output_must_differ');
  const config = loadConfig(configPath || undefined);
  const provider = withModel ? resolveAnnotationProvider() : null;
  if (withModel && !provider) throw new Error('ingest_model_requires_provider');
  const sourceFeatures = {};
  if (sourceReliability !== null) sourceFeatures.sourceReliability = Number(sourceReliability);
  if (freshness !== null) sourceFeatures.freshness = Number(freshness);
  for (const [key, value] of Object.entries(sourceFeatures)) {
    if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error(`ingest_${key}_must_be_0_to_1`);
  }
  const records = await ingestFile(inputAbsolute, config, {
    provider,
    withModel,
    sourceUrl: sourceUrl || undefined,
    sourceFeatures
  });
  if (withModel && records.chunks > 0 && records.length === 0) {
    process.stderr.write(`${JSON.stringify({ error: 'ingest_no_atoms_generated', chunks: records.chunks, rejected: records.rejected || [] })}\n`);
    process.exitCode = 1;
    return;
  }
  fs.mkdirSync(path.dirname(outputAbsolute), { recursive: true });
  const output = records.map(record => JSON.stringify(record)).join('\n');
  fs.writeFileSync(outputAbsolute, output ? `${output}\n` : '', 'utf8');
  process.stdout.write(`${JSON.stringify({ chunks: records.chunks || undefined, atoms: records.length, duplicates: records.duplicates || 0, rejected: records.rejected || [], output: outputAbsolute, method: withModel ? 'model_suggestions' : 'rules_chunking' })}\n`);
}

async function runAnnotate(args) {
  const configPath = takeOption(args, '--config');
  const inputPath = takeOption(args, '--input');
  const outputPath = takeOption(args, '--output');
  const withModel = args.includes('--model');
  if (withModel) args.splice(args.indexOf('--model'), 1);
  if (args.length) throw new Error('unknown_annotate_option');
  if (!inputPath || !outputPath) throw new Error('annotate_requires_input_and_output');
  const inputAbsolute = path.resolve(inputPath);
  const outputAbsolute = path.resolve(outputPath);
  if (inputAbsolute === outputAbsolute) throw new Error('input_and_output_must_differ');
  const config = loadConfig(configPath || undefined);
  const provider = withModel ? resolveAnnotationProvider() : null;
  if (withModel && !provider) throw new Error('annotate_model_requires_provider');
  const lines = fs.readFileSync(inputAbsolute, 'utf8').split(/\r?\n/);
  const output = [];
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index].trim();
    if (!line) continue;
    const record = JSON.parse(line);
    if (!record || typeof record !== 'object' || Array.isArray(record)) throw new Error(`input_line_${index + 1}_must_be_an_object`);
    const text = typeof record.text === 'string' ? record.text : record.content;
    if (typeof text !== 'string') throw new Error(`input_line_${index + 1}_requires_text_or_content`);
    const id = record.id === undefined ? `line-${index + 1}` : String(record.id);
    const annotated = await annotateRecord({ id, text }, config, { withModel, provider });
    output.push(annotated);
  }
  fs.mkdirSync(path.dirname(outputAbsolute), { recursive: true });
  fs.writeFileSync(outputAbsolute, output.length ? `${output.map(record => JSON.stringify(record)).join('\n')}\n` : '', 'utf8');
  const modelFailures = output.filter(record => record.modelStatus).length;
  process.stdout.write(`${JSON.stringify({ records: output.length, modelFailures, output: outputAbsolute })}\n`);
  if (withModel && output.length > 0 && modelFailures === output.length) process.exitCode = 1;
}

async function runBuildIndex(args) {
  const configPath = takeOption(args, '--config');
  const inputPath = takeOption(args, '--input');
  const outputPath = takeOption(args, '--output');
  if (args.length) throw new Error('unknown_build_index_option');
  if (!inputPath || !outputPath) throw new Error('build_index_requires_input_and_output');
  const inputAbsolute = path.resolve(inputPath);
  const outputAbsolute = path.resolve(outputPath);
  const baseConfigAbsolute = path.resolve(configPath || DEFAULT_CONFIG);
  if (inputAbsolute === outputAbsolute) throw new Error('input_and_output_must_differ');
  if (outputAbsolute === baseConfigAbsolute) throw new Error('build_index_output_must_differ_from_base_config');
  const config = loadConfig(baseConfigAbsolute);
  const lines = fs.readFileSync(inputAbsolute, 'utf8').split(/\r?\n/);
  const records = [];
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index].trim();
    if (!line) continue;
    let record;
    try { record = JSON.parse(line); }
    catch (_) { throw new Error('input_line_' + (index + 1) + '_is_not_valid_json'); }
    records.push(record);
  }
  const existingIds = (config.knowledgeItems || []).map(item => item.id);
  const additions = suggestionsToKnowledgeItems(records, existingIds);
  config.knowledgeItems = [...(config.knowledgeItems || []), ...additions];
  validateConfig(config);
  fs.mkdirSync(path.dirname(outputAbsolute), { recursive: true });
  fs.writeFileSync(outputAbsolute, `${JSON.stringify(config, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify({ records: records.length, added: additions.length, unreviewed: additions.filter(item => !item.reviewed).length, output: outputAbsolute })}\n`);
}

async function main() {
  const args = process.argv.slice(2);
  const command = args.shift();
  if (command === 'activate' || command === 'scan') return runScan(args);
  if (command === 'annotate') return runAnnotate(args);
  if (command === 'ingest') return runIngest(args);
  if (command === 'build-index') return runBuildIndex(args);
  throw new Error('usage: node src/cli.js activate <query> [--profile ID] | node src/cli.js annotate --input FILE --output FILE | node src/cli.js ingest --input FILE --output FILE [--model] | node src/cli.js build-index --input FILE --output FILE');
}

main().catch(error => {
  process.stderr.write(`${JSON.stringify({ error: error.message || 'operation_failed' })}\n`);
  process.exitCode = 1;
});
