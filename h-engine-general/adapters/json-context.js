#!/usr/bin/env node
'use strict';

const { loadConfig, resolveProvider } = require('../src/config');
const { scanWithOptionalModel } = require('../src/engine');

function extractText(value, key = '') {
  if (typeof value === 'string') return /^(prompt|text|content|message)$/i.test(key) || !key ? value : '';
  if (Array.isArray(value)) return value.map(item => extractText(item)).filter(Boolean).join('\n');
  if (!value || typeof value !== 'object') return '';
  return Object.entries(value)
    .filter(([childKey]) => /^(prompt|text|content|message|messages|input)$/i.test(childKey))
    .map(([childKey, child]) => extractText(child, childKey))
    .filter(Boolean).join('\n');
}

async function main() {
  let raw = '';
  for await (const chunk of process.stdin) raw += chunk;
  let input;
  try { input = JSON.parse(raw); } catch (_) { input = raw; }
  const text = (typeof input === 'string' ? input : extractText(input)).trim();
  if (!text) return;
  const config = loadConfig();
  const result = await scanWithOptionalModel(text, config, { provider: resolveProvider() });
  if (!result.hits.length) return;
  const items = result.hits.map(hit => `${hit.title} (${hit.id})`).join('; ');
  const related = result.related.map(hit => `${hit.title} (${hit.id})`).join('; ');
  const relatedNote = related ? ` Linked knowledge: ${related}.` : '';
  process.stdout.write(`${JSON.stringify({ additionalContext: `Relevant knowledge items: ${items}.${relatedNote} Retrieve only these items from the connected knowledge base if useful.` })}\n`);
}

main().catch(() => process.exitCode = 0);
