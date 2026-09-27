'use strict';

function endpointFor(provider) {
  if (provider.kind === 'anthropic') {
    return provider.baseUrl.endsWith('/messages') ? provider.baseUrl : `${provider.baseUrl}/messages`;
  }
  if (provider.kind === 'ollama') {
    return provider.baseUrl.endsWith('/api/chat') ? provider.baseUrl : `${provider.baseUrl}/api/chat`;
  }
  return provider.baseUrl.endsWith('/chat/completions')
    ? provider.baseUrl
    : `${provider.baseUrl}/chat/completions`;
}

function readContent(payload, kind) {
  if (kind === 'ollama') return payload && payload.message && payload.message.content;
  if (kind === 'anthropic') {
    const blocks = payload && payload.content;
    return Array.isArray(blocks) ? blocks.filter(block => block && block.type === 'text').map(block => block.text || '').join('') : '';
  }
  const content = payload && payload.choices && payload.choices[0] && payload.choices[0].message && payload.choices[0].message.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.map(part => part && part.text || '').join('');
  return '';
}

function parseJsonContent(content) {
  const trimmed = String(content || '').trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  const start = trimmed.indexOf('{');
  const end = trimmed.lastIndexOf('}');
  if (start < 0 || end < start) throw new Error('provider_invalid_json');
  try { return JSON.parse(trimmed.slice(start, end + 1)); }
  catch (_) { throw new Error('provider_invalid_json'); }
}

async function requestJson(provider, messages, fetchImpl = globalThis.fetch) {
  if (!provider || typeof fetchImpl !== 'function') throw new Error('provider_not_available');
  const endpoint = endpointFor(provider);
  const headers = { 'content-type': 'application/json' };
  if (provider.kind === 'anthropic') {
    headers['x-api-key'] = provider.apiKey;
    headers['anthropic-version'] = provider.apiVersion || '2023-06-01';
  } else if (provider.apiKey) headers.authorization = `Bearer ${provider.apiKey}`;
  const body = provider.kind === 'anthropic'
    ? {
      model: provider.model,
      max_tokens: provider.maxTokens || 4096,
      system: messages.filter(message => message.role === 'system').map(message => message.content).join('\n\n'),
      messages: messages.filter(message => message.role !== 'system'),
      temperature: 0
    }
    : provider.kind === 'ollama'
    ? { model: provider.model, messages, stream: false, format: 'json', options: { temperature: 0 } }
    : { model: provider.model, messages, stream: false, temperature: 0, response_format: { type: 'json_object' } };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), provider.timeoutMs || 12000);
  try {
    const response = await fetchImpl(endpoint, {
      method: 'POST', headers, body: JSON.stringify(body), signal: controller.signal
    });
    if (!response || !response.ok) throw new Error('provider_http_error');
    let payload;
    try { payload = await response.json(); } catch (_) { throw new Error('provider_invalid_response'); }
    return parseJsonContent(readContent(payload, provider.kind));
  } catch (error) {
    if (error && error.name === 'AbortError') throw new Error('provider_timeout');
    if (error && /^provider_/.test(error.message || '')) throw error;
    throw new Error('provider_request_failed');
  } finally {
    clearTimeout(timer);
  }
}

module.exports = { endpointFor, parseJsonContent, requestJson };
