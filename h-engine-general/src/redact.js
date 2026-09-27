'use strict';

function redact(text) {
  return String(text)
    .replace(/\bBearer\s+[A-Za-z0-9._~+\/-]+=*/gi, 'Bearer [REDACTED]')
    .replace(/\b(?:sk-[A-Za-z0-9_-]{16,}|(?:api[_-]?key|token|secret)\s*[:=]\s*[A-Za-z0-9._~+\/-]{12,})/gi, '[SECRET]')
    .replace(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi, '[EMAIL]')
    .replace(/(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)/g, '[PHONE]')
    .replace(/[A-Za-z]:\\Users\\[^\\\s]+(?:\\[^\s"'<>|]*)?/gi, '[LOCAL_PATH]')
    .replace(/\/home\/[^/\s]+(?:\/[^\s"'<>]*)?/g, '[LOCAL_PATH]');
}

module.exports = { redact };
