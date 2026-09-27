# H-Engine: Knowledge Weighting and Activation

**What:** A configurable index for weighting and activating small knowledge items.  
**Problem:** Large knowledge bases are hard to trigger precisely when all their contents are treated alike or loaded together.  
**Result:** Weighted rules shortlist relevant knowledge, linked items can surface together, and an optional model can rerank or suggest metadata.

## What this component does

H-Engine is the **knowledge weighting and activation layer**. It works on a compact index of knowledge items: summaries, types, themes, trigger phrases, positive weights, negative triggers, tags, and links. For a query it ranks a small set of likely items instead of returning the entire index.

The numeric `weight` is an explicit base priority from `0` to `1`. Each trigger may also carry a `weight` from `0` to `1`. The engine first calculates lexical relevance, then applies a preference profile over five optional item features: importance, source reliability, freshness, foundational value, and observed usage. Every feature is in `[0,1]`; missing values default to `0.5`.

```text
max(0, relevance × profileMultiplier - negativePenalty × matched negative-trigger weight)

relevance = item weight × (1 - e^(-sum of matched trigger weights))
profileMultiplier = clamp(profileFeatureScore / balancedFeatureScore, relevanceFloor, 2 - relevanceFloor)
```

The built-in profiles are `balanced`, `authoritative` (source-first), `current`, and `foundational`. Select one with `--profile`; the query must still match a positive trigger before a knowledge item can activate. Custom profiles and factor weights can be defined in `weightPolicy.profiles`. The default `balanced` profile preserves each item's initial weight; other profiles adjust it relative to that generic baseline. `minScore` sets the activation threshold and `topK` bounds the number of returned items. Scores and model-generated factors are ranking suggestions, not calibrated probabilities or objective truth. `links` let a consumer fetch related knowledge separately from direct matches.

This package does not train or change a language model's internal parameters. It applies explicit weights to knowledge metadata at retrieval time. It does not store the full database or fetch document bodies: an application can use returned IDs to retrieve content from its own file store, SQL/vector database, or MCP/API connector. The included CLI and adapter operate on the compact index only; library callers can pass IDs from an external retrieval system through `scan(text, config, { candidateIds })`.

## Start

Requires Node.js 18 or newer. There are no third-party runtime dependencies.

```bash
node src/cli.js activate "知识权重能让知识及时触发" --rules-only
```

The included `config.example.json` is a neutral sample. Copy it to `config.json` and edit `knowledgeItems`, labels, weights, triggers, and links for your database. `config.json` is ignored by Git so private knowledge stays local.

```bash
node src/cli.js activate "知识关联" --config config.json
node src/cli.js activate "related knowledge" --config config.json --rules-only
node src/cli.js activate "database release history" --config config.json --profile current
```

Each knowledge item has an ID, summary, type, theme, weight, positive triggers, optional negative triggers, tags, links, and optional `features`. Keep summaries short; keep full knowledge content in the storage system that owns it. See `config.example.json` for the schema.

## Optional model selection

Without provider settings, weighted phrase matching runs locally. The optional selection model handles the query-time choice among candidates; it is separate from the knowledge annotation model. If a selection provider is configured:

- Multiple weighted matches are sent as a short candidate list for model reranking.
- If no phrase matches, the model can choose from a compact catalog of item metadata, limited by `scoring.semanticCatalogLimit` (default 80). Full documents are not sent by this layer.
- A single weighted match is returned directly.
- Provider errors or invalid IDs fall back to weighted rules. If the compact-catalog selection also finds nothing useful, no item is activated.

Set these environment variables in the shell running the command:

| Variable | Purpose |
| --- | --- |
| `HENGINE_PROVIDER` | `ollama` or `openai-compatible` for query-time candidate selection; leave empty for weighted rules only |
| `HENGINE_MODEL` | Model name recognized by that endpoint |
| `HENGINE_BASE_URL` | Optional endpoint base URL; defaults to local Ollama or the OpenAI API base |
| `HENGINE_API_KEY` | Optional bearer key; keep it in the environment, never in config or source |
| `HENGINE_TIMEOUT_MS` | Request timeout, default 12000 ms |
| `HENGINE_REDACT` | Redact common secrets, email, phone numbers, and home paths before model calls; defaults to `true` |

The exact selection model is not fixed in code. Real knowledge IDs stay local; the model sees temporary candidate IDs, which the engine maps back after validating the response. Short summaries, themes, tags, and trigger phrases are still sent with a model request, so review that metadata before using a cloud endpoint. Basic redaction is not complete protection. The program does not write raw queries to a ledger or log.

## Import and suggest metadata

`ingest` accepts Markdown or plain text. It splits by headings and paragraphs, keeping a compact source location. Without a model it emits reviewable text chunks and keyword label suggestions; this is not semantic atomization. Add `--model` to ask a configured provider to propose standalone atoms, labels, themes, triggers, links, importance/foundational feature scores, and source evidence:

```bash
node src/cli.js ingest --input docs/guide.md --output inbox.jsonl --config config.json --model --source-url https://example.org/guide
```

The importer validates labels and links, checks source evidence, and sets `reviewed: false`. If the model's quote is not an exact substring, it may recover a likely supporting sentence from the original chunk using token overlap; that fallback and low-support matches are marked in `validationWarnings` and still require human review. Invalid suggestions are dropped individually and listed in the CLI result. **Annotation calls use a separate cloud-only provider setting; the local Ollama route-selection model is never used for annotation.** Supported routes are OpenAI-compatible APIs (such as DeepSeek, GLM, or GPT endpoints) and Anthropic's native API for Claude. Set `HENGINE_ANNOTATION_PROVIDER` to `openai-compatible` or `anthropic`, and set `HENGINE_ANNOTATION_MODEL`, `HENGINE_ANNOTATION_API_KEY`, and, for OpenAI-compatible APIs, `HENGINE_ANNOTATION_BASE_URL`. Anthropic defaults to `https://api.anthropic.com/v1`. By default it redacts common secrets, email addresses, phone numbers, and home paths before sending chunks and headings to the cloud service; redaction is heuristic, so inspect inputs first. `sourceReliability`, `freshness`, and `usage` cannot be inferred reliably from text alone: they default to neutral `0.5`, or can be supplied as `--source-reliability 0.8 --freshness 0.6`. Model-created labels, features, content, and triggers are suggestions and require human review.

The `annotate` command remains available for JSONL records that are already split into proposed atoms. Offline keyword rules suggest labels; `--model` asks for metadata suggestions:

The `annotate` command accepts JSONL records where each line is one proposed knowledge atom. Offline keyword rules can suggest a type label. Add `--model` to request suggestions for theme, positive and negative triggers, links to existing IDs, and a weight from `0` to `1`:

```bash
node src/cli.js annotate --input knowledge-inbox.jsonl --output suggestions.jsonl --config config.json --model
```

Every result has `reviewed: false`; check atomicity, source, theme, duplicates, weights, and links before adding it to a knowledge base. Model confidence and suggested weights are not calibrated guarantees.

## Build an activation index

Use `build-index` to turn atomization suggestions into the compact knowledge-item format used by `activate`. It carries over themes, type labels, trigger phrases, suggested weights, feature scores, source evidence, and review status. The command keeps `reviewed: false` when the suggestions have not been reviewed; generating an index does not mean the annotations have been verified.

```bash
node src/cli.js build-index --input suggestions.jsonl --config config.json --output indexed-config.json
node src/cli.js activate "SQLite embedded database" --config indexed-config.json --rules-only
```

The base configuration is never overwritten by `build-index`. Review the generated items and their citations before using them for high-stakes decisions.

## Agent integration

`adapters/json-context.js` accepts JSON on stdin, extracts fields named `prompt`, `text`, `content`, `message`, `messages`, or `input`, and emits a generic `additionalContext` object naming relevant knowledge item IDs and linked items. The calling application can use those IDs with its own retrieval or database connector.

## Checks

```bash
npm test
```

## Scope and license

H-Engine is a compact knowledge activation index with a basic Markdown/text chunker, not a full database, vector store, or general-purpose document parser. For a large collection, use an external retrieval adapter to supply a compact candidate set with `candidateIds`. No license has been selected yet; a public repository without a license does not grant general reuse rights.

The package is marked private in `package.json` to prevent accidental npm publication. It can still be reviewed, cloned, and run from a public GitHub repository.
