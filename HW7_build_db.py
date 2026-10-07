#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#HW07 - One-time database build
#
#Runs OUTSIDE the Streamlit app (in GitHub Codespaces), per HW7 step 2a, so the app only loads a
#finished database and never waits on embedding or scoring calls.
#
#Usage, from the repo root in a Codespaces terminal:
#   python HW7_build_db.py --pilot   Embeds everything, scores a small sample with both models,
#                                    prints a cost projection. Does not write the database.
#   python HW7_build_db.py           Full build. Scores every article and writes data/HW07/chroma_db.
#
#Every paid step is cached in data/HW07/build_cache, so an interrupted or repeated run never pays twice.

import sys
import os
import re
import json
import math
import time
import hashlib
import argparse
import platform
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

#Must run before chromadb is imported (same fix as Lab 4 and HW4)
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import numpy as np
import pandas as pd
import chromadb
from openai import OpenAI
from anthropic import Anthropic

#Paths
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / 'data' / 'HW07'
CSV_PATH = DATA_DIR / 'news.csv'
CHROMA_PATH = DATA_DIR / 'chroma_db'
CACHE_DIR = DATA_DIR / 'build_cache'
REPORT_PATH = DATA_DIR / 'build_report.json'
COLLECTION_NAME = 'HW7News'

#Embeddings (same model as the RAG lecture and HW4; the app must query with this model too)
EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_PRICE = 0.02 #USD per million tokens (RAG lecture slide)
EMBED_BATCH_SIZE = 100
EMBED_TEXT_CHARS = 8000

#The two scoring models (lower-cost vs higher-cost, HW7 step 4)
#Prices are USD per million tokens and are used only for the cost report.
#Checked Oct 2026: Sonnet 5.5 from Anthropic's pricing page, GPT-5.4 Mini from published price trackers.
SCORERS = {
    'mini': {'vendor': 'openai', 'model': 'gpt-5.4-mini', 'label': 'GPT-5.4 Mini',
             'secret': 'OPENAI_API_KEY', 'price_in': 0.75, 'price_out': 4.50,
             'temperature': 0}, #Accepted in the pilot: most repeatable scores
    'sonnet': {'vendor': 'anthropic', 'model': 'claude-sonnet-5-5', 'label': 'Claude Sonnet 5.5',
               'secret': 'ANTHROPIC_API_KEY', 'price_in': 2.00, 'price_out': 10.00,
               'temperature': None}, #Sonnet 5.5 rejects any non-default temperature
}

SCORE_BATCH_SIZE = 10 #Articles per scoring call
RETRY_BATCH_SIZES = (5, 1) #Smaller batches, then single articles, for anything a pass missed
SCORE_TEXT_CHARS = 1500 #Longest article is ~2,500 chars; the first 1,500 carry the story
ANTHROPIC_MAX_TOKENS = 16000 #Anthropic requires a cap; it also covers any adaptive thinking. Billed on use only
MAX_WORKERS = 4 #Parallel scoring calls
PILOT_ARTICLES = 20

#Hybrid score signals
COVERAGE_SIMILARITY = 0.82 #Cosine similarity at which two articles count as the same story
RELATED_LIMIT = 5 #Related article IDs kept per article, for context in the app
BREADTH_CAP = 3 #Mentioning 3+ other clients earns the full breadth score

#Legal significance rubric (risk + opportunity, from a law firm's point of view)
EVENT_TYPES = [
    'Litigation or legal dispute',
    'Regulatory or government action',
    'Deal, merger or investment',
    'Financing or capital markets',
    'Leadership or governance',
    'Cybersecurity, privacy or IP',
    'Labor or employment',
    'Product, expansion or partnership',
    'Financial results or market moves',
    'Other routine news',
]

TOOL_NAME = 'record_article_scores'

#Bump this whenever the rubric or schema changes; cached scores from an older version are archived and redone
RUBRIC_VERSION = 2 #v2: about-client check before scoring; completed deals count as opportunity, not risk

RUBRIC_PROMPT = """You are a senior analyst at a large global law firm. The firm monitors news about
its clients for two reasons: to spot legal RISK to a client, and to spot business OPPORTUNITY where a
client is likely to need new legal work. You score news articles for both.

Each article lists the client it was collected for. Work through every article in this order.

STEP 1, ABOUT_CLIENT: decide whether the article is actually about the client company: the company
itself, its business, its deals, or people acting for it. Answer false when the match is only a shared
name, abbreviation, person, place, venue or product that happens to match the client's name, or when
the client gets only a passing mention in a story about something else. If about_client is false,
risk and opportunity must both be 0.

STEP 2, score RISK and OPPORTUNITY for the client.

RISK, 0 to 10: how likely the story creates legal exposure or an urgent legal need for the client.
0 = no legal angle at all.
1-3 = routine business news with at most a remote legal angle.
4-6 = a credible legal issue: a contract dispute, compliance question, layoffs, a hinted investigation.
7-8 = an active proceeding: a lawsuit filed, a regulator investigating, an antitrust probe, a major data breach.
9-10 = severe exposure: criminal charges, major government enforcement, bet-the-company litigation.

OPPORTUNITY, 0 to 10: how likely the story leads the client to need new legal work.
0 = none.
1-3 = minor, such as a small partnership or routine product launch.
4-6 = expansion into a new market, a large financing, a significant partnership, a launch in a regulated area.
7-8 = an announced acquisition, IPO, major restructuring, or large fundraise.
9-10 = a transformative transaction such as a multi-billion-dollar merger or takeover bid.

DEALS: a completed, closed, or court- or shareholder-approved deal is OPPORTUNITY (integration,
financing and post-closing work), not RISK. Score risk for a deal only when the article reports a
dispute, challenge, regulatory block, investigation or litigation over it.

RULES
- Score from the article text only. Do not use outside knowledge about the companies.
- Stock-price commentary, analyst opinions, listicles, market-research press releases and product
reviews score low on both unless they report a specific legal or transactional event.
- When about_client is false, the reason must say what the name actually refers to.
- reason: one sentence of at most 25 words naming the specific event and why the firm should care.
If both scores are 2 or lower, say briefly why the story is routine.
- Call the record_article_scores tool once, with exactly one entry for every article ID provided."""

SCORE_SCHEMA = {
    'type': 'object',
    'properties': {
        'articles': {
            'type': 'array',
            'description': 'One entry per article ID provided.',
            'items': {
                'type': 'object',
                'properties': {
                    'id': {'type': 'string', 'description': 'The ARTICLE ID exactly as given.'},
                    'about_client': {'type': 'boolean',
                                     'description': 'True only if the article is actually about the client company.'},
                    'risk': {'type': 'integer', 'minimum': 0, 'maximum': 10,
                             'description': 'Legal risk to the client, 0 to 10.'},
                    'opportunity': {'type': 'integer', 'minimum': 0, 'maximum': 10,
                                    'description': 'Likelihood of new legal work, 0 to 10.'},
                    'event_type': {'type': 'string', 'enum': EVENT_TYPES},
                    'reason': {'type': 'string',
                               'description': 'One sentence, at most 25 words, naming the event and why it matters.'},
                },
                'required': ['id', 'about_client', 'risk', 'opportunity', 'event_type', 'reason'],
            },
        },
    },
    'required': ['articles'],
}

TOOL_DESCRIPTION = 'Records the risk and opportunity scores for every article in this batch.'

#OpenAI tool format (Functions lecture: type / function / name / description / parameters)
OPENAI_TOOL = {'type': 'function',
               'function': {'name': TOOL_NAME, 'description': TOOL_DESCRIPTION, 'parameters': SCORE_SCHEMA}}

#Anthropic tool format (Functions lecture "How about Claude?": name / description / input_schema)
ANTHROPIC_TOOL = {'name': TOOL_NAME, 'description': TOOL_DESCRIPTION, 'input_schema': SCORE_SCHEMA}

#Client name matching for the breadth signal
#Corporate suffixes are stripped so 'Xerox Corporation' also matches 'Xerox'
CORPORATE_SUFFIX = re.compile(r'(,?\s+(Inc\.?|Corporation|Corp\.?|Co\., Ltd\.|Ltd\.?|AB|plc|AG|SE))+$')

#Well-known alternate names that appear in the news text
EXTRA_ALIASES = {
    'Alphabet': ['Google'], 'Facebook': ['Meta'], 'Walt Disney': ['Disney'], 'Procter & Gamble': ['P&G'],
    'Hon Hai Precision': ['Foxconn', 'Hon Hai'], 'BBVA-Banco Bilbao Vizcaya': ['BBVA'],
    'JPMorgan Chase': ['JPMorgan', 'JP Morgan'], 'Goldman Sachs': ['Goldman'], 'Nvidia': ['NVIDIA'],
    'Toyota Motor': ['Toyota'], 'Hyundai Motor': ['Hyundai'], 'Samsung Electronics': ['Samsung'],
    'LG Electronics': ['LG'], 'General Motors': ['GM'], 'Hewlett Packard Enterprise': ['HPE'],
    'Seiko Epson Corporation': ['Epson'], 'Tata Consultancy Services': ['TCS'],
    'Thermo Fisher Scientific': ['Thermo Fisher'], 'Verizon Communications': ['Verizon'],
    'Valero Energy': ['Valero'], 'Motorola Solutions': ['Motorola'], 'Micron Technology': ['Micron'],
    'ON Semiconductor': ['onsemi'], 'NXP Semiconductors': ['NXP'], 'Infineon Technologies': ['Infineon'],
    'Akamai Technologies': ['Akamai'], 'DXC Technology': ['DXC'], 'NCR Voyix Corporation': ['NCR Voyix', 'NCR'],
    'United Microelectronics Corporation': ['UMC'], 'Renesas Electronics Corporation': ['Renesas'],
    'SK Hynix Inc.': ['SK hynix'], 'HCL Technologies Ltd.': ['HCLTech', 'HCL'],
    'Oaktree Capital Management': ['Oaktree'], 'The Carlyle Group': ['Carlyle'],
    'Brookfield Asset Management': ['Brookfield'], 'Blackstone Inc.': ['Blackstone'],
    'Generali Group': ['Generali'], 'ING Group': ['ING'], 'Andreessen Horowitz': ['a16z'],
    'Tiger Global Management': ['Tiger Global'], 'Clayton, Dubilier & Rice': ['CD&R'],
    'New Enterprise Associates': ['NEA'], 'Institutional Venture Partners': ['IVP'], 'Iconiq Capital': ['ICONIQ'],
    'EQT AB': ['EQT'], 'Clearlake Capital Group': ['Clearlake'], 'Greylock Partners': ['Greylock'],
    'Bessemer Venture Partners': ['Bessemer'], 'Lightspeed Venture Partners': ['Lightspeed'],
    'Norwest Venture Partners': ['Norwest'], 'Vista Equity Partners': ['Vista Equity'],
    'Leonard Green & Partners': ['Leonard Green'], 'Apax Partners': ['Apax'], 'GGV Capital': ['GGV'],
    'Balderton Capital': ['Balderton'], 'Genstar Capital': ['Genstar'], 'Redpoint Ventures': ['Redpoint'],
    'Jerusalem Venture Partners': ['JVP'], 'Greenspring Associates': ['Greenspring'],
    'Ally Bridge Group': ['Ally Bridge'], 'ASM Pacific Technology': ['ASM Pacific', 'ASMPT'],
    'LITE-ON Technology': ['Lite-On', 'LITE-ON'], 'Nanya Technology': ['Nanya'],
    'Powertech Technology Inc.': ['Powertech'], 'PEGATRON Corporation': ['Pegatron'],
    'Fujikura Ltd.': ['Fujikura'], 'Taiwan Semiconductor Manufacturing Co., Ltd. (TSMC)': ['TSMC', 'Taiwan Semiconductor'],
    'Advanced Micro Devices (AMD)': ['AMD'], 'Wistron Corporation': ['Wistron'], 'Qisda Corporation': ['Qisda'],
    'Xerox Corporation': ['Xerox'], 'ZTE Corporation': ['ZTE'], 'ASE Group': ['ASE Technology'],
}

#Names whose short form is an ordinary word, so only the full name counts
FULL_NAME_ONLY = {
    'Sharp Corporation': ['Sharp Corporation', 'Sharp Corp'], 'Matrix Partners': ['Matrix Partners'],
    'Quanta Computer': ['Quanta Computer'], 'Index Ventures': ['Index Ventures'],
    'Advent International': ['Advent International'], 'Insight Partners': ['Insight Partners'],
    'Insight Venture Partners': ['Insight Venture Partners'], 'Silver Lake': ['Silver Lake'],
    'Founders Fund': ['Founders Fund'],
}

#Hand-written patterns where an alias needs context ('macOS Sequoia' is not Sequoia Capital)
CUSTOM_PATTERNS = {
    'Sequoia Capital': r'(?<!macOS )(?<![A-Za-z0-9])Sequoia(?: Capital)?(?![A-Za-z0-9])',
}


#General helpers
RUN_LOG = CACHE_DIR / 'last_run.log' #Copy of everything printed, easy to open and copy from the editor


def log(message=''):
    print(message, flush=True)

    try:
        with open(RUN_LOG, 'a', encoding='utf-8') as handle:
            handle.write(str(message) + '\n')
    except OSError:
        pass


def load_json(path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            log(f'  Warning: could not read {path.name}; starting it fresh.')
    return default


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
    temp.replace(path) #Atomic, so a crash mid-write never corrupts the cache


def get_key(name): #Codespaces secrets first, then .streamlit/secrets.toml
    value = os.environ.get(name, '').strip()

    if value:
        return value

    secrets_file = REPO_ROOT / '.streamlit' / 'secrets.toml'

    if secrets_file.exists():
        try:
            import tomllib
            with open(secrets_file, 'rb') as handle:
                return str(tomllib.load(handle).get(name, '')).strip()
        except Exception:
            pass

    return ''


#Step 1: Load and clean the CSV
def article_id(url): #Stable ID, so caches survive reruns
    return 'art_' + hashlib.md5(url.encode('utf-8')).hexdigest()[:12]


def outlet_of(url): #Collapses subdomains, so in.investing.com and ca.investing.com are one outlet
    host = urlparse(url).netloc.lower().split(':')[0]

    if host.startswith('www.'):
        host = host[4:]

    parts = host.split('.')

    if len(parts) >= 3 and parts[-2] in {'co', 'com', 'net', 'org', 'gov', 'ac', 'edu'} and len(parts[-1]) == 2:
        return '.'.join(parts[-3:])

    return '.'.join(parts[-2:])


def split_document(document): #Every Document is 'title Description: text', some add ' content: text'
    title, _, rest = document.partition(' Description: ')
    description, _, content = rest.partition(' content: ')

    return title.strip(), description.strip(), content.strip()


def load_articles():
    raw = pd.read_csv(CSV_PATH)

    raw['company_name'] = raw['company_name'].astype(str).str.strip() #' TPG' had a leading space
    raw['Document'] = raw['Document'].astype(str).str.strip()
    raw['URL'] = raw['URL'].astype(str).str.strip()
    raw['published'] = pd.to_datetime(raw['Date'], utc=True, format='ISO8601') #Two formats: +00:00 and Z

    rows = []

    for url, group in raw.groupby('URL', sort=False):
        document = max(group['Document'], key=len) #12 URLs carry two versions; keep the fuller one
        companies = sorted(set(group['company_name'])) #4 articles were collected for two clients
        published = group['published'].min()
        title, description, content = split_document(document)

        rows.append({
            'id': article_id(url),
            'url': url,
            'outlet': outlet_of(url),
            'companies': companies,
            'document': document,
            'title': title,
            'description': description,
            'content': content,
            'published': published.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'published_ts': int(published.timestamp()),
        })

    articles = pd.DataFrame(rows).sort_values(['published_ts', 'id']).reset_index(drop=True)

    return raw, articles


#Step 2: Embeddings (cached)
def embedding_text(row):
    return f"Client: {', '.join(row['companies'])}. {row['document']}"[:EMBED_TEXT_CHARS]


def embed_articles(articles, client):
    ids_file = CACHE_DIR / 'embedding_ids.json'
    vectors_file = CACHE_DIR / 'embeddings.npy'
    cached = {}

    if ids_file.exists() and vectors_file.exists():
        cached_ids = json.loads(ids_file.read_text(encoding='utf-8'))
        cached_vectors = np.load(vectors_file)

        if len(cached_ids) == len(cached_vectors):
            cached = dict(zip(cached_ids, cached_vectors))

    missing = [i for i in range(len(articles)) if articles.at[i, 'id'] not in cached]
    tokens_used = 0

    if missing:
        log(f'  Embedding {len(missing)} articles with {EMBEDDING_MODEL}...')

        for start in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[start:start + EMBED_BATCH_SIZE]
            texts = [embedding_text(articles.loc[i]) for i in batch]
            response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)

            for i, item in zip(batch, response.data):
                cached[articles.at[i, 'id']] = np.asarray(item.embedding, dtype=np.float32)

            if getattr(response, 'usage', None) is not None:
                tokens_used += int(getattr(response.usage, 'total_tokens', 0) or 0)

            log(f'    {min(start + EMBED_BATCH_SIZE, len(missing))}/{len(missing)} embedded')
    else:
        log('  All embeddings found in cache (no API calls).')

    ordered_ids = list(articles['id'])
    matrix = np.vstack([cached[i] for i in ordered_ids]).astype(np.float32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ids_file.write_text(json.dumps(ordered_ids), encoding='utf-8')
    np.save(vectors_file, matrix)

    return matrix, tokens_used


#Step 3: Coverage (how many different outlets ran the same story)
def compute_coverage(articles, matrix, threshold):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    unit = matrix / np.clip(norms, 1e-12, None)
    similarity = unit @ unit.T
    np.fill_diagonal(similarity, -1.0) #An article is not its own neighbour

    outlets_per_article = []
    related_per_article = []

    for i in range(len(articles)):
        similar = np.where(similarity[i] >= threshold)[0]
        outlets = {articles.at[i, 'outlet']} | {articles.at[j, 'outlet'] for j in similar}
        closest = similar[np.argsort(-similarity[i, similar])][:RELATED_LIMIT]

        outlets_per_article.append(len(outlets))
        related_per_article.append(','.join(articles.at[j, 'id'] for j in closest))

    articles['coverage_outlets'] = outlets_per_article
    articles['related_ids'] = related_per_article

    most = max(outlets_per_article)

    #Log scale: going from 1 to 3 outlets matters more than going from 10 to 12
    articles['coverage_score'] = [
        round(math.log(n) / math.log(most), 4) if most > 1 else 0.0 for n in outlets_per_article
    ]

    return similarity


def report_coverage(articles, similarity, threshold):
    counts = articles['coverage_outlets'].value_counts().sort_index()
    log('  Outlets per story: ' + ', '.join(f'{k} outlet(s): {v}' for k, v in counts.items()))

    log('  Most widely covered stories:')
    shown = set()

    for _, row in articles.sort_values('coverage_outlets', ascending=False).iterrows():
        if row['title'] in shown: #Every article in a story shares the count, so list each story once
            continue

        shown.add(row['title'])
        log(f"    [{row['coverage_outlets']} outlets] {row['title'][:90]}")

        if len(shown) == 5:
            break

    #Pairs just above the cut-off, to judge whether the threshold is too loose
    upper = np.triu(similarity, k=1)
    near = np.argwhere((upper >= threshold) & (upper < threshold + 0.03))

    if len(near):
        log(f'  Sample pairs just above the {threshold} cut-off (should be the same story):')
        rng = np.random.default_rng(7)

        for a, b in near[rng.choice(len(near), size=min(4, len(near)), replace=False)]:
            log(f"    {similarity[a, b]:.3f} | {articles.at[a, 'title'][:60]} || {articles.at[b, 'title'][:60]}")


#Step 4: Client breadth (other clients named in the article)
def client_aliases(name):
    if name in FULL_NAME_ONLY:
        aliases = set(FULL_NAME_ONLY[name])
    else:
        base = re.sub(r'\s*\([^)]*\)', '', name).strip() #Drops '(AMD)' / '(TSMC)'; added back via EXTRA_ALIASES
        aliases = {name, base, CORPORATE_SUFFIX.sub('', base).strip()}

    aliases.update(EXTRA_ALIASES.get(name, []))

    for alias in list(aliases):
        if alias.isupper() and len(alias) >= 4: #'SONY' is written 'Sony' in most articles
            aliases.add(alias.title())

    return {alias for alias in aliases if len(alias) >= 2}


def build_client_patterns(all_clients):
    patterns = {}

    for name in all_clients:
        if name in CUSTOM_PATTERNS:
            patterns[name] = re.compile(CUSTOM_PATTERNS[name])
            continue

        #Case-sensitive with letter/digit boundaries, so 'Intel' does not match 'Intelligence'
        options = sorted(client_aliases(name), key=len, reverse=True)
        joined = '|'.join(r'(?<![A-Za-z0-9])' + re.escape(alias) + r'(?![A-Za-z0-9])' for alias in options)
        patterns[name] = re.compile(joined)

    return patterns


def compute_client_signals(articles, all_clients):
    patterns = build_client_patterns(all_clients)
    mentioned, others, breadth = [], [], []

    for _, row in articles.iterrows():
        text = row['document']
        own = set(row['companies'])
        mentioned.append(any(patterns[name].search(text) for name in own))
        found = sorted(name for name, pattern in patterns.items() if name not in own and pattern.search(text))
        others.append('; '.join(found))
        breadth.append(round(min(len(found), BREADTH_CAP) / BREADTH_CAP, 4))

    articles['client_mentioned'] = mentioned #False for ~20% of articles: often not about the client at all
    articles['other_clients'] = others
    articles['client_breadth_score'] = breadth


#Step 5: Recency
def compute_recency(articles):
    oldest = articles['published_ts'].min()
    newest = articles['published_ts'].max()
    span = max(newest - oldest, 1)

    articles['recency_score'] = ((articles['published_ts'] - oldest) / span).round(4)


#Step 6: Legal significance scoring with both models (function calling for structured output)
def scoring_message(batch):
    blocks = []

    for row in batch:
        blocks.append(
            f"ARTICLE ID: {row['id']}\n"
            f"CLIENT: {', '.join(row['companies'])}\n"
            f"PUBLISHED: {row['published'][:10]}\n"
            f"OUTLET: {row['outlet']}\n"
            f"TEXT: {row['document'][:SCORE_TEXT_CHARS]}"
        )

    return ('Score each of the following articles.\n\n' + '\n\n---\n\n'.join(blocks) +
            f'\n\nRespond only by calling the {TOOL_NAME} tool, with one entry for each of the '
            f'{len(batch)} article IDs above.')


def call_openai(client, cfg, user_text, state):
    request = dict(
        model=cfg['model'],
        messages=[{'role': 'system', 'content': RUBRIC_PROMPT}, {'role': 'user', 'content': user_text}],
        tools=[OPENAI_TOOL],
        tool_choice={'type': 'function', 'function': {'name': TOOL_NAME}}, #Forces the structured answer
    )

    if cfg['temperature'] is not None and state.get('temperature_ok', True):
        request['temperature'] = cfg['temperature']

    try:
        response = client.chat.completions.create(**request)
    except Exception as error:
        if 'temperature' in request and 'temperature' in str(error).lower(): #Some models only allow the default
            state['temperature_ok'] = False
            request.pop('temperature')
            response = client.chat.completions.create(**request)
        else:
            raise

    tool_calls = response.choices[0].message.tool_calls or []

    if not tool_calls:
        raise ValueError('The model did not call the scoring tool.')

    entries = []

    for call in tool_calls:
        entries.extend(json.loads(call.function.arguments).get('articles', []))

    usage = response.usage

    return entries, int(usage.prompt_tokens), int(usage.completion_tokens)


def call_anthropic(client, cfg, user_text, state):
    request = dict(
        model=cfg['model'],
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=RUBRIC_PROMPT, #Anthropic takes the system prompt as its own parameter
        messages=[{'role': 'user', 'content': user_text}],
        tools=[ANTHROPIC_TOOL],
        #Sonnet 5.5 rejects forced tool use ('tool' / 'any'), so the prompt asks for the call and
        #any reply without one counts as a failed batch and is retried in a later pass
        tool_choice={'type': 'auto'},
    )

    if cfg['temperature'] is not None and state.get('temperature_ok', True):
        request['temperature'] = cfg['temperature']

    try:
        response = client.messages.create(**request)
    except Exception as error:
        if 'temperature' in request and 'temperature' in str(error).lower():
            state['temperature_ok'] = False
            request.pop('temperature')
            response = client.messages.create(**request)
        else:
            raise

    if response.stop_reason == 'max_tokens':
        raise ValueError('The response hit max_tokens before the tool call finished.')

    entries = []

    for block in response.content:
        if block.type == 'tool_use':
            entries.extend((block.input or {}).get('articles', []))

    if not entries:
        raise ValueError('The model did not call the scoring tool.')

    return entries, int(response.usage.input_tokens), int(response.usage.output_tokens)


def to_bool(value): #Accepts true/false even if a model sends them as text
    if isinstance(value, bool):
        return value

    if isinstance(value, str) and value.strip().lower() in ('true', 'false'):
        return value.strip().lower() == 'true'

    return None


def to_score(value): #Clamps to 0-10 even if a model ignores the schema limits
    try:
        return max(0, min(10, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def clean_entries(entries, batch_ids):
    cleaned = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        aid = str(entry.get('id', '')).strip()
        about_client = to_bool(entry.get('about_client'))
        risk = to_score(entry.get('risk'))
        opportunity = to_score(entry.get('opportunity'))

        if aid not in batch_ids or about_client is None or risk is None or opportunity is None:
            continue #Incomplete entries are retried in a later pass

        if not about_client: #The rubric's rule, enforced in code in case a model breaks it
            risk, opportunity = 0, 0

        event_type = entry.get('event_type')

        cleaned[aid] = {
            'about_client': about_client,
            'risk': risk,
            'opportunity': opportunity,
            'event_type': event_type if event_type in EVENT_TYPES else 'Other routine news',
            'reason': ' '.join(str(entry.get('reason', '')).split())[:300],
        }

    return cleaned


def make_client(cfg):
    key = get_key(cfg['secret'])

    if not key:
        raise SystemExit(f"Missing {cfg['secret']}. Add it as a Codespaces secret, then restart the Codespace.")

    if cfg['vendor'] == 'openai':
        return OpenAI(api_key=key, max_retries=5, timeout=180)

    return Anthropic(api_key=key, max_retries=5, timeout=180)


def score_with_model(key, articles, limit=None, workers=MAX_WORKERS):
    cfg = SCORERS[key]
    cache_file = CACHE_DIR / f'scores_{key}.json'
    cache = load_json(cache_file, {})
    old_version = cache.get('rubric_version', 1)

    if cache and (cache.get('model') != cfg['model'] or old_version != RUBRIC_VERSION):
        archive = CACHE_DIR / f'scores_{key}_rubric_v{old_version}.json' #Kept for the write-up's before/after
        save_json(archive, cache)
        log(f"  [{cfg['label']}] Rubric or model changed; archived old scores to {archive.name} and rescoring.")
        cache = {}

    if not cache:
        cache = {'model': cfg['model'], 'rubric_version': RUBRIC_VERSION, 'scores': {},
                 'usage': {'input_tokens': 0, 'output_tokens': 0, 'calls': 0}}

    records = articles.to_dict('records')
    valid_ids = {r['id'] for r in records}
    cache['scores'] = {k: v for k, v in cache['scores'].items() if k in valid_ids}

    todo = [r for r in records if r['id'] not in cache['scores']]

    if limit is not None: #Pilot: a fixed random sample, so it covers many clients rather than one day
        order = np.random.default_rng(42).permutation(len(todo))
        todo = [todo[i] for i in order][:max(0, limit - len(cache['scores']))]

    run_usage = {'input_tokens': 0, 'output_tokens': 0, 'calls': 0, 'scored': 0, 'seconds': 0.0}

    if not todo:
        log(f"  [{cfg['label']}] Nothing to score (cached: {len(cache['scores'])}).")
        save_json(cache_file, cache)
        return cache, run_usage

    client = make_client(cfg)
    caller = call_openai if cfg['vendor'] == 'openai' else call_anthropic
    state = {}
    started = time.time()

    def run_batch(batch):
        batch_ids = {r['id'] for r in batch}
        entries, tokens_in, tokens_out = caller(client, cfg, scoring_message(batch), state)
        return clean_entries(entries, batch_ids), tokens_in, tokens_out

    def record(scores, tokens_in, tokens_out):
        cache['scores'].update(scores)

        for bucket in (cache['usage'], run_usage):
            bucket['input_tokens'] += tokens_in
            bucket['output_tokens'] += tokens_out
            bucket['calls'] += 1

        run_usage['scored'] += len(scores)
        save_json(cache_file, cache) #Saved after every batch, so progress is never lost

    for pass_number, size in enumerate((SCORE_BATCH_SIZE,) + RETRY_BATCH_SIZES, start=1):
        pending = [r for r in todo if r['id'] not in cache['scores']]

        if not pending:
            break

        batches = [pending[i:i + size] for i in range(0, len(pending), size)]
        log(f"  [{cfg['label']}] Pass {pass_number}: {len(pending)} articles in {len(batches)} calls")

        #Probe with one batch first, so a wrong model ID or key fails fast instead of 100 times
        if pass_number == 1:
            try:
                record(*run_batch(batches[0]))
            except Exception as error:
                raise SystemExit(f"[{cfg['label']}] The first scoring call failed:\n  {error}\n"
                                 f"Common causes: a wrong model ID ('{cfg['model']}'), a missing "
                                 f"{cfg['secret']} secret, or a request setting this model does not support.")

            batches = batches[1:]

        failures = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_batch, batch) for batch in batches]

            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    record(*future.result())
                except Exception as error:
                    failures += 1
                    log(f'    A batch failed and will be retried: {str(error)[:160]}')

                if done % 10 == 0 or done == len(futures):
                    log(f"    {len(cache['scores'])}/{len(records)} articles scored")

        if failures:
            log(f'    {failures} batch(es) failed in pass {pass_number}.')

    run_usage['seconds'] = round(time.time() - started, 1)
    missing = [r['id'] for r in todo if r['id'] not in cache['scores']]

    if missing:
        log(f"  [{cfg['label']}] {len(missing)} articles still unscored. Run the script again to finish them.")

    if state.get('temperature_ok') is False:
        log(f"  [{cfg['label']}] Note: this model rejected temperature=0, so it ran at its default temperature.")

    cache['temperature_zero'] = state.get('temperature_ok', True)
    save_json(cache_file, cache)

    return cache, run_usage


def agreement_summary(articles, caches, ids):
    mini = caches['mini']['scores']
    sonnet = caches['sonnet']['scores']
    regex = dict(zip(articles['id'], articles['client_mentioned']))
    n = len(ids)

    if n == 0:
        return '  No articles scored by both models yet.'

    about_agree = sum(mini[i]['about_client'] == sonnet[i]['about_client'] for i in ids)
    gaps = [abs(max(mini[i]['risk'], mini[i]['opportunity']) - max(sonnet[i]['risk'], sonnet[i]['opportunity']))
            for i in ids]
    lines = [
        f'  Agreement on {n} articles scored by both models:',
        f'    About-client answer matches: {about_agree} of {n}',
        f'    Significance within 2 points: {sum(g <= 2 for g in gaps)} of {n} (average gap {sum(gaps) / n:.1f})',
    ]

    for key, cfg in SCORERS.items():
        matches = sum(caches[key]['scores'][i]['about_client'] == bool(regex[i]) for i in ids)
        lines.append(f"    {cfg['label']} about-client vs name-matching flag: {matches} of {n} match")

    return '\n'.join(lines)


def cost_of(cfg, usage):
    return usage['input_tokens'] / 1e6 * cfg['price_in'] + usage['output_tokens'] / 1e6 * cfg['price_out']


#Step 7: Write the ChromaDB folder
def chroma_settings():
    try:
        from chromadb.config import Settings
        return Settings(anonymized_telemetry=False)
    except Exception:
        return None


def build_metadata(row, scores):
    metadata = {
        'title': row['title'][:500],
        'description': row['description'][:800],
        'url': row['url'],
        'outlet': row['outlet'],
        'companies': '; '.join(row['companies']),
        'company_count': len(row['companies']),
        'published': row['published'],
        'published_ts': int(row['published_ts']),
        'coverage_outlets': int(row['coverage_outlets']),
        'coverage_score': float(row['coverage_score']),
        'related_ids': row['related_ids'],
        'client_mentioned': bool(row['client_mentioned']),
        'other_clients': row['other_clients'],
        'client_breadth_score': float(row['client_breadth_score']),
        'recency_score': float(row['recency_score']),
    }

    for key in SCORERS:
        entry = scores[key][row['id']]
        metadata[f'{key}_about_client'] = bool(entry['about_client'])
        metadata[f'{key}_risk'] = int(entry['risk'])
        metadata[f'{key}_opportunity'] = int(entry['opportunity'])
        metadata[f'{key}_significance'] = round(max(entry['risk'], entry['opportunity']) / 10, 4)
        metadata[f'{key}_event_type'] = entry['event_type']
        metadata[f'{key}_reason'] = entry['reason']

    return metadata


def write_chroma(articles, matrix, scores):
    settings = chroma_settings()

    if settings is not None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH), settings=settings)
    else:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME) #A rebuild always starts from an empty collection
    except Exception:
        pass

    #Embeddings are supplied directly, so Chroma's built-in embedding model is never downloaded.
    #OpenAI embeddings are unit length, so Chroma's default L2 distance ranks exactly like cosine.
    collection = client.create_collection(COLLECTION_NAME)

    records = articles.to_dict('records')

    for start in range(0, len(records), 256):
        chunk = records[start:start + 256]

        collection.add(
            ids=[r['id'] for r in chunk],
            embeddings=[matrix[start + i].tolist() for i in range(len(chunk))],
            documents=[r['document'] for r in chunk],
            metadatas=[build_metadata(r, scores) for r in chunk],
        )

    #Self-check: count matches, and every article's nearest neighbour is itself
    probe = collection.query(query_embeddings=[matrix[0].tolist()], n_results=1)

    if collection.count() != len(records) or probe['ids'][0][0] != records[0]['id']:
        raise SystemExit('The database self-check failed. Do not commit this build.')

    return collection.count()


#Main
def main():
    parser = argparse.ArgumentParser(description='Builds the HW7 news database.')
    parser.add_argument('--pilot', action='store_true', help='Score a small sample and project the cost.')
    parser.add_argument('--threshold', type=float, default=COVERAGE_SIMILARITY,
                        help='Similarity at which two articles count as the same story.')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Parallel scoring calls.')
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(f'Could not find {CSV_PATH}. Put news.csv in data/HW07 first.')

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_LOG.write_text('', encoding='utf-8') #Each run starts a fresh log
    mode = 'PILOT' if args.pilot else 'FULL BUILD'
    log(f'=== HW7 database build ({mode}) ===')
    log(f'chromadb {chromadb.__version__} | Python {platform.python_version()}')

    log('\n[1/7] Loading and cleaning news.csv')
    raw, articles = load_articles()
    all_clients = sorted(raw['company_name'].unique())
    log(f'  {len(raw)} rows -> {len(articles)} unique articles ({len(raw) - len(articles)} duplicate rows removed)')
    log(f'  {len(all_clients)} clients | {articles["outlet"].nunique()} outlets | '
        f'{articles["published"].min()[:10]} to {articles["published"].max()[:10]}')

    log('\n[2/7] Embeddings')
    openai_client = make_client(SCORERS['mini']) #Embeddings always use OpenAI
    matrix, embed_tokens = embed_articles(articles, openai_client)

    log(f'\n[3/7] Coverage (same-story threshold {args.threshold})')
    similarity = compute_coverage(articles, matrix, args.threshold)
    report_coverage(articles, similarity, args.threshold)

    log('\n[4/7] Client breadth')
    compute_client_signals(articles, all_clients)
    log(f"  Articles naming their own client: {int(articles['client_mentioned'].sum())} of {len(articles)}")
    log(f"  Articles naming other clients: {int((articles['other_clients'] != '').sum())}")

    log('\n[5/7] Recency')
    compute_recency(articles)
    log('  Done.')

    log('\n[6/7] Legal significance scoring')
    limit = PILOT_ARTICLES if args.pilot else None
    caches, run_usages = {}, {}

    for key in SCORERS:
        caches[key], run_usages[key] = score_with_model(key, articles, limit=limit, workers=args.workers)

    log('\n  Cost:')
    total_projection = 0.0

    for key, cfg in SCORERS.items():
        usage = run_usages[key]
        spent = cost_of(cfg, usage)
        line = (f"    {cfg['label']}: {usage['calls']} calls, {usage['input_tokens']:,} in / "
                f"{usage['output_tokens']:,} out tokens, ${spent:.4f} this run, {usage['seconds']}s")

        scored_so_far = len(caches[key]['scores'])

        if args.pilot and scored_so_far:
            per_article = cost_of(cfg, caches[key]['usage']) / scored_so_far #All cached calls, not just this run
            remaining = len(articles) - scored_so_far
            projected = per_article * remaining
            total_projection += projected
            line += (f"\n      {scored_so_far} scored so far at ~${per_article:.5f} per article | "
                     f"projected for remaining {remaining}: ~${projected:.2f}")

        log(line)

    log(f'    Embeddings: {embed_tokens:,} tokens, ${embed_tokens / 1e6 * EMBEDDING_PRICE:.4f} this run')

    if args.pilot:
        log(f'    Projected cost to finish scoring with both models: ~${total_projection:.2f}')
        log('\n  Sample scores, both models (R = risk, O = opportunity):')
        shared = [i for i in articles['id'] if all(i in caches[k]['scores'] for k in SCORERS)][:PILOT_ARTICLES]

        for aid in shared:
            row = articles[articles['id'] == aid].iloc[0]
            log(f"    {', '.join(row['companies'])} | {row['title'][:80]}")

            for key, cfg in SCORERS.items():
                s = caches[key]['scores'][aid]
                about = 'about client' if s['about_client'] else 'NOT about client'
                log(f"      {cfg['label']:<18} R{s['risk']} O{s['opportunity']} | {about} | {s['event_type']} | {s['reason']}")

        log('\n' + agreement_summary(articles, caches, shared))

        log('\nPilot finished. The database was NOT written. Run without --pilot for the full build.')
        return

    incomplete = [cfg['label'] for key, cfg in SCORERS.items() if len(caches[key]['scores']) < len(articles)]

    if incomplete:
        raise SystemExit(f"Scoring is incomplete for: {', '.join(incomplete)}. Run the script again; "
                         'it resumes where it stopped. The database was NOT written.')

    log('\n[7/7] Writing ChromaDB')
    scores = {key: caches[key]['scores'] for key in SCORERS}
    count = write_chroma(articles, matrix, scores)
    log(f'  {count} articles written to {CHROMA_PATH.relative_to(REPO_ROOT)} (collection {COLLECTION_NAME})')

    report = {
        'built_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'chromadb_version': chromadb.__version__,
        'python_version': platform.python_version(),
        'collection': COLLECTION_NAME,
        'embedding_model': EMBEDDING_MODEL,
        'raw_rows': int(len(raw)),
        'articles': int(len(articles)),
        'duplicates_removed': int(len(raw) - len(articles)),
        'clients': int(len(all_clients)),
        'coverage_threshold': args.threshold,
        'coverage_outlets_distribution': {str(k): int(v) for k, v in articles['coverage_outlets'].value_counts().sort_index().items()},
        'client_mentioned': int(articles['client_mentioned'].sum()),
        'articles_naming_other_clients': int((articles['other_clients'] != '').sum()),
        'rubric_version': RUBRIC_VERSION,
        'model_agreement': agreement_summary(articles, caches, list(articles['id'])).split('\n'),
        'scorers': {
            key: {
                'model': cfg['model'],
                'label': cfg['label'],
                'about_client_true': int(sum(v['about_client'] for v in caches[key]['scores'].values())),
                'temperature_zero': caches[key].get('temperature_zero', True),
                'usage_total': caches[key]['usage'],
                'cost_total_usd': round(cost_of(cfg, caches[key]['usage']), 4),
            } for key, cfg in SCORERS.items()
        },
    }

    save_json(REPORT_PATH, report)
    log(f'  Build report saved to {REPORT_PATH.relative_to(REPO_ROOT)}')
    log(f'\nDone. Pin this in requirements.txt so Streamlit Cloud can read the database: chromadb=={chromadb.__version__}')


if __name__ == '__main__':
    main()
