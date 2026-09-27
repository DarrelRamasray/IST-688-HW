#DARREL RAMASRAY
#IST 688 - Building HC-AI Apps
#HW05

import streamlit as st
from openai import OpenAI
import sys
import re
import json
from pathlib import Path
from bs4 import BeautifulSoup

#The sqlite swap has to run BEFORE chromadb is imported, or chromadb keeps the system sqlite
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import chromadb

#Configuration
#HW5 reads the same collection HW4 built, so nothing is embedded twice
COLLECTION_NAME = 'HW4Collection'
EMBEDDING_MODEL = 'text-embedding-3-small' #Must match the model that built the collection
CHROMA_PATH = './ChromaDB_for_HW4'
DATA_SUBFOLDER = Path('data') / 'HW04'

EMBED_BATCH_SIZE = 100
MAX_EMBED_CHARS = 20000 #Safety net

CHAT_MODEL = 'gpt-5.4-mini'
N_RESULTS = 8 #Chunks pulled from Chroma per search
MAX_ORGS = 5 #Organizations returned per search
BUFFER_INTERACTIONS = 5 #Short-term memory: the last 5 interactions
MAX_SEARCHES_PER_TURN = 3 #Caps the searches one question can trigger

FILENAME_PREFIX = 'syracuse.campuslabs.com_engage_organization_'
ORG_URL_PREFIX = 'https://syracuse.campuslabs.com/engage/organization/'

#Page Parsing Constants (unchanged from HW4)
PAGE_SECTIONS = ['About', 'Contact Information', 'Additional Information', 'Contact',
                 'Public Events', 'Officers', 'Documents', 'News', 'Gallery']

NOISE_LINES = {'View Full Roster', 'View More Events', 'View past events.',
               'Sign In To View Officers', 'Sign in to view officers'}

EMPTY_VALUES = {'no response', 'n/a', 'na', 'none', 'tbd', 'tba', '-', ''}

CONSULTANT_MARKER = 'consultant in Student Engagement'

FIELD_LABELS = {
    'Description': 'Description',
    'Website': 'Website',
    'Meeting Day': 'Meeting Day',
    'Meeting time': 'Meeting Time',
    'AM/PM': 'AM/PM',
    'Meeting Location': 'Meeting Location',
    'President Name': 'President',
    'Vice-President Name': 'Vice President',
    'Secretary (or other eboard position) name': 'Secretary or Other E-Board Member',
    'Treasurer/Fiscal Agent Name': 'Treasurer',
    'Full-time SU/ESF Faculty/Staff Advisor name': 'Faculty or Staff Advisor',
    'Member/Selection Process': 'How To Join',
}

LEADERSHIP_ORDER = ['President', 'Vice President', 'Secretary or Other E-Board Member',
                    'Treasurer', 'Faculty or Staff Advisor', 'Student Engagement Consultant']

EVENT_DATE_PATTERN = re.compile(
    r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), ', re.IGNORECASE)

WEEKDAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                 'Saturday', 'Sunday']

#The saved pages list event dates with no year (e.g. "Wednesday, September 18"), so the
#model cannot tell whether an event has passed. Event lines are relabeled before the model
#sees them.
EVENT_LINE_STORED = 'Upcoming event: '
EVENT_LINE_SHOWN = 'Event listed on the page (no year given, may have passed): '

#HW4's query-routing regex (FOLLOW_UP_PATTERN, NEW_SEARCH_PATTERN, MEETING_INTENT_PATTERN,
#EVENT_INTENT_PATTERN) is gone. The model now writes the search query and fills in the
#filters itself through the tool below.

GREETING = ('Ask me about Syracuse University student organizations. I can look up what a '
            'group does, when and where it meets, who runs it, and how to join.')


#Step 3a: the tool definition
#Same structure as the lecture's get_current_weather example: name, description,
#parameters, required. Only 'query' is required; the two filters are optional.
TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'relevant_club_info',
        'description': (
            'Search the Syracuse University student organization directory and return '
            'organization records: description, website, contact details, meeting day, time '
            'and location, officers, upcoming events, and how to join. Call this for any '
            'question about student organizations, clubs, or groups, including follow-up '
            'questions about an organization discussed earlier in the conversation.'),
        'parameters': {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': (
                        'A standalone search phrase that makes sense without the chat history. '
                        'Replace words like "they" or "that club" with the organization name '
                        'from the conversation and include the topic asked about, e.g. '
                        '"<organization name> meeting day and location".')},
                'meeting_days': {
                    'type': 'array',
                    'items': {'type': 'string', 'enum': WEEKDAY_NAMES},
                    'description': (
                        'Only when looking for organizations that meet on these weekdays. '
                        'Leave out when asking about one named organization\'s schedule.')},
                'has_upcoming_events': {
                    'type': 'boolean',
                    'description': (
                        'Set true only when looking for organizations that have upcoming '
                        'events. Leave out when asking about one named organization.')},
            },
            'required': ['query'],
        },
    },
}]


#System Prompts
#Each call gets its own system prompt, rebuilt every turn and never stored in the history.
#Call 1 (search prompt): decide whether to search, ask first, or reply directly.
#Call 2 (answer prompt): searching is over; answer from the results. The code also adds a
#THIS QUESTION block listing the organizations the search returned, so the model gets a
#concrete list instead of a general rule.
ROLE_INTRO = """You are an assistant that helps Syracuse University students find and join
registered student organizations, using organization pages from the university's 'Cuse
Activities directory."""

SEARCH_SYSTEM_PROMPT = ROLE_INTRO + """

On this step you decide whether to search the directory with the relevant_club_info tool,
what to search for, or whether to ask the student something first.

WHEN TO SEARCH
- Call relevant_club_info for any question about an organization, club, or group: what it
does, when and where it meets, who runs it, how to contact or join it, or its events. This
includes follow-up questions about an organization named earlier in the conversation.
- Search again even when an earlier reply covered the organization. Answers may only use
results from a search made for the current question.
- To compare organizations or look up several named organizations, search once for each.

WHEN TO ASK FIRST
- Don't make assumptions about what values to plug into functions. Ask for clarification if
a user request is ambiguous.
- If the student asks for the "best", "good", "fun", "popular", or "easiest" organizations
without saying what they are interested in, do not search. The pages cannot rate
organizations. Ask one short question about what they care about, such as a field of study,
a hobby, a cause, a kind of activity, or a meeting day. Search once they answer.
- If "they", "it", or "that club" could mean more than one organization named in the
conversation, ask which one.

HOW TO FILL IN A SEARCH
- Write the query so it stands on its own. Replace words like "they", "it", or "that club"
with the organization's name from the conversation, and name the topic asked about.
- Set meeting_days or has_upcoming_events only when looking across the directory for
organizations that fit. Leave them out when the question is about one named organization.

REPLYING WITHOUT A SEARCH
- Messages that need no organization facts, such as thanks or a greeting, need no search.
Reply briefly.
- A reply without a search must not state any fact about any organization, including facts
from earlier replies.
- Never mention the tool or searching. Speak about the directory and the organization pages."""

ANSWER_SYSTEM_PROMPT = ROLE_INTRO + """

Searching for this question is finished. Answer the student's latest message from the
search results above.

HOW TO USE THE RESULTS
- Results arrive as organization records under ORGANIZATION headings. Each record has a
profile half describing what the organization is, and a logistics half holding contact
details, meeting time, leadership, events, and how to join.
- When the records do not cover the question, say so plainly rather than guessing, and
suggest how the student could rephrase.
- Many of these pages are incomplete. When a record says a detail is "not listed on this
organization page", tell the student that instead of supplying a value.
- Never invent an email address, officer name, meeting day, time, room number, website, or
event. A plausible guess is worse than saying the page does not list it.
- Do not rate or rank organizations as good, strong, popular, or better than others. The
pages do not measure that. Describe what each page says and let the student choose.
- Event dates on these pages have no year and come from saved copies of the pages, so they
may already have passed. Present them as events listed on the organization's page, never as
upcoming or current, and suggest checking the SOURCE link for current events.

ATTRIBUTION
- Name the organization you drew each fact from, and include its SOURCE link so the student
can read the full page.
- Attribute every detail to the organization whose record contains it. Never move a meeting
time, officer, or email from one organization to another.
- Describe only the organizations listed under THIS QUESTION below.

COUNTING
- Each search returns a small sample of a much larger directory, chosen by similarity to the
query. It is never the complete set of organizations that fit.
- Never state how many organizations exist or how many match a description, and never imply
the sample is exhaustive, unless THIS QUESTION gives you that total.

STYLE
- Be concise and direct. Short paragraphs, or a short list when several organizations fit.
- When several organizations are relevant, take them one at a time.
- End with a concrete next step when the record supports one, such as the contact email or
the joining instructions.
- Never refer to "the sample", "my results", "the records I have", or what you have "on
hand". When listing some organizations out of a larger total, introduce them plainly, for
example "Here are five of them:".
- Never say you cannot search, cannot look something up again, or cannot access the
directory. Never mention the filter note, the directory scope line, the search tool, or
"the records provided". The student cannot see any of that. Speak about the directory and
the organization pages, and give counts as plain fact: "33 organizations meet on Friday"."""


#HTML Parsing (unchanged from HW4)
def org_slug(file_name):
    stem = Path(file_name).stem

    if stem.startswith(FILENAME_PREFIX):
        stem = stem[len(FILENAME_PREFIX):]

    return stem


def org_url(file_name):
    return ORG_URL_PREFIX + org_slug(file_name)


def read_page(path):
    html = Path(path).read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'html.parser')

    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()

    heading = soup.find('h1')
    name = heading.get_text(strip=True) if heading else org_slug(path).replace('_', ' ').title()

    lines = [re.sub(r'\s+', ' ', line).strip() for line in soup.get_text('\n').split('\n')]
    lines = [line for line in lines if line and line not in NOISE_LINES]

    return name, lines


def split_page_sections(lines):
    marks = [(i, line) for i, line in enumerate(lines) if line in PAGE_SECTIONS]
    sections = {}

    for position, (start, label) in enumerate(marks):
        end = marks[position + 1][0] if position + 1 < len(marks) else len(lines)
        sections.setdefault(label, []).extend(lines[start + 1:end])

    return sections


def is_field_label(line):
    return line.endswith(':') or CONSULTANT_MARKER in line


def normalize_label(line):
    if CONSULTANT_MARKER in line:
        return 'Student Engagement Consultant'

    label = line.rstrip(':').strip()

    return FIELD_LABELS.get(label, label)


def parse_fields(block):
    fields = {}
    i = 0

    while i < len(block):
        if not is_field_label(block[i]):
            i += 1
            continue

        label = normalize_label(block[i])
        j = i + 1

        if j < len(block) and block[j].startswith('Please describe'):
            j += 1

        values = []

        while j < len(block) and not is_field_label(block[j]):
            values.append(block[j])
            j += 1

        value = ' '.join(values).strip()

        if value.lower() not in EMPTY_VALUES:
            fields[label] = value

        i = max(j, i + 1)

    return fields


def parse_contact(block):
    contact = {}
    address = []
    current = None

    for line in block:
        if line == 'Address':
            current = 'Address'
        elif line == 'Contact Email':
            current = 'Email'
        elif line == 'Phone Number':
            current = 'Phone'
        elif line in ('E:', 'P:'):
            continue
        elif current == 'Address':
            address.append(line)
        elif current in ('Email', 'Phone'):
            contact[current] = line
            current = None

    joined = ' '.join(address).replace(' ,', ',').strip()

    if joined and joined.lower() not in EMPTY_VALUES:
        contact['Address'] = joined

    return contact


def parse_officers(block):
    lines = [line for line in block if len(line) > 1]
    officers = []
    i = 0

    while i < len(lines) - 1:
        if lines[i].isupper() and len(lines[i]) > 2:
            officers.append((lines[i], lines[i + 1]))
            i += 2
        else:
            i += 1

    return officers


def parse_events(block, org_name):
    lines = [line for line in block
             if len(line) > 1 and 'no upcoming events' not in line.lower()]
    events = []

    for i, line in enumerate(lines):
        if i == 0 or not EVENT_DATE_PATTERN.match(line):
            continue

        title = lines[i - 1]
        location = ''

        if i + 1 < len(lines) and lines[i + 1] != org_name:
            location = lines[i + 1]

        events.append((title, line, location))

    return events


#Chunking (unchanged from HW4): each page splits into a profile half and a logistics half
def build_description(about_text, fields):
    parts = []

    for text in (about_text.strip(), fields.get('Description', '').strip()):
        if not text:
            continue

        if any(text[:80] in seen or seen[:80] in text for seen in parts):
            continue

        parts.append(text)

    return ' '.join(parts).strip()


def build_profile_chunk(name, url, description, fields):
    lines = [
        f'ORGANIZATION: {name}',
        f'SOURCE: {url}',
        'SECTION: Profile and Description',
        '',
        f'{name} is a registered student organization at Syracuse University.',
    ]

    if description:
        lines.append(f'Description: {description}')
    else:
        lines.append('Description: this organization did not provide a description on its page.')

    if fields.get('Website'):
        lines.append(f'Website: {fields["Website"]}')

    return '\n'.join(lines)


def build_logistics_chunk(name, url, contact, fields, officers, events):
    lines = [
        f'ORGANIZATION: {name}',
        f'SOURCE: {url}',
        'SECTION: Contact, Meetings, Leadership, Events, and How To Join',
        '',
    ]

    if any(contact.get(key) for key in ('Email', 'Phone', 'Address')):
        for key in ('Email', 'Phone', 'Address'):
            if contact.get(key):
                lines.append(f'{key}: {contact[key]}')
    else:
        lines.append('Contact details: none listed on this organization page.')

    day = fields.get('Meeting Day', '')
    clock = fields.get('Meeting Time', '')
    meridiem = fields.get('AM/PM', '') if (day or clock) else ''

    schedule = ' '.join(part for part in (day, clock, meridiem) if part)

    if schedule:
        lines.append(f'Meeting Day and Time: {schedule}')
    else:
        lines.append('Meeting Day and Time: not listed on this organization page.')

    if fields.get('Meeting Location'):
        lines.append(f'Meeting Location: {fields["Meeting Location"]}')

    named = []

    for role in LEADERSHIP_ORDER:
        if fields.get(role):
            lines.append(f'{role}: {fields[role]}')
            named.append(fields[role])

    extras = [f'{role.title()} - {person}' for role, person in officers if person not in named]

    if extras:
        lines.append('Also listed on the officer roster: ' + '; '.join(extras))

    if not named and not extras:
        lines.append('Leadership: no officers listed on this organization page.')

    for title, when, where in events:
        entry = f'Upcoming event: {title} - {when}'

        if where:
            entry += f' - {where}'

        lines.append(entry)

    if fields.get('How To Join'):
        lines.append(f'How To Join: {fields["How To Join"]}')
    else:
        lines.append('How To Join: no joining instructions were provided on this organization page.')

    return '\n'.join(lines)


def build_chunks(path):
    name, lines = read_page(path)
    sections = split_page_sections(lines)

    about = ' '.join(line for line in sections.get('About', [])
                     if line != name and len(line) > 1)

    fields = parse_fields(sections.get('Additional Information', []))
    contact = parse_contact(sections.get('Contact Information', []))
    officers = parse_officers(sections.get('Officers', []))
    events = parse_events(sections.get('Public Events', []), name)

    description = build_description(about, fields)
    slug = org_slug(path)
    url = org_url(path)

    metadata = {
        'organization': name,
        'slug': slug,
        'source_url': url,
        'has_description': bool(description),
        'has_meeting_time': bool(fields.get('Meeting Day')),
        'meeting_day': fields.get('Meeting Day', '') if fields.get('Meeting Day') in WEEKDAY_NAMES else '',
        'has_contact_email': bool(contact.get('Email')),
        'event_count': len(events),
    }

    profile = build_profile_chunk(name, url, description, fields)
    logistics = build_logistics_chunk(name, url, contact, fields, officers, events)

    return [
        (f'{slug}::profile', profile[:MAX_EMBED_CHARS], dict(metadata, section='profile')),
        (f'{slug}::logistics', logistics[:MAX_EMBED_CHARS], dict(metadata, section='logistics')),
    ]


#Embedding and Building the Collection (unchanged from HW4)
#Only runs if HW5 is opened before HW4 has built the collection on this server
def embed_batch(texts):
    client = st.session_state.openai_client
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)

    return [item.embedding for item in response.data]


def load_html_to_collection(folder, collection):
    paths = sorted(folder.rglob('*.html'))
    existing = set(collection.get()['ids'])

    ids, documents, metadatas = [], [], []

    for path in paths:
        for chunk_id, document, metadata in build_chunks(path):
            if chunk_id in existing:
                continue

            ids.append(chunk_id)
            documents.append(document)
            metadatas.append(metadata)

    if not ids:
        return 0

    progress = st.progress(0.0, text='Embedding the student organization pages...')

    for start in range(0, len(ids), EMBED_BATCH_SIZE):
        stop = min(start + EMBED_BATCH_SIZE, len(ids))
        embeddings = embed_batch(documents[start:stop])

        collection.add(
            ids=ids[start:stop],
            documents=documents[start:stop],
            embeddings=embeddings,
            metadatas=metadatas[start:stop]
        )

        progress.progress(stop / len(ids),
                          text=f'Embedded {stop} of {len(ids)} mini-documents...')

    progress.empty()

    return len(ids)


def find_data_folder():
    start = Path(__file__).resolve().parent

    for base in [start, *start.parents]:
        candidate = base / DATA_SUBFOLDER

        if candidate.is_dir():
            return candidate

    return None


def create_hw5_vectordb():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    st.session_state.HW5_ChromaClient = chroma_client #Held

    folder = find_data_folder()

    if folder is None:
        if collection.count() == 0:
            st.error(f'The folder {DATA_SUBFOLDER} with the organization pages was not found. '
                     'Add it to the repository and reload.')
            st.stop()

        return collection

    expected = 2 * len(list(folder.rglob('*.html'))) #Two mini-documents per page

    if collection.count() < expected:
        with st.spinner('Building the vector database (one time only)...'):
            added = load_html_to_collection(folder, collection)

        if added:
            st.success(f'Added {added} mini-documents to {COLLECTION_NAME}.')
        elif collection.count() == 0:
            st.error(f'No organization pages were loaded from {folder}.')

    return collection


#Short-Term Memory (same buffer as HW4)
#The full history stays on screen; only this slice is sent to the model.
def conversation_buffer(messages, interactions=BUFFER_INTERACTIONS):
    buffer = messages[-(interactions * 2):]

    #A raw slice can start on an assistant reply, leaving half an interaction
    while buffer and buffer[0]['role'] != 'user':
        buffer.pop(0)

    #Saved messages also carry caption data; the API only accepts role and content
    return [{'role': message['role'], 'content': message['content']} for message in buffer]


def chunk_body(document):
    parts = document.split('\n\n', 1)

    return parts[1] if len(parts) == 2 else ''


def merge_org_record(record):
    lines = [f"--- ORGANIZATION: {record['organization']} ---",
             f"SOURCE: {record['source_url']}"]

    if record['profile']:
        lines.append('PROFILE AND DESCRIPTION')
        lines.append(chunk_body(record['profile']))

    if record['logistics']:
        lines.append('CONTACT, MEETINGS, LEADERSHIP, EVENTS, AND HOW TO JOIN')
        lines.append(chunk_body(record['logistics']))

    return '\n'.join(lines)


#Step 3a: the Python function behind the tool
#Replaces HW4's detect_filters: the filter values come from the model, not from regex
def build_where(meeting_days=None, has_upcoming_events=None):
    if isinstance(meeting_days, str):
        meeting_days = [meeting_days]

    requested = {str(day).strip().capitalize() for day in (meeting_days or [])}
    days = [day for day in WEEKDAY_NAMES if day in requested] #Drops anything off the enum

    conditions = []
    labels = []

    if days:
        if len(days) == 1:
            conditions.append({'meeting_day': days[0]})
        else:
            conditions.append({'meeting_day': {'$in': days}})

        labels.append('meets on ' + ' or '.join(days))

    if has_upcoming_events is True:
        conditions.append({'event_count': {'$gt': 0}})
        labels.append('lists events') #Not "upcoming": the saved pages' dates may have passed

    if not conditions:
        return None, []

    where = conditions[0] if len(conditions) == 1 else {'$and': conditions}

    return where, labels


def count_matching_orgs(collection, where):
    try:
        found = collection.get(where=where, include=[])
    except Exception:
        found = collection.get(where=where)

    return len(found['ids']) // 2


def relevant_club_info(query, meeting_days=None, has_upcoming_events=None,
                       n_results=N_RESULTS, max_orgs=MAX_ORGS):
    """Searches ChromaDB with the query the model wrote and returns what it found."""
    collection = st.session_state.HW5_VectorDB

    #HW4 ran build_search_query here to guess at follow-ups. The model's query needs no help.
    query_embedding = embed_batch([query])[0]

    where, labels = build_where(meeting_days, has_upcoming_events)
    description = ' and '.join(labels)
    matched = count_matching_orgs(collection, where) if where else 0
    dropped_filter = ''

    if where and matched == 0: #Never return an empty context
        dropped_filter = description
        where, labels, matched, description = None, [], 0, ''

    results = collection.query(query_embeddings=[query_embedding],
                               n_results=n_results,
                               where=where)

    order = []
    records = {}

    for document, metadata in zip(results['documents'][0], results['metadatas'][0]):
        slug = metadata['slug']

        if slug not in records:
            if len(order) >= max_orgs:
                continue

            order.append(slug)
            records[slug] = {
                'organization': metadata['organization'],
                'source_url': metadata['source_url'],
                'profile': '',
                'logistics': ''
            }

        records[slug][metadata['section']] = document

    #Pull the other half of every matched organization
    missing = [f'{slug}::{section}'
               for slug in order
               for section in ('profile', 'logistics')
               if not records[slug][section]]

    if missing:
        siblings = collection.get(ids=missing)

        for document, metadata in zip(siblings['documents'], siblings['metadatas']):
            records[metadata['slug']][metadata['section']] = document

    blocks = [merge_org_record(records[slug]) for slug in order]
    organizations = [(records[slug]['organization'], records[slug]['source_url'])
                     for slug in order]

    context = '\n\n'.join(blocks).replace(EVENT_LINE_STORED, EVENT_LINE_SHOWN)
    total = collection.count() // 2 #Two mini-documents per organization

    if labels:
        note = (f'FILTER NOTE: {matched} of the {total} organizations in this directory '
                f'match "{description}". The {len(order)} records below are a sample of '
                f'those {matched}, not the complete list.')
    elif dropped_filter:
        #New in HW5: HW4 silently dropped a filter with no matches, so the model never knew
        note = (f'FILTER NOTE: no organization page in this directory of {total} lists '
                f'"{dropped_filter}". The {len(order)} records below are the closest matches '
                f'to the query without that filter, and they do not fit it.')
    else:
        note = (f'DIRECTORY SCOPE: this directory holds {total} organizations. The '
                f'{len(order)} records below are only the closest matches to this search. '
                f'They are not the full directory and not every organization that fits, so '
                f'no total can be counted from them.')

    caption_filter = description or (f'{dropped_filter} (no matches, filter dropped)'
                                     if dropped_filter else '')

    return {
        'content': note + '\n\n' + context, #What the model reads in the tool message
        'organizations': organizations, #Feeds the caption and the THIS QUESTION list
        'caption_filter': caption_filter,
        'filter_label': description, #Set only when a filter matched something
        'matched': matched,
        'dropped_filter': dropped_filter, #Set only when a filter matched nothing
    }


#Tool Dispatch
#Reads one tool call the way the lecture does: function.name, then json.loads(arguments)
def run_tool_call(call):
    if call.function.name != 'relevant_club_info':
        return f'Error: function {call.function.name} does not exist.', None

    try:
        arguments = json.loads(call.function.arguments or '{}')
    except json.JSONDecodeError:
        return 'Error: the search arguments could not be read as JSON.', None

    query = str(arguments.get('query', '')).strip()

    if not query:
        return 'Error: the search was called without a query.', None

    result = relevant_club_info(
        query,
        meeting_days=arguments.get('meeting_days'),
        has_upcoming_events=arguments.get('has_upcoming_events'),
    )

    search = {
        'query': query,
        'filter': result['caption_filter'],
        'organizations': result['organizations'],
        'filter_label': result['filter_label'],
        'matched': result['matched'],
        'dropped_filter': result['dropped_filter'],
    }

    return result['content'], search


#Turns a filter label into a plain sentence: "29 organizations in the directory meet on
#Wednesday." The model copied the quoted label word for word when given only the label.
def count_sentence(matched, filter_label):
    if matched == 1:
        phrase = filter_label.replace('lists events', 'lists events on its page')
        return f'1 organization in the directory {phrase}.'

    phrase = (filter_label.replace('meets on', 'meet on')
              .replace('lists events', 'list events on their pages'))

    return f'{matched} organizations in the directory {phrase}.'


#Step 3b: the turn-specific block added to the answer prompt
#The code knows exactly which organizations came back, so it tells the model instead of
#hoping the model sorts this turn's records from names in the conversation memory.
def build_answer_prompt(searches, skipped=0, failed=0):
    names = []

    for search in searches:
        for name, _ in search['organizations']:
            if name not in names:
                names.append(name)

    lines = ['', '', 'THIS QUESTION']

    if names:
        lines.append('- Organizations you may describe in this answer, and no others: '
                     + '; '.join(names) + '.')
        lines.append('- Organizations named earlier in the conversation but missing from that '
                     'list must not be described, even briefly.')
    else:
        lines.append('- No organization records came back for this question. Do not describe '
                     'any organization. Say the directory pages found do not cover the '
                     'question and suggest a clearer way to ask it.')

    for search in searches:
        if search['filter_label']:
            lines.append(f'- Open the answer with this fact, in these words or close to them: '
                         f'"{count_sentence(search["matched"], search["filter_label"])}" The '
                         f'organizations above are only some of those; introduce them plainly, '
                         f'for example "Here are five of them:".')

            if 'lists events' in search['filter_label']:
                lines.append('- The student asked about events. Say the event dates on these '
                             'pages have no year and may have passed, and suggest checking each '
                             'SOURCE link for current events.')
        elif search['dropped_filter']:
            lines.append(f'- No organization page in the directory lists '
                         f'"{search["dropped_filter"]}". Say that first. The records you have '
                         f'are the closest matches without that filter and do not fit it.')

    if skipped or failed:
        lines.append('- Part of the question may have no records above. For that part, say the '
                     'pages found do not cover it and suggest asking about it separately.')

    return '\n'.join(lines)


#The lecture's chat_completion_request utility, expanded:
#  - tools and tool_choice are only sent when given, rather than sent as null
#  - stream is passed through so the second call can stream
#  - errors are raised instead of returned, so the page can show them instead of
#    treating an exception object as a response
def chat_completion_request(messages, tools=None, tool_choice=None, model=CHAT_MODEL,
                            stream=False):
    request = {'model': model, 'messages': messages, 'stream': stream}

    if tools is not None:
        request['tools'] = tools

    if tool_choice is not None:
        request['tool_choice'] = tool_choice

    return st.session_state.openai_client.chat.completions.create(**request)


#Caption under each answer: shows the search the model chose to run
def render_search_caption(searches, skipped=0, failed=0):
    if searches is None: #The greeting, which never went through the model
        return

    if not searches and not failed:
        st.caption('No directory search was needed for this reply.')
        return

    for search in searches:
        text = f'Searched for "{search["query"]}"'

        if search['filter']:
            text += f' with the filter "{search["filter"]}"'

        if search['organizations']:
            links = ', '.join(f'[{name}]({url})' for name, url in search['organizations'])
            text += f'. Found: {links}'

        st.caption(text)

    if failed:
        st.caption(f'{failed} search request(s) from the model could not be run.')

    if skipped:
        st.caption(f'{skipped} more search(es) skipped. The limit is '
                   f'{MAX_SEARCHES_PER_TURN} per question.')


#Main App - Chat Interface
st.title('Chatbot')

st.write('Ask about any of Syracuse University\'s registered student organizations. The '
         'assistant decides when to search the directory and writes its own search, so '
         'follow-up questions like "when do they meet?" find the right organization.')

st.write(f'Memory is a rolling buffer of your last {BUFFER_INTERACTIONS} interactions. '
         'The line under each answer shows what the assistant searched for.')

if 'openai_client' not in st.session_state: #Shared with HW4; same client either way
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

#Reuse HW4's collection object if HW4 already opened it this session; otherwise open it here
if 'HW5_VectorDB' not in st.session_state:
    if st.session_state.get('HW4_VectorDB') is not None:
        st.session_state.HW5_VectorDB = st.session_state.HW4_VectorDB
    else:
        st.session_state.HW5_VectorDB = create_hw5_vectordb()

collection = st.session_state.HW5_VectorDB

#HW5 keys are prefixed so this chat never mixes with HW4's st.session_state.messages
if 'hw5_messages' not in st.session_state:
    st.session_state.hw5_messages = [{'role': 'assistant', 'content': GREETING}]

with st.sidebar:
    st.header(':material/settings: **Settings:**')

    st.subheader('Vector database')
    st.caption(f'Organizations: {collection.count() // 2}')
    st.caption(f'Mini-documents: {collection.count()}')

    st.subheader('Tool')
    st.caption('Function: relevant_club_info')
    st.caption('First call: search prompt, tool_choice="auto" (the model decides whether '
               'to search, ask first, or reply)')
    st.caption('Second call: answer prompt, tool_choice="none" (the model must answer)')
    st.caption(f'Search limit: {MAX_SEARCHES_PER_TURN} per question')

    st.subheader('Conversation')
    st.caption(f'Memory buffer: last {BUFFER_INTERACTIONS} interactions')
    turns_caption = st.empty()
    turns_caption.caption(f'Turns so far: {(len(st.session_state.hw5_messages) - 1) // 2}')

    show_context = st.checkbox('Show retrieved context', value=False)

    if st.button('Clear conversation'):
        st.session_state.hw5_messages = [{'role': 'assistant', 'content': GREETING}]
        st.session_state.pop('hw5_last_context', None)
        st.rerun()

for message in st.session_state.hw5_messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

        if message['role'] == 'assistant':
            render_search_caption(message.get('searches'), message.get('skipped', 0),
                                  message.get('failed', 0))

if prompt := st.chat_input('Ask about Syracuse student organizations...'):
    st.session_state.hw5_messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    messages_to_send = ([{'role': 'system', 'content': SEARCH_SYSTEM_PROMPT}]
                        + conversation_buffer(st.session_state.hw5_messages))

    searches = []
    skipped = 0
    failed = 0
    tool_results = []
    answer_block = ''

    try:
        #Call 1 (Step 3a): the model sees the tool and decides whether to use it
        with st.spinner('Thinking...'):
            first_response = chat_completion_request(messages_to_send, tools=TOOLS,
                                                     tool_choice='auto')

        reply = first_response.choices[0].message
        tool_calls = reply.tool_calls

        if tool_calls:
            #The assistant's tool request goes in first; each tool message must answer an
            #id listed here, as the lecture notes ("role 'tool' must be a response to a
            #preceding message with 'tool_calls'")
            messages_to_send.append({
                'role': 'assistant',
                'content': reply.content,
                'tool_calls': [
                    {'id': call.id,
                     'type': 'function',
                     'function': {'name': call.function.name,
                                  'arguments': call.function.arguments}}
                    for call in tool_calls
                ],
            })

            with st.spinner('Searching the organization directory...'):
                for index, call in enumerate(tool_calls):
                    if index >= MAX_SEARCHES_PER_TURN:
                        content = ('Skipped: the search limit for one question was reached. '
                                   'Answer from the results already returned.')
                        skipped += 1
                    else:
                        content, search = run_tool_call(call)

                        if search:
                            searches.append(search)
                            tool_results.append(content)
                        else:
                            failed += 1 #The error text still goes back to the model

                    #Every tool call id gets exactly one tool message, or the API rejects
                    #the second call
                    messages_to_send.append({
                        'role': 'tool',
                        'tool_call_id': call.id,
                        'content': content,
                    })

            #Call 2 (Step 3b): swap the search prompt for the answer prompt plus this turn's
            #THIS QUESTION block. Only message 0 changes, so the tool request and tool
            #results stay in the order the API requires.
            answer_block = build_answer_prompt(searches, skipped, failed)
            messages_to_send[0] = {'role': 'system',
                                   'content': ANSWER_SYSTEM_PROMPT + answer_block}

            #Same tools, but tool_choice='none' removes the option to search again, so the
            #model has to answer from the results
            stream = chat_completion_request(messages_to_send, tools=TOOLS,
                                             tool_choice='none', stream=True)

            with st.chat_message('assistant'):
                response = st.write_stream(stream)

                #write_stream returns a list if any chunk was not text; memory needs a string
                if not isinstance(response, str):
                    response = ''.join(str(part) for part in response)

                render_search_caption(searches, skipped, failed)

        else:
            #tool_choice='auto' let the model answer directly (thanks, greetings, or a
            #clarifying question such as "what are you interested in?"), so its first reply
            #is the answer
            response = reply.content or ('No answer came back. Rephrase the question '
                                         'and try again.')

            with st.chat_message('assistant'):
                st.markdown(response)
                render_search_caption(searches)

    except Exception as error:
        st.session_state.hw5_messages.pop() #Drop the failed turn
        st.error(f'The request failed: {error}')
        st.stop()

    st.session_state.hw5_messages.append({
        'role': 'assistant',
        'content': response,
        'searches': searches,
        'skipped': skipped,
        'failed': failed,
    })

    st.session_state.hw5_last_context = '\n\n=====\n\n'.join(
        ([answer_block.strip()] if answer_block else []) + tool_results)

turns_caption.caption(f'Turns so far: {(len(st.session_state.hw5_messages) - 1) // 2}')

if show_context:
    with st.expander('Context sent to the model for the last question'):
        st.text(st.session_state.get('hw5_last_context') or 'No search was run.')
