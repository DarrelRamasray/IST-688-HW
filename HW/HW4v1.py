import streamlit as st
from openai import OpenAI
import sys
import re
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup

try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

COLLECTION_NAME = 'HW4Collection'
EMBEDDING_MODEL = 'text-embedding-3-small'
CHROMA_PATH = './ChromaDB_for_HW4'
DATA_SUBFOLDER = Path('data') / 'HW04'

REBUILD_COLLECTION = False

EMBED_BATCH_SIZE = 100
MAX_EMBED_CHARS = 20000

CHAT_MODEL = 'gpt-5.4-mini'
N_RESULTS = 8
MAX_ORGS = 5
BUFFER_INTERACTIONS = 5

FILENAME_PREFIX = 'syracuse.campuslabs.com_engage_organization_'
ORG_URL_PREFIX = 'https://syracuse.campuslabs.com/engage/organization/'

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

FOLLOW_UP_PATTERN = re.compile(
    r'\b(they|them|their|theirs|it|its|that one|this one|those|these|'
    r'the club|the org|the organization|the group|there)\b', re.IGNORECASE)

GREETING = ('Ask me about Syracuse University student organizations. I can look up what a '
            'group does, when and where it meets, who runs it, and how to join.')

RAG_SYSTEM_PROMPT = """You are an assistant that helps Syracuse University students find and
join registered student organizations. You answer from organization pages taken from the
university's 'Cuse Activities directory.

HOW TO USE THE ORGANIZATION CONTEXT
- Organization records are provided below under ORGANIZATION headings. They were retrieved
from a vector database by matching the student's question.
- Each record has a profile half describing what the organization is, and a logistics half
holding contact details, meeting time, leadership, events, and how to join.
- Answer from these records. When they do not cover the question, say so plainly rather
than guessing, and suggest how the student could rephrase.
- Many of these pages are incomplete. When a record says a detail is "not listed on this
organization page", tell the student that instead of supplying a value.
- Never invent an email address, officer name, meeting day, time, room number, website, or
event. A plausible guess is worse than saying the page does not list it.

ATTRIBUTION
- Name the organization you drew each fact from, and include its SOURCE link so the student
can read the full page.
- Attribute every detail to the organization whose record contains it. Never move a meeting
time, officer, or email from one organization to another.
- Do not describe an organization that has no record in the context, even if it came up
earlier in the conversation.

STYLE
- Be concise and direct. Short paragraphs, or a short list when several organizations fit.
- When several organizations are relevant, take them one at a time.
- End with a concrete next step when the record supports one, such as the contact email or
the joining instructions."""


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

    schedule = ' '.join(fields[key] for key in ('Meeting Day', 'Meeting Time', 'AM/PM')
                        if fields.get(key))

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
        'has_contact_email': bool(contact.get('Email')),
        'event_count': len(events),
    }

    profile = build_profile_chunk(name, url, description, fields)
    logistics = build_logistics_chunk(name, url, contact, fields, officers, events)

    return [
        (f'{slug}::profile', profile[:MAX_EMBED_CHARS], dict(metadata, section='profile')),
        (f'{slug}::logistics', logistics[:MAX_EMBED_CHARS], dict(metadata, section='logistics')),
    ]


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


def create_hw4_vectordb():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if REBUILD_COLLECTION:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    st.session_state.HW4_ChromaClient = chroma_client

    folder = find_data_folder()

    if folder is None:
        if collection.count() == 0:
            st.error(f'Could not find the folder {DATA_SUBFOLDER} containing the organization HTML pages.')
            st.stop()

        return collection

    expected = 2 * len(list(folder.rglob('*.html')))

    if collection.count() < expected:
        with st.spinner('Building the vector database (one time only)...'):
            added = load_html_to_collection(folder, collection)

        if added:
            st.success(f'Added {added} mini-documents to {COLLECTION_NAME}.')
        elif collection.count() == 0:
            st.error(f'No organization pages were loaded from {folder}.')

    return collection
def conversation_buffer(messages, interactions=BUFFER_INTERACTIONS):
    return messages[-(interactions * 2):]


def build_search_query(question):
    recent = st.session_state.get('last_organizations', [])

    if not recent:
        return question

    if FOLLOW_UP_PATTERN.search(question) or len(question.split()) <= 5:
        return question + ' ' + ' '.join(name for name, _ in recent)

    return question


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


def get_info_from_vectorDB(collection, question, n_results=N_RESULTS, max_orgs=MAX_ORGS):
    search_query = build_search_query(question)
    query_embedding = embed_batch([search_query])[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

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

    return '\n\n'.join(blocks), organizations


st.set_page_config(page_title='HW4 - SU Student Organizations', page_icon='\N{TANGERINE}')

st.title(":orange[HW 4:] :grey[Syracuse] Student Organization Chatbot")

if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets.OPENAI_API_KEY)

if 'HW4_VectorDB' not in st.session_state:
    st.session_state.HW4_VectorDB = create_hw4_vectordb()

collection = st.session_state.HW4_VectorDB

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 'content': GREETING}]

with st.sidebar:
    st.subheader(":material/settings: **Vector Database:**")
    #st.subheader('Vector database')
    st.caption(f'Organizations: {collection.count() // 2}')
    st.caption(f'Mini-documents: {collection.count()}')

    st.subheader(":material/settings: **Conversation:**")
    #st.subheader('Conversation')
    st.caption(f'Memory buffer: last {BUFFER_INTERACTIONS} interactions')
    st.caption(f'Turns so far: {len(st.session_state.messages) // 2}')

    show_context = st.checkbox('Show retrieved context', value=False)

    if st.button('Clear conversation'):
        st.session_state.messages = [{'role': 'assistant', 'content': GREETING}]
        st.session_state.pop('last_organizations', None)
        st.session_state.pop('last_context', None)
        st.rerun()

for message in st.session_state.messages:
    st.chat_message(message['role']).write(message['content'])

if prompt := st.chat_input('Ask about Syracuse student organizations...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    client = st.session_state.openai_client

    with st.spinner('Searching the organization directory...'):
        context, organizations = get_info_from_vectorDB(collection, prompt)

    system_message = {
        'role': 'system',
        'content': RAG_SYSTEM_PROMPT + '\n\nORGANIZATION CONTEXT\n' + context
    }

    messages_to_send = [system_message] + conversation_buffer(st.session_state.messages)

    st.session_state.last_organizations = organizations
    st.session_state.last_context = context

    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_to_send,
            stream=True,
        )

        with st.chat_message('assistant'):
            response = st.write_stream(stream)

    except Exception as error:
        st.session_state.messages.pop()
        st.error(f'This request has failed: {error}')
        st.stop()

    st.session_state.messages.append({'role': 'assistant', 'content': response})

if st.session_state.get('last_organizations'):
    links = ' | '.join(f'[{name}]({url})' for name, url in st.session_state.last_organizations)
    st.caption('Organizations retrieved for the last question: ' + links)

    if show_context:
        with st.expander('Context sent to the model'):
            st.text(st.session_state.last_context)
