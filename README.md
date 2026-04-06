# Multi-Agent Research System with HITL

## Опис проєкту

Цей проєкт — мультиагентна дослідницька система, побудована за патерном:

**Plan → Research → Critique → Save Report**

Система приймає запит користувача в REPL-інтерфейсі, створює план дослідження, виконує пошук у локальній базі знань і/або в інтернеті, критично перевіряє результати, а потім пропонує зберегти фінальний звіт через механізм **Human-in-the-Loop (HITL)**.

Проєкт реалізовано на основі:
- `langchain`
- `langgraph`
- `langchain-openai`
- `FAISS`
- `BM25`
- `CrossEncoder reranking`
- `DDGS`
- `trafilatura`
- `httpx`

---

## Основна ідея архітектури

У системі є 4 ролі:

1. **Supervisor**
   - координує всі етапи
   - вирішує, коли викликати Planner, Researcher, Critic і save_report
   - керує циклом доопрацювання

2. **Planner**
   - аналізує запит користувача
   - визначає, які джерела треба використати:
     - лише локальну базу знань
     - лише веб
     - або обидва джерела
   - повертає структурований `ResearchPlan`

3. **Researcher**
   - виконує саме дослідження
   - користується інструментами пошуку
   - збирає матеріал і формує findings у markdown

4. **Critic**
   - оцінює якість findings
   - перевіряє повноту, структуру, актуальність або groundedness
   - повертає структурований `CritiqueResult`
   - може вимагати доопрацювання (`REVISE`) або погодити результат (`APPROVE`)

Після схвалення Critic:
- Supervisor формує фінальний звіт
- викликає `save_report`
- користувач через HITL обирає:
  - `approve`
  - `edit`
  - `reject`

---

## Потік виконання

```text
User
  ↓
Supervisor
  ↓
Planner
  ↓
Researcher
  ↓
Critic
  ├─ APPROVE → save_report → HITL → file saved
  └─ REVISE  → Researcher → Critic → ...
```

---

## Структура проєкту

```text
multy-agents_L8/
├── main.py
├── supervisor.py
├── config.py
├── schemas.py
├── tools.py
├── ingest.py
├── retriever.py
├── requirements.txt
├── data/
├── vector_store/
├── output/
└── agents/
    ├── __init__.py
    ├── planner.py
    ├── research.py
    └── critic.py
```

---

## Опис модулів

## 1. `config.py`

Модуль містить:
- налаштування застосунку через `pydantic_settings`
- назви моделей
- параметри пошуку та RAG
- system prompts для всіх агентів

### Основні параметри `Settings`

- `api_key` — OpenAI API key
- `model_name` — модель LLM, за замовчуванням `gpt-4o-mini`
- `embedding_model` — модель для ембеддингів
- `max_search_results = 3` — скільки результатів повертає один `web_search`
- `max_url_content_length = 12000` — ліміт тексту з одного URL
- `data_dir = "data"` — папка з файлами для ingestion
- `vector_store_dir = "vector_store"` — папка з індексом
- `chunk_size = 1000`
- `chunk_overlap = 200`
- `semantic_k = 8`
- `bm25_k = 8`
- `final_k = 5`
- `reranker_model = "BAAI/bge-reranker-base"`
- `max_iterations = 6`
- `max_revision_rounds = 2`

### Prompts

У файлі окремо винесено prompts для:
- `PLANNER_SYSTEM_PROMPT`
- `RESEARCH_SYSTEM_PROMPT`
- `CRITIC_SYSTEM_PROMPT`
- `SUPERVISOR_SYSTEM_PROMPT`

Це дозволяє змінювати поведінку агентів без переписування основної логіки.

---

## 2. `schemas.py`

Містить Pydantic-схеми.

### `ResearchPlan`
Структурований план дослідження:
- `goal`
- `search_queries`
- `sources_to_check`
- `output_format`

### `CritiqueResult`
Структурований результат критики:
- `verdict`
- `is_fresh`
- `is_complete`
- `is_well_structured`
- `strengths`
- `gaps`
- `revision_requests`

### `LocalCritiqueResult`
Додаткова схема для локальних задач.
На поточному етапі основна робоча схема Critic — це `CritiqueResult`, але `LocalCritiqueResult` залишено як окрему схему для можливої подальшої еволюції системи.

---

## 3. `tools.py`

Містить усі інструменти, які використовують агенти.

### `web_search`
Шукає інформацію в інтернеті через `DDGS` і повертає:
- title
- url
- snippet

### `read_url`
Зчитує основний текст веб-сторінки через:
- `trafilatura`
- запасний варіант через `httpx`

### `knowledge_search`
Шукає інформацію в локальній базі знань через `retriever.py`

### `save_report`
Записує markdown-звіт у папку `output/`

### `write_report`
Сумісний alias для `save_report`

---

## 4. `ingest.py`

Це пайплайн побудови локальної бази знань.

### Що робить:
1. Читає файли з `data/`
   - `.pdf`
   - `.txt`
   - `.md`
2. Розбиває їх на chunks
3. Створює embeddings
4. Будує FAISS-індекс
5. Зберігає chunks для BM25

### Результат:
- `vector_store/` — FAISS
- `vector_store/chunks.json` — текстові chunks для BM25

### Запуск:
```bash
python ingest.py
```

---

## 5. `retriever.py`

Реалізує **hybrid retrieval**.

### Компоненти пошуку:
1. **Semantic search**
   - через `FAISS`
2. **BM25 search**
   - через `BM25Retriever`
3. **Deduplication**
4. **Reranking**
   - через `CrossEncoder`

### Потік:
```text
Query
  → FAISS
  → BM25
  → merge
  → deduplicate
  → rerank
  → top final_k documents
```

### Головна функція:
- `search_knowledge_base(query: str) -> str`

---

## 6. `agents/planner.py`

Створює Planner-агента.

### Використовує:
- `ChatOpenAI`
- `create_agent`
- tools:
  - `web_search`
  - `knowledge_search`

### Особливості:
- використовує `response_format=ResearchPlan`
- повертає валідований structured output
- логування внутрішніх tool calls в термінал

### Основна функція:
- `plan(request: str) -> str`

---

## 7. `agents/research.py`

Створює Researcher-агента.

### Використовує tools:
- `web_search`
- `read_url`
- `knowledge_search`

### Основна функція:
- `research(request: str) -> str`

### Особливості:
- виконує пошук по плану або по revision feedback
- повертає markdown findings
- логування tool calls і tool results у термінал

### Обмеження
Researcher має інструкції:
- не запускати автоматично всі search queries
- починати з 1–2 найрелевантніших
- для web-only задач використовувати до 2 `web_search` перед synthesis
- читати до 2 URL за раунд
- на revision rounds шукати лише те, чого бракує

---

## 8. `agents/critic.py`

Створює Critic-агента/верифікатор.


### Допоміжні функції:
- `extract_keywords`
- `kb_result_looks_relevant`
- `build_verification_query`

### Режими роботи

#### A. Local-only critique
Якщо задача стосується лише локальної бази знань:
- робиться один `knowledge_search`
- зовнішні джерела не використовуються
- structured critic оцінює:
  - groundedness
  - coverage
  - structure

#### B. External / mixed critique
Якщо задача зовнішня або змішана:
- будується `verification_query`
- робиться один `knowledge_search`
- перевіряється релевантність KB
- якщо KB нерелевантна, вона ігнорується
- виконується один `web_search`
- structured critic повертає `CritiqueResult`

### Основна функція:
- `critique(request) -> str`

---

## 9. `supervisor.py`

Створює Supervisor-агента.

### Tools:
- `plan`
- `research`
- `critique`
- `save_report`

### Middleware:
- `HumanInTheLoopMiddleware(interrupt_on={"save_report": True})`

### Checkpointer:
- `InMemorySaver()`

### Призначення:
- оркестрація повного циклу
- запуск HITL перед `save_report`

### Важливий нюанс
Supervisor зараз побудований через `create_agent(...)`, тому частина логіки циклу покладається на:
- system prompt
- поведінку LLM

Це означає, що обмеження на число revise-rounds є частково інструкцією, а не повністю жорстким циклом у Python-коді.

---

## 10. `main.py`

Це REPL-інтерфейс користувача.

### Основні функції

#### `new_thread_id()`
Створює новий `thread_id` для кожного нового запиту.

Це потрібно, щоб окремі запити не змішувалися між собою у checkpointer state.

#### `run_supervisor(user_input)`
Запускає Supervisor для нового запиту.

#### `resume_supervisor(decision_payload)`
Відновлює виконання після HITL interrupt.

#### `print_interrupt(...)`
Показує:
- tool
- args
- preview content

#### `handle_interrupt(interrupt_value)`
Обробляє:
- `approve`
- `edit`
- `reject`

### Як працює HITL

#### approve
Дозволяє `save_report`

#### reject
Скасовує запис

#### edit
Не запускає повний новий цикл редагування агентом, а:
- бере початковий `content`
- додає в кінець блок:

```markdown
## User requested changes
...
```

- і знову передає оновлений `save_report`

### Логіка revision_count
У `main.py` є:
- `revision_count`
- `max_revisions = settings.max_revision_rounds`

Але це не повний глобальний контроль усього циклу Supervisor.
Це лише допоміжний захист усередині логіки resume/interrupt.

---

## 11. `requirements.txt`

Список залежностей:

- `langchain`
- `langgraph`
- `langchain-openai`
- `ddgs`
- `trafilatura`
- `httpx`
- `pydantic`
- `pydantic-settings`
- `langchain-community`
- `faiss-cpu`
- `tiktoken`
- `python-dotenv`
- `pypdf`
- `langchain-text-splitters`
- `rank-bm25`
- `sentence-transformers`

---

## Як працює система крок за кроком

### 1. User вводить запит
Наприклад:
```text
який буде курс євро до долара в 2026 році
```

### 2. `main.py`
- створює новий `thread_id`
- запускає `run_supervisor()`

### 3. Supervisor викликає `plan(...)`

### 4. Planner повертає `ResearchPlan`
Приклад:
- goal
- search_queries
- sources_to_check
- output_format

### 5. Supervisor викликає `research(...)`
Передає Researcher не просто короткий пошуковий запит, а ширшу інструкцію на основі plan.

### 6. Researcher
- викликає `web_search`, `read_url`, `knowledge_search`
- формує findings у markdown

### 7. Supervisor викликає `critique(...)`
Передає:
- original request
- plan
- findings

### 8. Critic
- оцінює findings
- повертає `APPROVE` або `REVISE`

### 9A. Якщо `REVISE`
Supervisor знову викликає `research(...)`

### 9B. Якщо `APPROVE`
Supervisor викликає `save_report(...)`

### 10. Middleware зупиняє запис
Користувач бачить:
- filename
- content preview
- approve / edit / reject

### 11. Після `approve`
Файл зберігається в `output/`

---

## Скільки разів агент може зробити певну дію

Нижче важливо розрізняти:
- **жорсткий ліміт у коді**
- **м’яке обмеження через prompt**

## Planner
### Може:
- `web_search`
- `knowledge_search`

### Жорсткий ліміт:
- немає

### М’який ліміт:
- Planner має робити light exploratory research лише за потреби

---

## Researcher
### Може:
- `web_search`
- `read_url`
- `knowledge_search`

### Жорсткі ліміти:
- один `web_search` повертає не більше **3 результатів**
- один `read_url` читає один URL

### М’які ліміти:
- починати з 1–2 пошукових запитів
- у web-only задачах — до 2 `web_search` за раунд
- до 2 `read_url` за раунд
- не повторювати подібні пошуки без потреби
- revision rounds мають бути вузькими

---

## Critic
### Local-only
- 1 `knowledge_search`
- 0 `web_search`
- 0 `read_url`

### External / mixed
- 1 `knowledge_search`
- 1 `web_search`
- 0 `read_url`

### Structured evaluation
- 1 виклик `structured_critic_llm`

Тут обмеження вже набагато більш жорсткі та контрольовані кодом.

---

## Supervisor
### Може викликати:
- `plan`
- `research`
- `critique`
- `save_report`

### Жорсткий ліміт:
- `save_report` завжди з HITL interrupt

### М’який ліміт:
- максимум `settings.max_revision_rounds = 2`

Увага: на поточному етапі це радше правило prompt-оркестрації, ніж повністю жорсткий цикл на Python.

---

## Main / HITL
### Користувач може зробити:
- `approve`
- `edit`
- `reject`

### approve
Файл зберігається

### reject
Запис скасовується

### edit
Контент модифікується і знову подається на підтвердження

---

## Як підготувати локальну базу знань

1. Покласти файли в папку `data/`
2. Запустити:

```bash
python ingest.py
```

3. Після цього з’являться:
- `vector_store/`
- `vector_store/chunks.json`

---

## Як запускати програму

```bash
python main.py
```

---

## Приклад сценарію роботи

1. Користувач ставить питання
2. Planner робить план
3. Researcher виконує дослідження
4. Critic перевіряє
5. Якщо треба — запускається доопрацювання
6. Якщо все добре — Supervisor пропонує зберегти звіт
7. Користувач підтверджує або редагує
8. Файл зберігається

---

## Поточні сильні сторони системи

- Чітке розділення ролей між агентами
- Structured output у Planner і Critic
- Підтримка локальної бази знань
- Hybrid retrieval
- HITL перед записом
- Прозорий лог у терміналі
- Новий `thread_id` для кожного нового user request

---
