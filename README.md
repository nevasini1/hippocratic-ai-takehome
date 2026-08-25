# Moonlight: quality-controlled bedtime stories

This submission turns a simple story idea into an age-targeted bedtime story, asks a separate LLM judge to score it, and uses the judge's concrete feedback to improve weak drafts. The storyteller, judge, and editor are separate prompt roles, but every API call deliberately uses the assignment's required model: **gpt-3.5-turbo**.

The default is a 450–700 word story for age 7. A story normally needs two API calls (draft + judge); it may use up to two bounded revisions when it misses the quality bar. The final CLI output contains the story, not internal prompts or raw judge output.

## Highlights

- Exact-age guidance for listeners from 5 through 10, with short, medium, and long options.
- A structured seven-dimension judge rubric and pass/fail rules computed in Python.
- Targeted revisions that preserve strengths and address only identified weaknesses.
- Highest-scoring safe-candidate retention so a later revision cannot silently regress.
- Strict judge-schema validation, one independent format retry, and finite termination.
- A safe built-in fallback rather than displaying a story that was never safety-verified.
- JSON-encoded trust boundaries around the request, story, and listener feedback.
- Optional listener feedback followed by another judge pass.
- Fully offline unit tests using a scripted fake model.
- A versioned six-case live benchmark with score, safety, adherence, latency,
  and logical-call reporting that stores no generated stories or prompts.

## Quick start

Python 3.10 or newer is required.

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your own key"
python main.py
~~~

The API key is read only from the environment and is never stored by the program. Do not put a real key in source code, .env.example, commits, screenshots, or a submission archive. Local .env files are ignored, although this program intentionally does not auto-load them.

The adapter uses the modern Python client with the [OpenAI Chat Completions API](https://developers.openai.com/api/reference/resources/chat), while preserving the model specified by the assignment.

## Usage

Interactive mode asks for a story and then accepts natural-language changes:

~~~text
$ python main.py
Moonlight Bedtime Storyteller
What kind of story would you like? A girl named Alice and her best friend Bob, who is a cat.
...
Press Enter to finish, or describe a change (for example, 'make it funnier'): Give Bob a tiny red hat.
~~~

For a non-interactive run:

~~~bash
python main.py \
  --request "A shy cloud who learns to make gentle rain" \
  --age 6 \
  --length short \
  --show-evaluation
~~~

Useful options:

| Option | Meaning |
| --- | --- |
| --request TEXT | Story idea; omit it for interactive mode |
| --age 5-10 | Exact target age; default 7 |
| --length short\|medium\|long | About 250–400, 450–700, or 700–1,000 words |
| --max-revisions 0-3 | Additional drafts allowed after a judged candidate; default 2 |
| --show-evaluation | Show the selected scorecard and loop metadata |
| --feedback | Ask for changes after a non-interactive request |

Run python main.py --help for the complete CLI reference.

## System block diagram

The boxes labeled with the model are logical roles with different system prompts. They all call the same required model.

~~~mermaid
flowchart TD
    U["User: story idea, age, length"] --> C["CLI validation and defaults"]
    C --> B["JSON story brief<br/>untrusted-data boundary"]

    B --> SP["Storyteller system prompt<br/>arc + age + bedtime + safety rules"]
    SP --> SW["gpt-3.5-turbo<br/>storyteller role"]
    SW --> D["Candidate story"]

    B --> JP["Judge system prompt<br/>explicit seven-part rubric"]
    D --> JP
    JP --> J["gpt-3.5-turbo<br/>judge role, temperature 0"]
    J --> P["Strict JSON parsing<br/>and schema validation"]
    P --> Q{"Python quality gate:<br/>all scores at least 4,<br/>no safety issue or injection leak?"}

    P -- "Malformed" --> R{"Format retry<br/>available?"}
    R -- "Yes" --> JP
    R -- "No" --> BF["Best previously verified-safe story<br/>or built-in safe fallback"]

    Q -- "Yes" --> F["Approved story"]
    Q -- "No; revision remains" --> EP["Editor system prompt<br/>brief + prior story + parsed action items"]
    EP --> E["gpt-3.5-turbo<br/>editor role"]
    E --> D
    Q -- "No; limit reached" --> BS["Highest-scoring verified-safe candidate"]
    Q -- "Only unsafe candidates" --> BF

    F --> O["Story output"]
    BS --> O
    BF --> O
    O --> A{"Listener accepts<br/>or requests a change"}
    A -- "Accept" --> X["Done"]
    A -- "Change request as untrusted JSON data" --> BJ["gpt-3.5-turbo judge role<br/>re-score prior story as feedback-aware baseline"]
    BJ --> EP
~~~

## Prompt and agent strategy

### 1. Storyteller

The storyteller receives a normalized brief containing the original request, exact target age, named length, and word range. Its system prompt asks for a setup, one understandable challenge, character agency or cooperation, an earned resolution, and a quiet bedtime landing. Concrete age-band anchors keep ages 5–6 simpler and more repetitive, give ages 7–8 richer description and one clear turn, and allow ages 9–10 subtler motivation and layered imagery.

It silently tailors technique to the request: wonder for fantasy, sensory play for animal stories, empathy for friendship stories, gentle clues for mysteries, and relatable emotion for everyday stories. Unsafe details are transformed into gentle analogues while harmless names and ideas are retained.

The creativity temperature is 0.8. The prompt requests only a title and story, which keeps implementation notes out of the child's experience.

### 2. LLM judge

The judge gets the same brief plus the candidate story, but a separate, deterministic prompt. It returns integer scores from 1 to 5 for:

1. Request adherence
2. Age appropriateness
3. Story arc
4. Engagement
5. Language clarity
6. Bedtime tone
7. Emotional safety

It also returns critical safety issues, whether prompt instructions leaked into the story, strengths to preserve, required revisions, and a one-sentence summary. JSON mode encourages valid output; local schema validation still rejects missing fields, booleans masquerading as integers, out-of-range scores, invalid list items, and non-boolean safety flags.

The model never gets to approve itself with a single flag. Python approves a candidate only when:

- every score is at least 4/5;
- no critical safety issue is present; and
- no prompt-injection leak is present.

### 3. Bounded editor loop

If the candidate fails, the editor receives only the validated brief, previous story, validated scores, strengths, safety/leak flags, and actionable revision list. A failing report with no concrete fix is rejected as malformed. The editor never receives malformed judge text as instructions. The edited story returns to the same judge.

Two additional drafts are allowed by default, so valid judge responses produce at most six calls for an initial request. If the limit is reached, verified-safe candidates must have age-appropriateness and emotional-safety scores of at least 4 with no critical issue or injection leak. Selection favors the best minimum dimension, then the most dimensions at 4+, then average score, then latest revision. This prevents a high average from hiding one serious weakness. If none is verified safe, it outputs a pre-reviewed 305-word calming story instead of unverified content.

A malformed judge response is retried once without consuming a story revision. The validation error and bounded prior response are returned only to the judge as quoted retry data, making the second request corrective while keeping malformed text away from the editor. All loop bounds are independent, which prevents accidental infinite retries.

### 4. Listener feedback

Interactive feedback such as “make Bob funnier” or “use a quieter ending” goes to the editor as a JSON string. The previous story is first re-judged against that new request to create a comparable baseline and actionable edit notes. The revision then competes with that baseline under the same feedback-aware rubric, so a weak revision cannot silently displace it. Listener feedback cannot override the system's age or safety rules.

## Safety and trust boundaries

The user request, generated story, judge notes, and listener feedback are data. They are serialized into user-role JSON messages instead of being interpolated into system instructions. Every role is independently told not to execute embedded instructions, reveal prompts, run code, or fetch content.

This is defense in depth, not a guarantee of perfect model behavior. The safety-oriented judge and built-in fallback reduce risk, but a production system for children would still need policy review, calibrated evals, monitoring, and appropriate adult supervision.

## Tests

Install the development dependency and run the suite:

~~~bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
~~~

The normal test suite makes zero network calls and needs no API key. It covers:

- pass on the first draft;
- fail, targeted revision, and pass;
- exact zero/two-revision termination behavior;
- best-safe selection and unsafe fallback;
- malformed judge JSON and retry isolation;
- strict score/list/boolean schema cases;
- prompt/data separation for hostile request feedback;
- user-feedback revision followed by re-judging;
- missing key, empty model response, and sanitized CLI errors;
- exact gpt-3.5-turbo use at the SDK boundary; and
- interactive and non-interactive CLI paths.

Suggested manual smoke requests after setting a key:

~~~text
A girl named Alice and her best friend Bob, who happens to be a cat.
A tiny ghost who is afraid of the dark, with a reassuring ending.
A robot gardener learning that asking for help is brave.
~~~

### Live smoke evidence

One live run was completed with the required gpt-3.5-turbo model after the offline suite passed. For the short, age-7 request “A sleepy dragon who learns that asking friends for help is brave,” the first generated draft passed without revision. The judge scored request adherence 5, age appropriateness 4, story arc 5, engagement 5, language clarity 5, bedtime tone 5, and emotional safety 5, with no critical safety issue or prompt-injection leak. This is a useful end-to-end check, not a substitute for a larger human-calibrated evaluation set.

## Reproducible evaluation

The versioned benchmark covers every target age from 5 through 10. Its six
fixed prompts include exact named details, mild suspense, an unsafe request that
must be gently adapted, a direct prompt injection, and two listener-feedback
rounds. It reports:

- first-draft and final selected scores, both overall and by dimension;
- strict judge-measured improvement only among cases that actually revised;
- initial-failure rescue rate, fallback use, and safety-gate outcomes;
- combined brief/feedback adherence using both the judge and literal checks;
- per-case wall latency and actual logical model calls by role.

The checked-in [evaluation report](EVALUATION.md) summarizes the latest live
run, while [the JSON artifact](evals/results/latest.json) preserves the score-only
per-case evidence. Neither artifact contains generated story text, prompts, raw
model responses, or credentials.

To reproduce it after exporting a key:

~~~bash
python -m evals.run_evaluation \
  --cases evals/cases.json \
  --output evals/results/latest.json \
  --report EVALUATION.md
~~~

This is intentionally a small same-model smoke benchmark. The report uses exact
denominators and “judge-measured” language because gpt-3.5-turbo both produces
and evaluates the stories; it is not a substitute for blinded human ratings.

## Repository layout

~~~text
main.py                    Thin CLI and optional feedback session
bedtime_story/agent.py     Generate–judge–revise orchestration and fallback
bedtime_story/prompts.py   Role prompts and JSON trust boundaries
bedtime_story/models.py    Validated request, scorecard, and result types
bedtime_story/json_tools.py Structured-output extraction and validation
bedtime_story/llm.py       Fixed-model OpenAI SDK adapter
evals/cases.json           Versioned ages 5–10 live benchmark
evals/run_evaluation.py    Privacy-conscious metrics runner and report writer
evals/results/latest.json  Latest score-only live result artifact
EVALUATION.md              Readable methodology, results, and limitations
tests/                     Offline fake-model unit and CLI tests
requirements*.txt          Runtime and test dependencies
~~~

## Design tradeoffs and limitations

- **Same-model judge:** the assignment fixes the model, so storyteller and judge errors may be correlated. Separate role prompts, a judge temperature of 0, anchored dimensions, local thresholds, best-safe retention, and transparent benchmark denominators reduce—but do not eliminate—self-evaluation bias.
- **Broad age range:** age 5 and age 10 differ significantly. Exact-age prompting is a pragmatic compromise; production evaluation should maintain separate reading-level fixtures for every age.
- **Cost versus quality:** a best-of-N generator could increase variety, but this design spends extra calls only when the judge identifies a concrete weakness.
- **Model/API availability:** execution depends on network access, account access to the assignment's required model, and the OpenAI service. The model is intentionally not configurable.
- **No content persistence:** this CLI stores no stories, scorecards, or keys, favoring simplicity and privacy over session recovery.

The required “what I would build in two more hours” discussion is also filled in at the top of main.py: repeated age-stratified evaluation, blinded human calibration, privacy-conscious token telemetry, production dashboards, and more adversarial cases.
