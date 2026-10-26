# Method

How hermes-evals turns runs into numbers, how to calibrate a model-based evaluator, and how to tag failures.

## Pass rates and confidence intervals

A run passes when every evaluator on its case passes and the run finished without an error. A run that errors
after the model started answering, such as hitting its time budget, counts as a fail and stays in the
denominator. A run where the provider rejected the request before any output is a setup error. It is never
recorded, and the job stops.

The method follows Miller, *Adding Error Bars to Evals* (Anthropic, 2024). It lives in `stats.py`.

1. **Average within each case first.** A case's score is the share of its repeats that passed. The pass rate
   is the mean of case scores, so a case with more repeats does not count more.
2. **Measure uncertainty across cases.** The standard error is the standard deviation of case scores divided
   by the square root of the number of cases. Cases are the sampled unit; repeats share a prompt and an
   expected answer, so they are not independent and can't be allowed to shrink the error. Treating 40 cases ×
   5 repeats as 200 independent trials would make every interval far too narrow.
3. **Turn that into an interval with Wilson's formula.** A plain mean ± 1.96 × SE can run past 0% or 100%,
   and when every run passes it collapses to a single point that claims certainty. The Wilson score
   interval doesn't. It is computed at the *effective sample size* the standard error implies, which
   always lies between the number of cases (repeats fully agree) and the number of runs (repeats
   independent). When every case scores the same, the standard error is zero and the interval falls back
   to the number of cases, the conservative end.

So 5 cases that all pass report 100% with a 95% interval of about 57–100%, not "100–100%". With few cases
the intervals are wide, and that is the honest answer. At a 70% pass rate:

| Cases | Approximate 95% interval |
|---|---|
| 10 | ±28 points |
| 40 | ±14 points |
| 160 | ±7 points |

**Comparing two configurations** properly means running both on the same cases and computing the
difference in case scores with its own standard error, a paired comparison. Overlapping intervals are not
that test: two intervals can overlap while the paired difference is clearly real. The harness doesn't
compute paired differences yet, so treat a gap of a few points between runs as unproven.

## Calibrating evaluators

Any evaluator that makes a judgment call is a measurement that can be wrong: the Jev judges, and code that
approximates judgment, such as `grounded`. Mechanical checks against values you wrote, such as `answer_correct`
or `read_only`, don't need this. Calibration is an eval of the evaluator: the evaluator is the system under test
and your labels are the ground truth. The method follows current practice for validating LLM judges against
expert labels.

1. **Build a calibration set.** `hermes-evals sample RUN_DIR --into calibration/<set> --evaluators pass_fail`
   copies a run's responses into the set. You can also write responses by hand, such as one good and one bad
   per important case. Hand-written responses make a fixed benchmark; sampled ones show how the evaluator does on
   what your agent actually produces. Use both.
2. **Label blind, pass or fail.** Fill in `labels.csv`. It never shows the evaluator's verdict. Label as a
   domain expert, and aim for about as many fails as passes, even if real failures are rare.
3. **Measure on dev.** `hermes-evals calibrate calibration/<set> --evaluator pass_fail` re-runs the evaluator on
   every labeled response in the dev split and reports:
   - **TPR**: when you said pass, how often it passed. **TNR**: when you said fail, how often it failed.
     These are the headline numbers, with Wilson intervals. Aim for 90% on both; below 80% the evaluator isn't
     trustworthy. Judges often err lenient: high TPR, low TNR.
   - **Kappa**, as a secondary number. It folds both directions into one and is best for comparing two humans.
   - **The best threshold on dev**, by balanced accuracy, (TPR + TNR) / 2.
   - **The disagreements.** Each is a response where you and the evaluator differ. Read them.
4. **Iterate on dev only.** Change the threshold, a case's description of a good response, or the judge model
   (`--threshold`, `--model`), and measure again. An evaluator error, such as an unreadable judge reply, counts as
   a disagreement, never as a dropped row.
5. **Report on test, once.** `--split test` gives the number you quote. Responses are assigned to dev or test by
   a hash of their ID, so a response never moves between splits. Looking at test while tuning overfits.
6. **Correct judged pass rates.** With TPR and TNR known, `--run RUN_DIR` corrects that run's judged pass rate
   for the evaluator's errors: true rate = (observed + TNR - 1) / (TPR + TNR - 1) (Rogan and Gladen, 1978).

## Failure analysis

`hermes-evals label RUN_DIR` writes `failure_tags.csv`, one row per failed run with the evaluators that
failed. Tag each run with one cause, then run `hermes-evals grade RUN_DIR` again to count them:

| Cause | Meaning | Usual fix |
|---|---|---|
| `model` | The model reasoned or acted wrongly given correct instructions and tools. | Different model, or training data |
| `prompt` | The system prompt, skills or case wording led it astray. | Edit the prompt or skill |
| `grader` | The agent was right and the evaluator was wrong. | Fix the evaluator, then `grade` again |
| `retrieval` | It used the wrong table, column or tool, or got bad data back. | Schema docs, tool descriptions, data |

The report counts the tags. Read every `grader` failure first. A grader bug inflates the failure rate of
every configuration equally, and it is the cheapest fix.

## Discipline

Borrowed from how Hermes's own evals are run, plus the statistics above:

- Run at least 3 repeats before comparing configurations. Use 5 for anything you publish.
- Keep one configuration per run. The harness enforces this: each run clones the agent once, and runs are
  never resumed.
- Keep denominators identical across configurations, and count errored runs as fails.
- Weak and mid-sized models show differences that frontier models hide.
- Read the trajectories of surprising results before quoting a number.
- Stamp every run. `run.json` records the agent (host, profile, model, provider, MCP servers, skills,
  fingerprints of SOUL and config, Hermes version), the dataset and, once graded, the evaluator config.
  Every trajectory carries its model, tokens and cost.
