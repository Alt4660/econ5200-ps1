 # AI Appendix: Problem Set 1, Phase 4

## 1. Prompt (as sent)

```
[Prep] Act as a partner reviewing an analytics deliverable before it
goes to a client board. You are looking for the objection that would
embarrass us in the room.

[Request] I audited a retailer's "average basket value" metric. I found
the naive mean was inflated by a right-skewed B2B tail and by a change
in what was logged between years. I recommend the dashboard show a
trimmed mean with an explicit B2B exclusion rule. Give me the three
strongest objections to that recommendation, ordered by how likely the
client is to raise them, and for each one tell me what evidence would
settle it.

[Iterate] Be specific to this metric. Do not give me generic data
quality advice.

[Mechanism Check] For each objection, say whether it is a question
about the data, about the statistic, or about the business definition
of the metric -- these need different kinds of answer.

[Evaluate] Which single additional analysis would most reduce our
exposure?
```

## 2. Raw reply (unedited)

Here is my response to the prompt as sent. I'm using your notebook's numbers where they sharpen the objections.

## Objection 1: "Your fix doesn't touch the second problem"
**Category: data (with a definition question underneath)**

Your two faults are separate. B2B orders are a tail problem, and a trimmed mean and exclusion rule address that. But cancelled orders appearing only in year 2 is a change in the population being measured. Cancellations sit inside the body of the distribution (about 4.7% of year 2 rows, only modestly larger than normal baskets), so trimming won't remove them. A board member who reads "trimmed mean with B2B exclusion" will ask, "Does this fix the year-over-year comparison?" Your own estimators say no: 3.85% (trimmed) and 3.46% (exclusion) growth against a true 1.58%.

**Evidence that settles it:** Recompute growth on a like-for-like basis, with cancelled orders excluded in both years, plus the B2B rule. If the remaining gap to truth closes, the recommendation needs a logging-consistency rule added. If it doesn't, something else is going on.

## Objection 2: "That isn't the average basket value, and it won't reconcile to revenue"
**Category: statistic and business definition**

Finance found the problem by reconciling to revenue. A 10%-trimmed mean of right-skewed data is biased downward as a level estimate. In your table it's about 62.5 against a true 75.5, so a trimmed mean times order count won't match consumer revenue. You'd be swapping a metric that's wrong in growth for one that's wrong in level. The client will ask whether the dashboard is measuring "average spend per order" or "typical order."

**Evidence that settles it:** Compare each candidate's level against the known truth (you have it in the simulation), and check candidate × order count against consumer-only revenue. Then ask the client what the pricing decision actually needs, the mean or the typical basket. If it's revenue-linked, a mean on cleaned data beats a trimmed mean.

## Objection 3: "Your B2B rule is arbitrary, and it was tuned on data where you knew the answer"
**Category: statistic, with a data component**

The Tukey fence on logs is recomputed within each year, so the threshold moves when the mix changes. With 15 B2B orders in year 1 and 30 in year 2, year 2's fence is a different cutoff. The rule can also clip legitimate large consumer orders while missing smaller B2B ones. Your B2B median is about $3,000 against a consumer median near $55, so the planted case is easy. A real client's B2B orders may not be separated that cleanly. Stacking a 10% trim on top of the exclusion also double-counts the tail treatment.

**Evidence that settles it:** Check the rule's classification against the `is_b2b` flag (a confusion matrix). Apply a fixed year-1 fence to both years. Run a sensitivity sweep over k. And use a rule whose fence doesn't depend on year-specific data.

## The single analysis that most reduces exposure

Run a **like-for-like growth comparison**: apply a fixed B2B fence and the same cancellation treatment to both years, then report growth against the true 1.58% with a bootstrap confidence interval. It directly tests Objection 1, which is the one most likely to be raised and the one your current numbers already suggest is live. It also covers Objection 3 through the fixed fence, and it gives you the confidence statement the board slide needs.

---

**For your appendix:** save the prompt exactly as you sent it, this reply unedited, and one line for every change you make afterward. If you pursue Objection 1 and your Phase 2 recommendation changes (for example, adding a logging-consistency rule), say so in the notebook, since that's what the rubric rewards. I'd suggest building the like-for-like analysis next, and I can write that code with you.

## 3. Code the AI supplied for the analysis (follow-up reply, as sent)

```python
# Like-for-like growth: same B2B rule and same cancellation treatment in both years

# 1. Fixed B2B fence from year 1 only, applied to both years
y1_logs = np.log(df.loc[df["year"] == 1, "basket_value"])
q1, q3 = y1_logs.quantile([0.25, 0.75])
fixed_fence = np.exp(q3 + 1.5 * (q3 - q1))

# 2. Drop B2B (by the fence) and cancelled orders in both years
like_for_like = df[(df["basket_value"] <= fixed_fence) & (df["cancelled"] == False)]
lfl_mean = like_for_like.groupby("year")["basket_value"].mean()
lfl_growth = (lfl_mean[2] / lfl_mean[1] - 1) * 100

print("like-for-like means:", lfl_mean.round(2).to_dict())
print("like-for-like growth (%):", round(lfl_growth, 2))
print("true growth (%):", round(true_growth, 2))

# 3. Does the fence catch the planted B2B orders? (answer key check)
flagged = df["basket_value"] > fixed_fence
print(pd.crosstab(df["is_b2b"], flagged, rownames=["actually B2B"], colnames=["above fence"]))

# 4. Bootstrap 95% interval for growth, resampling within each year
def growth_once(data):
    m = data.groupby("year")["basket_value"].mean()
    return (m[2] / m[1] - 1) * 100

boot = []
for _ in range(1000):
    resampled = pd.concat([
        g.sample(len(g), replace=True, random_state=int(rng.integers(1_000_000)))
        for _, g in like_for_like.groupby("year")
    ])
    boot.append(growth_once(resampled))
print("95% interval for growth (%):", np.percentile(boot, [2.5, 97.5]).round(2))
```

## 4. Later prompts in the same chat

For the exchanges below, paste the full prompt and the AI's raw reply from the chat under each heading.

- **Fixed fence:** "Why do you say, I now recommend the B2B exclusion rule with a fixed fence, not re-estimated each year..."
- **Findings text:** "rewrite what I would say for what you found and whether it changes your answer"
- **Accuracy check:** "is this accurate to what my data shows?"
- **Corrections:** "I want you to only fix the three claims you told me are genuinely inaccurate."
- **Board slide:** "write the 150 word board slide at the end"

## 5. Changes I made

- Sent the prompt exactly as written in the brief, even though my own Phase 2 recommendation was the B2B exclusion mean alone, not a trimmed mean plus exclusion. Parts of Objections 2 and 3 (trimmed-mean level bias, "double-counting the tail") therefore don't apply to my actual recommendation.
- Chose Objection 1 (the logging change) as the strongest, because the AI's Evaluate answer pointed to it and my own Phase 2 numbers (3.46% vs a true 1.58%) already showed the gap.
- Ran the AI's like-for-like code unchanged apart from deleting its first two header comment lines. I did not run the sensitivity sweep over k suggested for Objection 3.
- Used the confusion table from that code to partly address Objection 3.
- Did not pursue Objection 2 (revenue reconciliation) beyond noting in my findings that the rule understates the level by about 2 dollars.
- Asked the AI whether the "fixed fence (not re-estimated each year)" in its draft recommendation was supported by my data. It ran a per-year vs fixed comparison in the chat and found little difference (fixed 1.31% vs per-year 1.63%, against a true 1.58%), so I removed the fixed-fence claim from my recommendation and did not include that comparison in the notebook.
- The AI's first draft of the findings text said the cancelled orders accounted for "about 1.9 points" of fake growth. That was wrong (3.46 minus 1.31 is about 2.2), and I corrected it.
- The AI's draft said the 93 non-B2B rows above the fence were "probably" legitimate large baskets. I checked against the data: 84 are completed consumer orders and 9 are cancelled, so I used those counts.
- Rewrote the AI's findings text in my own wording and kept only the claims my output supports.
- Revised my Phase 2 recommendation after the result: from the B2B exclusion mean alone (3.46% growth) to the B2B exclusion rule plus excluding cancelled orders in both years, reported with its cutoff multiplier and its interval.
- The board slide was drafted by the AI from my notebook output, written to match my style. [Edit this line to say what you changed, or that you used it as written.]
