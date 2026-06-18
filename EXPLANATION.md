## What the dataset is

45,211 records from a Portuguese bank's telemarketing campaigns. Each row is one customer contact. The goal is to predict whether they subscribed to a term deposit (`y = yes/no`).

The core problem is class imbalance — only 11.7% of customers said yes. That means if you build a model that just predicts "no" every time, you get 88% accuracy and it's completely useless. So accuracy alone is a bad metric here. I focused on F1 and ROC-AUC instead.

---

## The one preprocessing decision worth explaining

`pdays` stores how many days since the customer was last contacted from a previous campaign. But `-1` means they were *never* contacted before — it's not a real number, it's a sentinel value. Treating it as -1 in a numeric column would confuse the model (it'd look like "contacted 1 day in the future" or something nonsensical).

I split it into two things:
- `was_contacted_before` — binary flag (0 or 1)
- `pdays` — zeroed out where it was -1

That's a cleaner representation of what the data actually means.

---

## Why XGBoost over the alternatives

I compared Logistic Regression (baseline) vs XGBoost. The LR baseline is useful because it's simple and interpretable — if XGBoost doesn't beat it meaningfully, it's not worth the complexity.

It does beat it:

| | LR | XGBoost |
|---|---|---|
| F1 | 0.516 | 0.590 |
| ROC-AUC | 0.889 | 0.931 |

The AUC gap is the real signal. AUC measures how well the model *ranks* customers by likelihood — 0.931 means XGBoost is doing a genuinely better job separating subscribers from non-subscribers, not just getting lucky on the threshold.

LR struggles here because the relationship between features and subscription isn't linear. XGBoost can capture things like "cellular contact AND previous campaign success AND long call = very likely yes" as an interaction, which LR can't express.

---

## What the model actually learned

Top features by importance:

1. **contact type** — cellular vs telephone vs unknown. Cellular contact converts much better. Possibly because cellular numbers are more likely to reach the actual person.
2. **duration** — length of the last call. Longer calls = higher subscription rate. *Important caveat below.*
3. **housing loan** — customers without a housing loan subscribe more. Makes sense — they have more financial flexibility.
4. **poutcome** — previous campaign outcome. If they said yes before, they're likely to say yes again.

**The duration problem:** Duration is the second most important feature but it's a post-call measurement — you don't know how long the call will be before you make it. So in a real system where you're trying to decide *who to call*, duration would be data leakage and you'd drop it. I kept it here because the task is predictive modelling on historical data, not a live deployment. But this is the first thing I'd flag if this were going into production.

---

## The 5 customer predictions

- **Customer #2 (98.8% yes, actual yes):** 63 years old, management, had a successful previous campaign, was contacted before. The model is very confident — prior success is a massive signal.
- **Customer #1 (59.7% yes, actual yes):** Older, retired, decent balance, long call. Moderate confidence — the call duration and age profile pushed it over.
- **Customer #3 (2.5% no, actual no):** Short call (128s), no prior contact, unknown previous outcome. Model correctly reads this as a low-probability contact.
- **Customer #5 (0.3% no, actual no):** Very short call, minimal balance, zero prior history. Near-certain no.
- **Customer #4 (10.7% no, actual yes):** This is the interesting one — a false negative. The model was uncertain (10.7%) but called it wrong. The call was only 114 seconds, which pulled the prediction toward no. The customer subscribed anyway. This is exactly the tension with duration as a feature — a short call doesn't always mean disinterest.

---

## What I'd do differently with more time

- Drop `duration` and retrain to see how much performance degrades — that'd give a cleaner picture of a deployable model
- Try threshold tuning: the default 0.5 cutoff isn't optimal for imbalanced data; lowering it would trade precision for recall, which may be worth it in a marketing context
- Proper cross-validation instead of a single train/test split
