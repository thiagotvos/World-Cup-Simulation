# How Accurate Is the System? (Metrics Report)

This document shows how well the trained model actually performs, using two different comparisons: how it stacks up against the sportsbook betting market on the real, ongoing 2026 World Cup, and how it would have done predicting real past World Cups. It's meant to give the group hard numbers to point to during the presentation.

## How to read the numbers below

Every match has three possible outcomes: home win, draw, or away win. A system that just guessed randomly would be right about **33%** of the time. Professional football prediction models (including the betting market itself) typically land somewhere in the **50-60%** range — football is simply hard to predict, even for experts, which is part of what makes it exciting to watch. Keep that range in mind when judging the numbers below; anything meaningfully above 33% shows the system learned something real.

Three metrics show up throughout this report:

- **Accuracy** — the percentage of matches where the system's top pick (home/draw/away) matched the real outcome. The easiest number to explain to anyone.
- **Log loss** — a stricter score that also rewards being *confident and correct*, and punishes being *confident and wrong*, not just picking the right side. Lower is better. It's the standard way analysts compare probability-based forecasts, which is why we report it alongside accuracy.
- **F1 score** — checks that the system isn't just good at spotting the common outcome (home wins are the most frequent result in football) while missing rarer ones like away wins and draws. Higher is better.

## Part 1 — Our model vs. the real sportsbook betting market (2026 World Cup)

The 2026 World Cup is currently in progress. We took the 102 matches already played and compared what our model predicted against what real sportsbooks priced for those same matches, and against what actually happened.

| | Accuracy | Log loss | F1 score |
|---|---|---|---|
| **Our model** | 61.8% | 0.858 | 0.449 |
| **Betting market** | 68.6% | 0.802 | 0.559 |

The betting market is still slightly ahead — which is expected and not a bad result for us. Sportsbooks price matches using information our system doesn't have access to (injuries, lineup news, last-minute changes) and represent the collective judgment of professional bettors putting real money behind their predictions. Coming within striking distance of that benchmark, using only historical results, team strength ratings, and recent form, is a meaningful outcome for a project built in this timeframe.

### Bonus check: goal predictions vs. real expected goals (xG)

For the same 102 matches, we also compared our model's predicted goals per team against **xG (expected goals)**, a modern stat that estimates how many goals a team's actual chances were worth, regardless of the final score.

| | Home team | Away team |
|---|---|---|
| Average error (goals) | 0.42 | 0.50 |

In other words, our model's goal expectations are typically within about half a goal of what really happened on the pitch — a solid result for a system that only sees team names, ratings, and match context, with no play-by-play data.

## Part 2 — How would it have done on real, past World Cups?

To get a second, independent read on accuracy, we tested the model against **every official FIFA World Cup match from 1930 to 2022** — 964 matches across 22 editions — comparing its predicted outcome to what actually happened in history.

**Important caveat, in plain terms:** the model studied historical matches to learn from them, the same way a student studies past exams. Most of those 22 World Cups were part of that study material, so a good score there mostly shows the model learned real patterns well, not that it could have predicted the future. Only the **2022 World Cup** was never shown to the model in any form — that's the one genuinely "blind" test in this report, closest to what predicting an unplayed tournament actually looks like.

| | Accuracy | Log loss | F1 score | Matches |
|---|---|---|---|---|
| **All 22 editions (1930-2022) combined** | 50.2% | 1.030 | 0.332 | 964 |
| **2022 World Cup only — genuinely unseen data** | 53.1% | 0.997 | 0.393 | 64 |

The 2022 result is the most meaningful single number in this section: on a World Cup it never studied, the model still called the right outcome in just over half of all matches — comfortably above the 33% random baseline, in a competition where even the best sportsbooks and analysts are far from perfect.

### Accuracy by World Cup edition

| Edition | Accuracy | Matches | Status |
|---|---|---|---|
| 1930 | 61.1% | 18 | studied by the model |
| 1934 | 35.3% | 17 | studied by the model |
| 1938 | 44.4% | 18 | studied by the model |
| 1950 | 45.5% | 22 | studied by the model |
| 1954 | 50.0% | 26 | studied by the model |
| 1958 | 45.7% | 35 | studied by the model |
| 1962 | 53.1% | 32 | studied by the model |
| 1966 | 56.3% | 32 | studied by the model |
| 1970 | 62.5% | 32 | studied by the model |
| 1974 | 55.3% | 38 | studied by the model |
| 1978 | 39.5% | 38 | studied by the model |
| 1982 | 42.3% | 52 | studied by the model |
| 1986 | 57.7% | 52 | studied by the model |
| 1990 | 51.9% | 52 | studied by the model |
| 1994 | 46.2% | 52 | studied by the model |
| 1998 | 46.9% | 64 | studied by the model |
| 2002 | 46.9% | 64 | studied by the model |
| 2006 | 57.8% | 64 | studied by the model |
| 2010 | 43.8% | 64 | studied by the model |
| 2014 | 53.1% | 64 | studied by the model |
| 2018 | 51.6% | 64 | used for fine-tuning checks (not directly studied) |
| **2022** | **53.1%** | **64** | **never seen — genuinely blind test** |

## Takeaways

- Against the real betting market on the live 2026 World Cup, the model is competitive (61.8% vs. 68.6% accuracy) without ever beating the market outright — which is the expected, honest outcome for this kind of project.
- On the one truly unseen historical tournament (2022), the model correctly called 53.1% of match outcomes — well above random chance, and in line with what serious football prediction systems typically achieve.
- Goal predictions track real expected-goals data (xG) closely, within roughly half a goal per team on average.
- These numbers, together, are what let us say the model is doing something genuinely useful, not just producing plausible-looking noise.

## How to reproduce these numbers

```bash
# Model vs. sportsbook odds/xG on the live 2026 World Cup
python -m app.evaluate --model-dir model/saved_model --profiles data/raw/team_profiles.csv

# Model vs. real results across every past World Cup (1930-2022)
python -m app.backtest --model-dir model/saved_model --profiles data/raw/team_profiles.csv
```
