# Where the Project's Data Comes From

This document explains, in plain language, what data we used to "teach" the system to predict World Cup 2026 results and where each piece came from. It's meant to give context to teammates who didn't work on the technical side, to help with the group presentation.

## The general idea

The system works like a "student" that studies thousands of past football matches to learn patterns: when a team tends to win, when it tends to lose, when a game tends to be close. After that learning phase, it uses what it learned to simulate the entire 2026 World Cup — groups, knockout rounds, final — and estimate which teams are more likely to advance and win the title.

For this to work well, it needs good input data. We used four main sources.

## 1. Match history (the system's "experience")

- **What it is:** almost 50,000 international football matches, from 1872 to today, including **every World Cup edition** since 1930.
- **What it's for:** this is the main study material. The more matches the system analyzes, the better it learns the difference between a strong team and a weak one, and how that usually translates into a scoreline.
- **Where it came from:** a public database of historical football results.

## 2. Elo — a "strength score" for each national team

- **What it is:** a scoring system used for decades in football (and other sports) to measure a team's strength based on the results it has been getting, against whom, and in what competition. It works like a numeric ranking that goes up when a team beats strong opponents and goes down when it loses to weaker ones.
- **What it's for:** it gives the system a reliable sense of how strong each team is today, instead of guessing that from the team's name alone.
- **Where it came from:** a database specialized in calculating this Elo score for the 48 teams competing in the 2026 World Cup, with history going back to each team's international debut.

## 3. The official FIFA ranking

- **What it is:** the official ranking that FIFA itself publishes periodically, similar to Elo, but it's the "official" number the media and FIFA itself use.
- **What it's for:** it works as a second opinion on each team's strength, complementing Elo.
- **Where it came from:** two different sources — one with the most current ranking (mid-2026) and another with the ranking's history over several decades, which let us know each team's position *at the time* of each old match, not just its position today.

## 4. Expected goals (xG) and betting odds — used to "check the work"

- **xG (expected goals):** a modern football metric that estimates how many goals a team "should" have scored based on the quality of the chances it created, not just the final score. It's a way of telling whether a team played well even if the result didn't reflect it.
- **Betting odds:** the prices that sportsbooks offer for each outcome (win, draw, loss). Betting markets tend to be very good at pricing the real chance of each result, so we used them as a "benchmark" to compare against: does our system do better or worse than the market?
- **What it's for:** these two data points were **not used to train** the system — they're only used afterward, to check whether the predictions it generates make sense, by comparing them against reality and against the betting market.
- **Where it came from:** real data from the 2026 World Cup, which is already underway (matches already played so far).

## Flags and the group draw

Besides the data used to "train" the system, we also put together the visual side of the project:

- **Flags** for all 48 competing teams, shown on screen instead of generic icons.
- **The official draw for the 12 groups** of the 2026 World Cup, so the simulation uses the tournament's real matchups instead of made-up groups.

## Quality checks we made along the way

A few adjustments were needed to make the data trustworthy:

- **We removed matches from tournaments that aren't "World Cup level"** (competitions between unofficial teams or very informal exhibition games), because they had extremely lopsided scorelines (teams winning by 20 or 30 goals) that were confusing the system's learning.
- **We capped the influence of unusually large blowouts**, even in official matches, so that one exceptional result wouldn't distort the overall learning.
- **We fixed team names that were spelled differently across sources** (e.g. "USA" in one dataset and "United States" in another), to make sure the system always recognizes it's the same team.
- **We found and fixed a connection issue between the website and the prediction system**, which was causing the live demo to ignore each team's strength data entirely. Once fixed, results became much more realistic — favorites winning more often, just like in real football.
- **We started feeding each team's real recent match history into the simulation**, instead of having every team start the tournament with a blank slate. Without it, a single unlucky early result made a team look far weaker than it really is for its very next match, which was causing too many strong teams to be knocked out together in the group stage.

## The end result

After gathering and cleaning this data, the system was trained and tested. Today it:

- Simulates the group stage, best third-placed teams, knockout rounds (with extra time and penalties when needed), and the final of the 2026 World Cup using the real 48 teams and real groups
- Produces plausible win probabilities (favorites win more often, but upsets happen, just like in real life)
- Can be compared against the betting market and against real expected-goals data from the ongoing 2026 World Cup, to measure how good it actually is
