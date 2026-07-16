# How the Model Turns Data Into a Prediction

This document explains, in plain language, what happens between "here are two teams about to play" and "here's the predicted result." It's meant for teammates who didn't work on the technical side, to help with the group presentation.

## The short version

For every match, the system looks at the two teams, compares how strong each one is, and uses what it learned from studying thousands of past matches to guess how many goals each team is likely to score. From that guess, it works out the chance of a home win, a draw, or an away win — and then "plays" the match by drawing a realistic scoreline from those chances. It repeats this for every match of the tournament: groups, knockout rounds, and the final.

## Step 1: Comparing the two teams

Before predicting anything, the system doesn't look at team names — it looks at the *difference* between the two teams across a handful of signals:

- **Elo score difference** — which team's strength rating is higher, and by how much
- **FIFA ranking difference** — the same idea, using the official FIFA ranking
- **Recent form difference** — how well each team has been doing in its last few matches (win rate, goals scored, goals conceded)
- **Context** — what stage of the competition this is (group stage, round of 16, final...) and what kind of competition (World Cup, qualifier, friendly...), since teams tend to play differently depending on how much is at stake

This is the same kind of thinking a football analyst does before a match — "Team A is ranked much higher and has been in great form, Team B has been struggling" — just turned into numbers the system can work with.

## Step 2: The learning part

The system uses a **neural network**, which is a type of program loosely inspired by how the brain recognizes patterns. Instead of us writing rules like "if the Elo difference is bigger than X, predict a win," the network figured out its own rules by studying almost 50,000 real historical matches: for each one, it saw the two teams' strength difference and recent form going in, and the actual final score coming out. Over many rounds of studying that history, it gradually adjusted itself to get better and better at guessing realistic scorelines.

This is why data quality mattered so much: if the network studies matches with extreme, unrealistic scorelines (like 30-0 blowouts between amateur teams), it can pick up bad habits. That's part of why we cleaned the data before training it, as explained in the companion data document.

## Step 3: From "expected goals" to win/draw/loss chances

For any given match, the trained network doesn't just spit out a final score — it estimates **how many goals each team is expected to score on average** (for example, "Brazil 1.9, Panama 0.6"). A single number like that doesn't tell you the exact score, because football has randomness — a team expected to score 1.9 goals might score 0, 1, 2, 3, or more on a given day.

To turn that expectation into something useful, the system uses a well-established statistical method from sports analytics that models how likely each possible scoreline is, given those two expected-goal numbers. Adding up all the scorelines where the home team wins gives the home win probability; the same for draws and away wins. This is the same general approach real football analysts and betting markets use to price match outcomes.

## Step 4: Actually "playing" the match

Once the system has the win/draw/loss chances, it randomly draws one specific scoreline consistent with those chances — the same way a fair dice roll respects the odds without always giving the same result. That's why running the exact same matchup twice can give different scores: a heavy favorite will win most of the time, but not always, just like in real football.

## Step 5: Extra time and penalties

For any knockout match (round of 32 onward) that ends level after regular time, the system simulates 30 extra minutes the same way, using a smaller expected-goals number since less time is left to score. If the teams are still level after that, it simulates a penalty shootout, giving a slight edge to whichever team the model considers stronger. Only the final match is shown minute-by-minute with this extra-time sequence on screen; every other knockout match uses the same underlying logic but just displays the final result.

## Step 6: Running the whole tournament

The system repeats steps 1-5 for every match needed to complete the tournament:

1. All group-stage matches, to produce the group standings
2. The 8 best third-placed teams, to fill out the round of 32
3. Each knockout round, one at a time, using the actual winners from the previous round
4. The final (with the extra minute-by-minute animation and, if needed, extra time and penalties)

Because each team's "recent form" updates after every simulated match, the system's read on a team can shift slightly as the tournament goes on — similar to how a team's form and confidence can shift over a real tournament.

## How we know if it's any good

A model is only useful if its predictions are checked against reality. We compare the system's predictions for the real, ongoing 2026 World Cup matches against two independent references:

- **Betting market odds** for those same matches, converted into implied probabilities — the market is historically hard to beat, so it's a meaningful benchmark
- **Real expected-goals (xG) data** from those same matches, to see how close the system's goal expectations are to what actually happened on the pitch

This comparison doesn't just tell us "is the model right or wrong" — it tells us how the model stacks up against tools professionals already trust.
