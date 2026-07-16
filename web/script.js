const introScreen = document.getElementById("introScreen");
const simulationScreen = document.getElementById("simulationScreen");
const stageContent = document.getElementById("stageContent");
const startButton = document.getElementById("startButton");
const advanceButton = document.getElementById("advanceButton");
const statusText = document.getElementById("statusText");
const stageTitle = document.getElementById("stageTitle");
const stageText = document.getElementById("stageText");
const phasePills = Array.from(document.querySelectorAll(".phase-pill"));
const speedButtons = Array.from(document.querySelectorAll("[data-speed]"));
const API_BASE = window.WC_API_BASE || "http://127.0.0.1:5000";
const SPEED_PRESETS = {
  slow: 1200,
  medium: 850,
  fast: 420,
};
const KNOCKOUT_STAGES = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final"];

const DEFAULT_TEAM_META = {
  team: "Unknown",
  code: "UNK",
  flag: null,
};

const demoTournament = {
  groups: {
    A: ["Team A1", "Team A2", "Team A3", "Team A4"],
    B: ["Team B1", "Team B2", "Team B3", "Team B4"],
    C: ["Team C1", "Team C2", "Team C3", "Team C4"],
    D: ["Team D1", "Team D2", "Team D3", "Team D4"],
    E: ["Team E1", "Team E2", "Team E3", "Team E4"],
    F: ["Team F1", "Team F2", "Team F3", "Team F4"],
    G: ["Team G1", "Team G2", "Team G3", "Team G4"],
    H: ["Team H1", "Team H2", "Team H3", "Team H4"],
    I: ["Team I1", "Team I2", "Team I3", "Team I4"],
    J: ["Team J1", "Team J2", "Team J3", "Team J4"],
    K: ["Team K1", "Team K2", "Team K3", "Team K4"],
    L: ["Team L1", "Team L2", "Team L3", "Team L4"],
  },
};

const demoProfiles = {};
Object.entries(demoTournament.groups).forEach(([groupName, teams], groupIndex) => {
  teams.forEach((team, teamIndex) => {
    const seed = groupIndex * 4 + teamIndex;
    demoProfiles[team] = {
      elo: 1900 - seed * 8,
      fifa_rank: 1 + seed,
    };
  });
});

const tournamentConfig = window.WC_TOURNAMENT || demoTournament;
const teamMetaConfig = window.WC_TEAM_META || {};

const appState = {
  phase: "intro",
  actualPhase: "intro",
  result: null,
  animateDelay: SPEED_PRESETS.medium,
  speed: "medium",
  advanceHandler: null,
  groupTrackerElements: [],
  currentGroupIndex: 0,
  selectedGroupIndex: 0,
  bracketSlots: {},
  liveGroupStandings: {},
  reached: { groups: false, thirds: false, knockout: false, final: false },
  animating: false,
};

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function setStatus(message) {
  statusText.textContent = message;
}

function setSpeed(speed) {
  if (!Object.prototype.hasOwnProperty.call(SPEED_PRESETS, speed)) {
    return;
  }
  appState.speed = speed;
  appState.animateDelay = SPEED_PRESETS[speed];
  speedButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.speed === speed);
  });
}

function setPhase(phase) {
  appState.phase = phase;
  phasePills.forEach((pill) => {
    pill.classList.toggle("active", pill.dataset.phase === phase);
    pill.classList.toggle("done", isPhaseDone(appState.actualPhase, pill.dataset.phase));
  });
  refreshPhaseNav();
}

function isPhaseDone(currentPhase, pillPhase) {
  const order = ["groups", "thirds", "knockout"];
  return order.indexOf(pillPhase) < order.indexOf(currentPhase);
}

function refreshPhaseNav() {
  // Navigation is only safe when nothing is actively animating: both the
  // phase being left and the phase being entered need to be fully settled,
  // otherwise a running animation loop keeps mutating DOM the peek just
  // rebuilt (or replaced) out from under it.
  const canNavigate = !appState.animating && Boolean(appState.reached[appState.phase]);
  phasePills.forEach((pill) => {
    const pillPhase = pill.dataset.phase;
    const enabled = canNavigate && Boolean(appState.reached[pillPhase]);
    pill.disabled = !enabled;
    pill.classList.toggle("clickable", enabled);
  });
}

function goToPhase(phase) {
  const result = appState.result;
  if (!result || appState.animating || !appState.reached[appState.phase] || !appState.reached[phase]) {
    return;
  }
  if (phase === "groups") {
    showGroupsSnapshot(result);
  } else if (phase === "thirds") {
    showThirdsStage(result, true);
  } else if (phase === "knockout") {
    showKnockoutSnapshot(result);
  }
}

function resumeActualPhase() {
  const result = appState.result;
  if (!result) {
    return;
  }
  if (appState.actualPhase === "knockout" || appState.actualPhase === "final") {
    showKnockoutSnapshot(result);
  } else if (appState.actualPhase === "thirds") {
    // Not a peek: thirds is still the true current phase, so restore the
    // normal forward-progressing button instead of another "Continue".
    showThirdsStage(result, false);
  } else {
    showGroupsSnapshot(result);
  }
}

function showSimulationScreen() {
  introScreen.classList.add("is-hidden");
  simulationScreen.classList.remove("is-hidden");
}

function clearStageContent() {
  stageContent.innerHTML = "";
  appState.groupTrackerElements = [];
  appState.bracketSlots = {};
}

function setAdvanceButton(label, handler) {
  appState.advanceHandler = handler;
  advanceButton.textContent = label;
  advanceButton.classList.remove("is-hidden");
}

function hideAdvanceButton() {
  appState.advanceHandler = null;
  advanceButton.classList.add("is-hidden");
}

function buildMatchCard(entry, kind) {
  const card = document.createElement("article");
  card.className = `match-card ${kind}`;

  const score = document.createElement("div");
  score.className = "match-score";
  score.innerHTML = `
    <div class="match-team home">
      ${buildTeamMark(entry.home_team)}
      <strong>${entry.home_team}</strong>
    </div>
    <span class="score-pill">${entry.home_goals}<span> - </span>${entry.away_goals}</span>
    <div class="match-team away">
      <strong>${entry.away_team}</strong>
      ${buildTeamMark(entry.away_team)}
    </div>
  `;

  card.appendChild(score);
  return card;
}

function goalDifference(row) {
  return Number(row.goal_difference ?? row.goalDifference ?? row.gd ?? row.goals_for - row.goals_against);
}

function formatStage(stage) {
  const label = (stage || "").replaceAll("_", " ").trim();
  if (!label) {
    return "Match";
  }
  return label.charAt(0).toUpperCase() + label.slice(1);
}

function stageDisplayName(stage) {
  const labels = {
    round_of_32: "Round of 32",
    round_of_16: "Round of 16",
    quarterfinal: "Quarterfinals",
    semifinal: "Semifinals",
    final: "Final",
  };
  return labels[stage] || formatStage(stage);
}

function groupMatchesByStage(matches) {
  return matches.reduce((acc, match) => {
    if (!acc[match.stage]) {
      acc[match.stage] = [];
    }
    acc[match.stage].push(match);
    return acc;
  }, {});
}

function hashString(value) {
  let hash = 0;
  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) % 360;
  }
  return hash;
}

function normalizeTeamKey(team) {
  return String(team || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function resolveTeamMeta(team) {
  const key = normalizeTeamKey(team);
  const meta = teamMetaConfig[key] || teamMetaConfig[team] || {};
  const code = meta.code || getTeamCode(team);
  return {
    team: meta.team || team || DEFAULT_TEAM_META.team,
    code,
    flag: meta.flag || meta.flag_url || meta.flagPath || null,
  };
}

function getTeamCode(team) {
  const tokens = String(team || "")
    .replace(/[^a-z0-9 ]/gi, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (!tokens.length) {
    return "TBD";
  }
  if (tokens.length === 1) {
    return tokens[0].slice(0, 3).toUpperCase();
  }
  return tokens.slice(0, 3).map((token) => token.charAt(0).toUpperCase()).join("");
}

function getTeamTone(team) {
  return `hsl(${hashString(team)}, 70%, 46%)`;
}

function buildTeamMark(team) {
  const meta = resolveTeamMeta(team);
  if (!team) {
    return `<span class="team-mark empty"></span>`;
  }
  if (meta.flag) {
    return `
      <span class="team-mark flag-mark" style="--team-tone: ${getTeamTone(team)};">
        <img src="${meta.flag}" alt="" aria-hidden="true" />
      </span>
    `;
  }
  const tone = getTeamTone(team);
  return `
    <span class="team-mark" style="--team-tone: ${tone};">
      <span>${meta.code}</span>
    </span>
  `;
}

function buildBracketTeam(team, side) {
  const empty = !team;
  const meta = resolveTeamMeta(team);
  return `
    <div class="bracket-team ${side} ${empty ? "empty" : ""}" data-side="${side}">
      <div class="team-mark-stack">
        ${buildTeamMark(team)}
        <span class="team-code">${empty ? "" : meta.code}</span>
      </div>
    </div>
  `;
}

function buildBracketSlot(stage, index, match = null, empty = false, revealResult = false) {
  const homeTeam = match ? match.home_team : "";
  const awayTeam = match ? match.away_team : "";
  const hasPenalties = revealResult && match && match.decided_by === "penalties";
  return `
    <article class="bracket-slot ${empty ? "placeholder" : ""}" data-stage="${stage}" data-match-index="${index}">
      <div class="bracket-slot-body">
        ${buildBracketTeam(homeTeam, "home")}
        <div class="bracket-scorebox">
          <div class="bracket-minute visually-hidden" data-role="minute"></div>
          <div class="bracket-score" data-role="score">${match && revealResult ? `${match.home_goals} X ${match.away_goals}` : ""}</div>
          <div class="bracket-penalties ${hasPenalties ? "" : "visually-hidden"}" data-role="penalties">${hasPenalties ? `${match.penalty_home_goals} x ${match.penalty_away_goals}` : ""}</div>
        </div>
        ${buildBracketTeam(awayTeam, "away")}
      </div>
    </article>
  `;
}

function buildBracketColumn(
  stage,
  totalCount,
  initialMatches = [],
  revealResult = false,
  side = "left",
  startIndex = 0,
  endIndex = totalCount,
) {
  const slots = [];
  for (let index = startIndex; index < endIndex; index += 1) {
    slots.push(buildBracketSlot(stage, index, initialMatches[index] || null, !initialMatches[index], revealResult));
  }
  return `
    <section class="bracket-column ${stage} bracket-column--${side}" data-stage-column="${stage}" style="--slot-count: ${endIndex - startIndex};">
      <div class="bracket-column-body">
        ${slots.join("")}
      </div>
    </section>
  `;
}

function buildPodiumTeam(team) {
  const empty = !team;
  const meta = resolveTeamMeta(team);
  return `
    <div class="podium-team ${empty ? "empty" : ""}">
      <div class="team-mark-stack">
        ${buildTeamMark(team)}
        <span class="team-code">${empty ? "TBD" : meta.code}</span>
      </div>
    </div>
  `;
}

function buildBronzeCard() {
  return `
    <article class="bronze-card">
      <div class="bracket-column-head">
        <h4>Third Place</h4>
      </div>
      <div class="bracket-slot bronze" data-stage="third_place" data-match-index="0">
        <div class="bracket-slot-body">
          <div class="bracket-team home empty" data-side="home">
            <div class="team-mark-stack">
              ${buildTeamMark("")}
              <span class="team-code">TBD</span>
            </div>
          </div>
          <div class="bracket-team away empty" data-side="away">
            <div class="team-mark-stack">
              ${buildTeamMark("")}
              <span class="team-code">TBD</span>
            </div>
          </div>
        </div>
        <div class="bracket-slot-footer">
          <div class="bracket-score" data-role="score"></div>
          <div class="bracket-winner visually-hidden" data-role="winner"></div>
        </div>
      </div>
    </article>
  `;
}

function renderGroupLayout() {
  simulationScreen.classList.add("groups-mode");
  stageContent.className = "workspace groups-view";
  stageContent.innerHTML = `
    <article class="feed-panel">
      <div class="panel-head">
        <h3>Match Feed</h3>
        <span id="feedStatus">Group stage in progress</span>
      </div>
      <div id="matchFeed" class="match-feed"></div>
    </article>

    <aside class="detail-panel">
      <div id="detailPanel" class="standing-stack"></div>
    </aside>
  `;
}

function renderThirdsLayout() {
  simulationScreen.classList.add("groups-mode");
  stageContent.className = "thirds-layout";
  stageContent.innerHTML = `
    <article class="thirds-card thirds-card--full">
      <div class="stage-panel-head">
        <h3>Best Third-Placed Teams</h3>
      </div>
      <div id="thirdsPanel" class="third-stack"></div>
    </article>
  `;
}

function renderKnockoutLayout() {
  simulationScreen.classList.remove("groups-mode");
  stageContent.className = "knockout-layout";
  const grouped = groupMatchesByStage(appState.result?.knockout_results || []);
  const columnStages = KNOCKOUT_STAGES.filter((stage) => stage !== "final");
  const stageSizes = {
    round_of_32: 16,
    round_of_16: 8,
    quarterfinal: 4,
    semifinal: 2,
    final: 1,
  };
  const stageColumnsLeft = columnStages
    .map((stage) => {
      const count = stageSizes[stage];
      const half = count / 2;
      const initialMatches = stage === "round_of_32" ? grouped[stage] || [] : [];
      return buildBracketColumn(stage, count, initialMatches, false, "left", 0, half);
    })
    .join("");
  const stageColumnsRight = [...columnStages]
    .reverse()
    .map((stage) => {
      const count = stageSizes[stage];
      const half = count / 2;
      const initialMatches = stage === "round_of_32" ? grouped[stage] || [] : [];
      return buildBracketColumn(stage, count, initialMatches, false, "right", half, count);
    })
    .join("");
  const finalColumn = buildBracketColumn("final", 1, [], false, "center", 0, 1);
  stageContent.innerHTML = `
    <article class="knockout-board knockout-board--full knockout-board--centered">
      <div class="knockout-podium knockout-podium--top">
        <div class="podium-label">Champions</div>
        <div id="championPodiumTeam">${buildPodiumTeam("")}</div>
      </div>
      <div id="bracketBoard" class="bracket-board">
        <div class="bracket-half bracket-half--left">${stageColumnsLeft}</div>
        <div class="bracket-final-column">${finalColumn}</div>
        <div class="bracket-half bracket-half--right">${stageColumnsRight}</div>
      </div>
      <div class="knockout-third-place">
        <div class="podium-label">3rd Place</div>
        ${buildBracketSlot("third_place", 0, null, true, false)}
      </div>
    </article>
  `;

  appState.bracketSlots = {};
  document.querySelectorAll(".bracket-slot[data-stage]").forEach((slot) => {
    const stage = slot.dataset.stage;
    const matchIndex = Number(slot.dataset.matchIndex || "0");
    appState.bracketSlots[`${stage}:${matchIndex}`] = {
      slot,
      home: slot.querySelector('[data-side="home"] .team-code'),
      away: slot.querySelector('[data-side="away"] .team-code'),
      minute: slot.querySelector('[data-role="minute"]'),
      score: slot.querySelector('[data-role="score"]'),
      penalties: slot.querySelector('[data-role="penalties"]'),
    };
  });
}

function buildGroupTracker(groupResults, activeIndex, selectedIndex = activeIndex, interactive = false) {
  const chips = groupResults
    .map((group, index) => {
      const classes = ["group-chip"];
      if (index < activeIndex) {
        classes.push("done");
      } else if (index === activeIndex) {
        classes.push("active");
      }
      if (index === selectedIndex) {
        classes.push("selected");
      }
      const disabled = interactive ? "" : "disabled";
      return `<button type="button" class="${classes.join(" ")}" data-group-index="${index}" ${disabled}>Group ${group.group_name}</button>`;
    })
    .join("");

  return `
    <div class="group-tracker">${chips}</div>
  `;
}

function createLiveStandings(group) {
  return group.standings.map((row) => ({
    team: row.team,
    points: 0,
    goals_for: 0,
    goals_against: 0,
    goal_difference: 0,
  }));
}

function sortStandings(rows) {
  return [...rows].sort((a, b) => {
    if (b.points !== a.points) {
      return b.points - a.points;
    }
    if (b.goal_difference !== a.goal_difference) {
      return b.goal_difference - a.goal_difference;
    }
    if (b.goals_for !== a.goals_for) {
      return b.goals_for - a.goals_for;
    }
    return a.team.localeCompare(b.team);
  });
}

function updateLiveStandings(groupName, match) {
  const rows = appState.liveGroupStandings[groupName];
  if (!rows) {
    return;
  }
  const home = rows.find((row) => row.team === match.home_team);
  const away = rows.find((row) => row.team === match.away_team);
  if (!home || !away) {
    return;
  }

  home.goals_for += match.home_goals;
  home.goals_against += match.away_goals;
  home.goal_difference = home.goals_for - home.goals_against;
  away.goals_for += match.away_goals;
  away.goals_against += match.home_goals;
  away.goal_difference = away.goals_for - away.goals_against;

  if (match.home_goals > match.away_goals) {
    home.points += 3;
  } else if (match.home_goals < match.away_goals) {
    away.points += 3;
  } else {
    home.points += 1;
    away.points += 1;
  }

  appState.liveGroupStandings[groupName] = sortStandings(rows);
}

function buildStandingsTable(group, standings = group.standings) {
  const rows = standings
    .map((row, index) => `
      <tr class="${index < 2 ? "qualifies" : ""}">
        <td>${index + 1}</td>
        <td>${row.team}</td>
        <td>${row.points}</td>
        <td>${goalDifference(row)}</td>
        <td>${row.goals_for}</td>
      </tr>
    `)
    .join("");

  return `
    <div class="standings-card">
      <h4>Group ${group.group_name} Standings</h4>
      <table class="standings-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Team</th>
            <th>Pts</th>
            <th>GD</th>
            <th>GF</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderGroupDetails(groupResults, activeIndex, interactive = false, clearFeed = false) {
  const detailPanel = document.getElementById("detailPanel");
  const matchFeed = document.getElementById("matchFeed");
  if (!detailPanel) {
    return;
  }
  detailPanel.innerHTML = `
    ${buildGroupTracker(groupResults, activeIndex, appState.selectedGroupIndex, interactive)}
    ${buildStandingsTable(
      groupResults[activeIndex],
      appState.liveGroupStandings[groupResults[activeIndex].group_name] || groupResults[activeIndex].standings,
    )}
  `;

  if (matchFeed && clearFeed) {
    matchFeed.innerHTML = "";
  }

  if (interactive) {
    const buttons = Array.from(detailPanel.querySelectorAll("[data-group-index]"));
    buttons.forEach((button) => {
      button.addEventListener("click", () => {
        const nextIndex = Number(button.dataset.groupIndex);
        appState.selectedGroupIndex = nextIndex;
        renderGroupDetails(groupResults, nextIndex, true);
        renderGroupFeed(groupResults[nextIndex]);
      });
    });
  }
}

function renderGroupFeed(group) {
  const matchFeed = document.getElementById("matchFeed");
  if (!matchFeed) {
    return;
  }
  matchFeed.innerHTML = "";
  group.matches.forEach((match) => {
    matchFeed.appendChild(buildMatchCard(match, "group"));
  });
  matchFeed.scrollTop = matchFeed.scrollHeight;
}

function renderThirdsTable(result) {
  const panel = document.getElementById("thirdsPanel");
  if (!panel) {
    return;
  }

  const thirds = result.group_results
    .map((group) => ({
      group: group.group_name,
      row: group.standings[2],
    }))
    .sort((a, b) => {
      if (b.row.points !== a.row.points) {
        return b.row.points - a.row.points;
      }
      const gdDiff = goalDifference(b.row) - goalDifference(a.row);
      if (gdDiff !== 0) {
        return gdDiff;
      }
      if (b.row.goals_for !== a.row.goals_for) {
        return b.row.goals_for - a.row.goals_for;
      }
      return a.row.team.localeCompare(b.row.team);
    });

  panel.innerHTML = `
    <div class="standings-card">
      <h4>Third-Place Ranking</h4>
      <table class="thirds-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Team</th>
            <th>Group</th>
            <th>Pts</th>
            <th>GD</th>
            <th>GF</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${thirds
            .map((item, index) => {
              const qualifies = index < 8;
              return `
                <tr class="${qualifies ? "qualifies" : ""}">
                  <td>${index + 1}</td>
                  <td>${item.row.team}</td>
                  <td>${item.group}</td>
                  <td>${item.row.points}</td>
                  <td>${goalDifference(item.row)}</td>
                  <td>${item.row.goals_for}</td>
                  <td>${qualifies ? "Qualified" : "Out"}</td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderChampionPanel(result) {
  return result;
}

function setBracketTeam(stage, matchIndex, side, team, flash = false) {
  const key = `${stage}:${matchIndex}`;
  const refs = appState.bracketSlots[key];
  if (!refs) {
    return;
  }
  const teamNode = side === "home" ? refs.home : refs.away;
  const sideNode = refs.slot.querySelector(`[data-side="${side}"]`);
  if (!teamNode || !sideNode) {
    return;
  }
  const meta = resolveTeamMeta(team);
  teamNode.textContent = team ? meta.code : "";
  const mark = sideNode.querySelector(".team-mark");
  if (mark) {
    mark.style.setProperty("--team-tone", getTeamTone(team || "placeholder"));
    mark.classList.toggle("empty", !team);
    mark.classList.toggle("flag-mark", Boolean(team) && Boolean(meta.flag));
    if (team && meta.flag) {
      mark.innerHTML = `<img src="${meta.flag}" alt="${meta.team}" aria-hidden="true" />`;
    } else if (team) {
      mark.innerHTML = `<span>${meta.code}</span>`;
    } else {
      mark.innerHTML = "";
    }
  }
  sideNode.classList.toggle("empty", !team);
  const homeFilled = (refs.slot.querySelector('[data-side="home"] .team-code')?.textContent || "").trim().length > 0;
  const awayFilled = (refs.slot.querySelector('[data-side="away"] .team-code')?.textContent || "").trim().length > 0;
  refs.slot.classList.toggle("placeholder", !(homeFilled && awayFilled));
  if (flash && team) {
    sideNode.classList.remove("arrived");
    void sideNode.offsetWidth;
    sideNode.classList.add("arrived");
  }
}

function setBracketMatch(stage, matchIndex, match) {
  const key = `${stage}:${matchIndex}`;
  const refs = appState.bracketSlots[key];
  if (!refs) {
    return;
  }
  refs.score.textContent = `${match.home_goals} X ${match.away_goals}`;
  if (refs.penalties) {
    const hasPenalties = match.decided_by === "penalties" && match.penalty_home_goals !== null && match.penalty_away_goals !== null;
    refs.penalties.textContent = hasPenalties ? `PEN ${match.penalty_home_goals} x ${match.penalty_away_goals}` : "";
    refs.penalties.classList.toggle("visually-hidden", !hasPenalties);
  }
  if (refs.minute) {
    const wentToExtraTime = match.regulation_home_goals !== null && match.regulation_home_goals !== undefined;
    refs.minute.textContent = wentToExtraTime ? "AET" : "";
    refs.minute.classList.toggle("visually-hidden", !wentToExtraTime);
  }
  refs.slot.classList.remove("placeholder");
  refs.slot.classList.add("resolved");
  const homeSide = refs.slot.querySelector('[data-side="home"]');
  const awaySide = refs.slot.querySelector('[data-side="away"]');
  if (homeSide) {
    homeSide.classList.toggle("winner", match.winner === match.home_team);
  }
  if (awaySide) {
    awaySide.classList.toggle("winner", match.winner === match.away_team);
  }
}

function promoteWinnerToNextStage(stage, matchIndex, winner) {
  const currentStageIndex = KNOCKOUT_STAGES.indexOf(stage);
  const nextStage = KNOCKOUT_STAGES[currentStageIndex + 1];
  if (!nextStage) {
    return;
  }
  const nextMatchIndex = Math.floor(matchIndex / 2);
  const side = matchIndex % 2 === 0 ? "home" : "away";
  setBracketTeam(nextStage, nextMatchIndex, side, winner, true);
}

function setBronzeMatch(match) {
  return match;
}

function updateChampionPanel(result) {
  return result;
}

function resetToIntro() {
  appState.phase = "intro";
  appState.actualPhase = "intro";
  appState.result = null;
  appState.currentGroupIndex = 0;
  appState.selectedGroupIndex = 0;
  appState.liveGroupStandings = {};
  appState.reached = { groups: false, thirds: false, knockout: false, final: false };
  appState.animating = false;
  introScreen.classList.remove("is-hidden");
  simulationScreen.classList.add("is-hidden");
  setPhase("groups");
  setStatus("Ready to simulate");
  advanceButton.classList.add("is-hidden");
  advanceButton.textContent = "Advance";
  stageContent.innerHTML = "";
}

async function animateGroupStage(result) {
  appState.animating = true;
  setPhase("groups");
  stageTitle.textContent = "Group Stage";
  stageText.textContent = "Matches are being simulated in real time, group by group.";
  renderGroupLayout();
  hideAdvanceButton();
  setStatus(`Simulating group stage... Speed: ${appState.speed}`);

  appState.liveGroupStandings = {};
  result.group_results.forEach((group) => {
    appState.liveGroupStandings[group.group_name] = createLiveStandings(group);
  });

  const matchFeed = document.getElementById("matchFeed");
  for (let groupIndex = 0; groupIndex < result.group_results.length; groupIndex += 1) {
    const group = result.group_results[groupIndex];
    appState.currentGroupIndex = groupIndex;
    appState.selectedGroupIndex = groupIndex;
    renderGroupDetails(result.group_results, groupIndex, false, true);

    for (const match of group.matches) {
      matchFeed.appendChild(buildMatchCard(match, "group"));
      updateLiveStandings(group.group_name, match);
      renderGroupDetails(result.group_results, groupIndex, false, false);
      matchFeed.scrollTop = matchFeed.scrollHeight;
      await wait(appState.animateDelay);
    }

    renderGroupDetails(result.group_results, groupIndex, false, false);
    await wait(Math.round(appState.animateDelay * 0.45));
  }

  setStatus("Group stage complete. Advance to the best third-placed teams.");
  appState.selectedGroupIndex = result.group_results.length - 1;
  renderGroupDetails(result.group_results, appState.selectedGroupIndex, true, false);
  renderGroupFeed(result.group_results[appState.selectedGroupIndex]);
  appState.reached.groups = true;
  appState.animating = false;
  refreshPhaseNav();
  setAdvanceButton("Advance", () => showThirdsStage(result));
}

function showGroupsSnapshot(result) {
  setPhase("groups");
  stageTitle.textContent = "Group Stage";
  stageText.textContent = "Group stage results.";
  renderGroupLayout();
  const matchFeed = document.getElementById("matchFeed");
  result.group_results.forEach((group) => {
    group.matches.forEach((match) => {
      matchFeed.appendChild(buildMatchCard(match, "group"));
    });
  });
  appState.selectedGroupIndex = result.group_results.length - 1;
  renderGroupDetails(result.group_results, appState.selectedGroupIndex, true, false);
  renderGroupFeed(result.group_results[appState.selectedGroupIndex]);
  setStatus("Viewing the completed group stage.");
  setAdvanceButton("Continue", resumeActualPhase);
}

function showThirdsStage(result, isPeek = false) {
  setPhase("thirds");
  stageTitle.textContent = "Best Thirds";
  stageText.textContent = "The eight strongest third-placed teams are selected here.";
  renderThirdsLayout();
  renderThirdsTable(result);
  if (isPeek) {
    setStatus("Viewing the best third-placed teams.");
    setAdvanceButton("Continue", resumeActualPhase);
    return;
  }
  appState.actualPhase = "thirds";
  appState.reached.thirds = true;
  refreshPhaseNav();
  setStatus("Third-place table ready. Advance to the knockout stage.");
  setAdvanceButton("Advance", () => animateKnockoutStage(result));
}

async function animateKnockoutStage(result) {
  appState.animating = true;
  appState.phase = "knockout";
  appState.actualPhase = "knockout";
  setPhase("knockout");
  stageTitle.textContent = "Knockout";
  stageText.textContent = "Knockout rounds are simulated first. The final starts when you press Play Final.";
  renderKnockoutLayout();
  hideAdvanceButton();
  setStatus(`Animating knockout rounds... Speed: ${appState.speed}`);

  const grouped = groupMatchesByStage(result.knockout_results);
  const stageLabels = {
    round_of_32: "Starting Round of 32",
    round_of_16: "Round of 16",
    quarterfinal: "Quarterfinals",
    semifinal: "Semifinals",
    final: "Final",
  };

  for (const stage of KNOCKOUT_STAGES.filter((stage) => stage !== "final")) {
    const stageMatches = grouped[stage] || [];
    const columns = document.querySelectorAll(`[data-stage-column="${stage}"]`);
    columns.forEach((column) => column.classList.add("active"));
    setStatus(stageLabels[stage] || stageDisplayName(stage));

    for (let matchIndex = 0; matchIndex < stageMatches.length; matchIndex += 1) {
      const match = stageMatches[matchIndex];
      setBracketMatch(stage, matchIndex, match);
      promoteWinnerToNextStage(stage, matchIndex, match.winner);
      await wait(appState.animateDelay);
    }

    columns.forEach((column) => {
      column.classList.remove("active");
      column.classList.add("done");
    });

  }

  await animateThirdPlaceMatch(grouped);

  stageTitle.textContent = "Final Ready";
  stageText.textContent = "The finalists are ready. Press Play Final to start the minute-by-minute simulation.";
  setStatus("Semifinals complete. Press Play Final to begin.");
  appState.reached.knockout = true;
  appState.animating = false;
  refreshPhaseNav();
  setAdvanceButton("Play Final", () => animateFinalMatch(result, grouped));
}

function showKnockoutSnapshot(result) {
  setPhase("knockout");
  renderKnockoutLayout();

  const grouped = groupMatchesByStage(result.knockout_results);
  for (const stage of KNOCKOUT_STAGES.filter((stage) => stage !== "final")) {
    const stageMatches = grouped[stage] || [];
    stageMatches.forEach((match, matchIndex) => {
      setBracketMatch(stage, matchIndex, match);
      promoteWinnerToNextStage(stage, matchIndex, match.winner);
    });
    document.querySelectorAll(`[data-stage-column="${stage}"]`).forEach((column) => {
      column.classList.add("done");
    });
  }

  const thirdPlaceMatch = grouped.third_place?.[0];
  if (thirdPlaceMatch) {
    setBracketTeam("third_place", 0, "home", thirdPlaceMatch.home_team);
    setBracketTeam("third_place", 0, "away", thirdPlaceMatch.away_team);
    setBracketMatch("third_place", 0, thirdPlaceMatch);
  }

  if (appState.reached.final) {
    const finalMatch = grouped.final?.[0];
    if (finalMatch) {
      setBracketTeam("final", 0, "home", finalMatch.home_team, false);
      setBracketTeam("final", 0, "away", finalMatch.away_team, false);
      setBracketMatch("final", 0, finalMatch);
      updateFinalScore(finalMatch.home_goals, finalMatch.away_goals);
      updateFinalMinute("FT");
    }
    const championNode = document.getElementById("championPodiumTeam");
    if (championNode) {
      championNode.innerHTML = buildPodiumTeam(result.champion);
    }
    document.querySelector('[data-stage-column="final"]')?.classList.add("done");
    stageTitle.textContent = "Champions";
    stageText.textContent = "The tournament is complete.";
    setStatus("Full time.");
    setAdvanceButton("Restart Simulation", resetToIntro);
  } else {
    stageTitle.textContent = "Final Ready";
    stageText.textContent = "The finalists are ready. Press Play Final to start the minute-by-minute simulation.";
    setStatus("Semifinals complete. Press Play Final to begin.");
    setAdvanceButton("Play Final", () => animateFinalMatch(result, grouped));
  }
}

async function animateThirdPlaceMatch(grouped) {
  const thirdPlaceMatch = grouped.third_place?.[0];
  if (!thirdPlaceMatch) {
    return;
  }

  setStatus("Third-place match");
  setBracketTeam("third_place", 0, "home", thirdPlaceMatch.home_team, true);
  setBracketTeam("third_place", 0, "away", thirdPlaceMatch.away_team, true);
  await wait(appState.animateDelay);
  setBracketMatch("third_place", 0, thirdPlaceMatch);
  await wait(appState.animateDelay);
}

function buildGoalEventsInRange(homeGoals, awayGoals, startMinute, endMinute) {
  const teams = [];
  let homeRemaining = homeGoals;
  let awayRemaining = awayGoals;
  while (homeRemaining > 0 || awayRemaining > 0) {
    if (homeRemaining > 0 && (awayRemaining === 0 || homeRemaining >= awayRemaining)) {
      teams.push("home");
      homeRemaining -= 1;
    } else {
      teams.push("away");
      awayRemaining -= 1;
    }
  }

  const totalGoals = teams.length;
  const usedMinutes = new Set();
  const span = endMinute - startMinute - 2;
  return teams.map((team, goalIndex) => {
    let minute = startMinute + Math.max(2, Math.round(((goalIndex + 1) * span) / (totalGoals + 1)));
    while (usedMinutes.has(minute)) {
      minute = Math.min(endMinute - 1, minute + 1);
    }
    usedMinutes.add(minute);
    return { minute, team };
  }).sort((a, b) => a.minute - b.minute);
}

function updateFinalScore(homeGoals, awayGoals) {
  const refs = appState.bracketSlots["final:0"];
  if (refs?.score) {
    refs.score.textContent = `${homeGoals} X ${awayGoals}`;
  }
}

function updateFinalMinute(label) {
  const refs = appState.bracketSlots["final:0"];
  if (refs?.minute) {
    refs.minute.textContent = label;
    refs.minute.classList.remove("visually-hidden");
  }
}

async function animatePenaltyShootout(match, delay) {
  if (match.decided_by !== "penalties") {
    return;
  }
  const refs = appState.bracketSlots["final:0"];
  if (!refs?.penalties) {
    return;
  }

  const homeTarget = Number(match.penalty_home_goals || 0);
  const awayTarget = Number(match.penalty_away_goals || 0);
  const rounds = Math.max(homeTarget, awayTarget);
  refs.penalties.classList.remove("visually-hidden");
  for (let round = 1; round <= rounds; round += 1) {
    const homeScore = Math.min(round, homeTarget);
    const awayScore = Math.min(round, awayTarget);
    refs.penalties.textContent = `PEN ${homeScore} x ${awayScore}`;
    updateFinalMinute("PEN");
    setStatus("Final in progress");
    await wait(Math.max(180, Math.round(delay * 0.8)));
  }
}

async function animateFinalMatch(result, grouped) {
  const finalMatch = grouped.final?.[0];
  if (!finalMatch) {
    setAdvanceButton("Restart Simulation", resetToIntro);
    return;
  }

  appState.animating = true;
  refreshPhaseNav();
  hideAdvanceButton();
  setStatus("Final in progress");
  const finalColumn = document.querySelector('[data-stage-column="final"]');
  finalColumn?.classList.add("active");
  updateFinalMinute("0'");
  updateFinalScore(0, 0);

  const wentToExtraTime =
    finalMatch.regulation_home_goals !== null && finalMatch.regulation_home_goals !== undefined;
  const regHomeGoals = wentToExtraTime ? finalMatch.regulation_home_goals : finalMatch.home_goals;
  const regAwayGoals = wentToExtraTime ? finalMatch.regulation_away_goals : finalMatch.away_goals;
  const etHomeGoals = wentToExtraTime ? finalMatch.home_goals - regHomeGoals : 0;
  const etAwayGoals = wentToExtraTime ? finalMatch.away_goals - regAwayGoals : 0;

  const regulationEvents = buildGoalEventsInRange(regHomeGoals, regAwayGoals, 0, 90);
  let homeGoals = 0;
  let awayGoals = 0;
  const minuteDelay = Math.max(90, Math.round(appState.animateDelay * 0.35));

  for (let minute = 1; minute <= 90; minute += 1) {
    const minuteEvents = regulationEvents.filter((event) => event.minute === minute);
    minuteEvents.forEach((event) => {
      if (event.team === "home") {
        homeGoals += 1;
      } else {
        awayGoals += 1;
      }
    });
    updateFinalScore(homeGoals, awayGoals);
    updateFinalMinute(`${minute}'`);
    setStatus("Final in progress");
    await wait(minuteDelay);
  }

  if (wentToExtraTime) {
    setStatus("Full time. The final goes to extra time.");
    await wait(Math.max(350, minuteDelay * 2));
    updateFinalMinute("ET");
    setStatus("Extra time in progress");
    await wait(Math.max(350, minuteDelay));

    const extraTimeEvents = buildGoalEventsInRange(etHomeGoals, etAwayGoals, 90, 120);
    for (let minute = 91; minute <= 120; minute += 1) {
      const minuteEvents = extraTimeEvents.filter((event) => event.minute === minute);
      minuteEvents.forEach((event) => {
        if (event.team === "home") {
          homeGoals += 1;
        } else {
          awayGoals += 1;
        }
      });
      updateFinalScore(homeGoals, awayGoals);
      updateFinalMinute(`${minute}'`);
      setStatus("Extra time in progress");
      await wait(minuteDelay);
    }
    setStatus(
      homeGoals === awayGoals
        ? "Extra time ends level. Penalties will decide the champion."
        : "Extra time ends."
    );
  } else {
    setStatus("Full time.");
  }

  await wait(Math.max(350, minuteDelay * 2));
  await animatePenaltyShootout(finalMatch, minuteDelay);
  setBracketMatch("final", 0, finalMatch);
  finalColumn?.classList.remove("active");
  finalColumn?.classList.add("done");

  const championNode = document.getElementById("championPodiumTeam");
  if (championNode) {
    championNode.innerHTML = buildPodiumTeam(result.champion);
  }

  appState.actualPhase = "final";
  appState.reached.final = true;
  appState.animating = false;
  refreshPhaseNav();
  setAdvanceButton("Restart Simulation", resetToIntro);
}

async function runSimulation() {
  startButton.disabled = true;
  setStatus("Generating tournament simulation...");

  try {
    const response = await fetch(`${API_BASE}/api/simulate`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tournament: tournamentConfig,
        // demoProfiles only makes sense alongside demoTournament's placeholder
        // team names. With the real WC_TOURNAMENT draw, omit profiles so the
        // backend loads the real team ratings (data/raw/team_profiles.csv)
        // instead of every team defaulting to zero strength.
        profiles: window.WC_TOURNAMENT ? undefined : demoProfiles,
        runs: 1,
        seed: Date.now() % 1000000,
      }),
    });

    if (!response.ok) {
      throw new Error(`Simulation request failed: ${response.status}`);
    }

    const payload = await response.json();
    appState.result = payload.result;

    showSimulationScreen();
    await animateGroupStage(appState.result);
  } catch (error) {
    console.error(error);
    setStatus("Simulation failed. Check the backend logs.");
    startButton.disabled = false;
    return;
  }

  startButton.disabled = false;
}

advanceButton.addEventListener("click", () => {
  if (appState.advanceHandler) {
    appState.advanceHandler();
  }
});

startButton.addEventListener("click", runSimulation);

speedButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setSpeed(button.dataset.speed);
  });
});

phasePills.forEach((pill) => {
  pill.addEventListener("click", () => {
    goToPhase(pill.dataset.phase);
  });
});

setSpeed("medium");
