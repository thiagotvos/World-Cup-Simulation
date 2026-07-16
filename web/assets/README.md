# Web Assets

Place visual assets for the frontend here.

Recommended structure:

- `flags/` for national team flag images used in group cards and bracket slots
- `logos/` if you later decide to add competition or brand logos

Suggested flag format:

- PNG or SVG
- square or circular crop
- consistent size, such as `128x128` or `256x256`

The frontend reads flag paths from `web/config.js` through `window.WC_TEAM_META`.
