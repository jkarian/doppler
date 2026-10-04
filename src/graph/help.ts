// Plain-language help for node settings, shown as pop-ups in the music monitor (and anywhere else that
// wants it). A node type's own `doc` on an input wins; these fill in the rest. Keys: "Type.input", or
// just "input" for settings that mean the same thing everywhere.

import { NODE_TYPES } from "./nodes.ts";

const HELP: Record<string, string> = {
  // Shared
  intensity: "How bright it is.",
  color: "Its colour.",
  level: "0-1 on/off level, usually wired from a Setup node so the song switches it.",
  decay: "Seconds for the pulse to fade after each hit.",
  attack: "Seconds to rise when it switches on.",
  release: "Seconds to fade when it switches off.",
  reach: "How far into the canyon the light carries before it fades (0 = only near rock, 1 = all the way to the far land).",
  width: "Beam thickness.",
  glow: "How far the soft glow around the beam spreads, in beam widths.",
  hit: "How brightly it marks the rock where it lands.",
  seed: "Changes the random choices (positions, aims) while keeping them repeatable.",
  count: "How many.",
  window: "Seconds averaged over: longer = smoother and slower.",

  // Sun
  "Sun.gain": "Overall sun brightness: multiplies whatever drives it. Raise it to make the sun stand out from the other lights (the brightness cap can limit how far it shows).",
  "Sun.floorLight": "How much sun falls on the floor: flat ground lower than floorBelow (canyon floor, river) and the cave floor around us. 0 = none; 1 = lit like everything else. Plateau tops and high ledges always get sun.",
  "Sun.bounce": "Warm light bounced off sunlit rock and the bright sky into the faces turned away from the sun, which otherwise stay dark when we look toward the sun. Grows as the sun climbs. 0 = none.",
  "Sun.shadowSoftness": "Edge of the sun's cast shadows: 0 = hard and crisp like real sunlight, 1 = soft.",
  "Sun.shadowDepth": "How solid far ridges are when casting shadows (how deep behind their face the rock counts as solid, as a fraction of their distance). Higher = cliffs throw longer, fuller shadows across the canyon; too high and light can't reach anything behind a ridge.",
  "Sun.caveDepth": "How deep the rock behind the cave walls around us counts as solid when the sun casts shadows (multiple of the wall's distance). Raise it if sunlight leaks onto cave walls that the rim of the opening should shade.",
  "Sun.floorBelow": "Real height in feet relative to the camera: flat ground below this is floor (on the canyon: river about -5900, plateau tops about -2900, so -4300 splits them).",
  "Sun.maxArc": "Highest the sun may climb, in real degrees above the horizon. It eases into this ceiling instead of stopping dead.",
  "Sun.azimuth": "Where the sun sits left-right, in real degrees (0 = straight ahead). Further to the side, its light rakes across the canyon walls instead of coming from behind them.",
  "Sun.enabled": "1 = sun on, 0 = off.",
  "Sun.rays": "Visible sun shafts in the air.",
  "Sun.flare": "Lens flare strength when the sun is in view.",
  "Sun.skyBoost": "Extra sky glow (pulses with the kick).",
  "SunTint.warmth": "Shifts the sun's colour range: above 1 stays red-orange longer as it rises, below 1 turns white sooner.",

  // Sun motion and on/off
  "Spurts.minGap": "Fewest seconds between sun jumps on strong bars (drops always jump).",
  "Spurts.sizeMin": "Smallest jump on a strong bar, in degrees (scaled by how intense the music is).",
  "Spurts.sizeMax": "Biggest jump on a strong bar, in degrees.",
  "Spurts.dropMin": "Smallest jump on a drop, in degrees.",
  "Spurts.dropMax": "Biggest jump on a drop, in degrees.",
  "Spurts.rise": "Seconds the sun takes to rise to the top of a jump.",
  "Spurts.hold": "Seconds it holds at the top.",
  "Spurts.returnBase": "Seconds to sink back down (plus returnPerDeg for each degree of height).",
  "Spurts.returnPerDeg": "Extra seconds to sink back per degree it rose: higher jumps take longer to come down.",
  "Spurts.max": "Highest the jumps can lift the sun above its resting height, in degrees.",
  "Spurts.heaviness": "How heavy the sun feels: lower = slower, more inertia.",
  "Gate.on": "How strong the kick and intensity must get (fraction of the track's range) to switch the sun on.",
  "Gate.off": "How low they must drop to switch it off. Keep below 'on' so it doesn't flicker.",
  "Gate.attack": "Seconds to snap on.",
  "Gate.release": "Seconds to fade off.",

  // Sections, drops, pulses
  "Section.quiet": "Value in quiet sections (intro, breakdown, outro).",
  "Section.buildStart": "Value at the start of a build.",
  "Section.buildEnd": "Value at the end of a build (it ramps from start to end).",
  "Section.drop": "Value in drops.",
  "Section.normal": "Value in normal sections.",
  "Section.ease": "Seconds to glide between values when the section changes.",
  "DropHit.window": "Bars of wind-up before a drop.",
  "Pump.attack": "Seconds for the pump to swell on each beat.",
  "Pump.release": "Seconds for it to let go.",
  "Beat.decay": "Seconds the beat pulse takes to fade.",
  "Phrase.decay": "Seconds the phrase-start pulse takes to fade.",

  // Setups
  "Setup.fadeIn": "Seconds to fade up when one of its sections starts. Short makes drops hit.",
  "Setup.fadeOut": "Seconds to fade down when its sections end.",

  // Nook lights
  "NookLights.radius": "How far each pool of light reaches, in feet (real size, so far pools look smaller). Each light can scale it: scroll over its ring in placement mode (N).",
  "NookLights.standoff": "How far in front of the rock each light sits, as a fraction of its reach. More = softer, wider pools.",
  "NookLights.near": "Distance in feet: lights nearer than this wait for big moments (drops); further ones fire on the tick-tock.",
  "NookLights.nearDecay": "Seconds for the near lights to fade after a big moment.",
  "NookLights.decay": "Seconds for the mid and far lights to fade after each tick or tock.",
  "NookLights.attack": "Seconds for a light to come on.",
  "NookLights.pattern": "0: hits walk through the lights in order. 1: ticks fire the left half, tocks the right. 2: random. 3: on every downbeat a wave runs from the nearest light out to the farthest, one light per sweepStep beats (near lights included; they still flash on drops too).",

  // Lasers
  "SkyLaser.from": "0 = beams from the sky, 1 = rigs standing on the rock.",
  "SkyLaser.elevMin": "Random rigs: lowest aim, degrees up.",
  "SkyLaser.elevMax": "Random rigs: highest aim, degrees up.",
  "SkyLaser.uMin": "Random spots: left edge of the area they're chosen in (0-1 across the picture).",
  "SkyLaser.uMax": "Random spots: right edge of the area.",
  "SkyLaser.vMin": "Random spots: top edge of the area (0-1 down the picture).",
  "SkyLaser.vMax": "Random spots: bottom edge of the area.",
  "SkyLaser.minDepth": "Random spots: only on rock at least this far away.",
  "SkyLaser.tilt": "Sky beams: random lean from vertical, degrees.",
  "SkyLaser.drift": "Degrees each beam sways while lit.",
  "SkyLaser.driftSpeed": "Sways per second.",
  "SkyLaser.sheet": "0 = beams, 1 = vertical curtains of light.",
  "SkyLaser.sheetWidth": "Curtain width, degrees.",
  "SkyLaser.fade": "How fast beams dim between re-aims (0 = steady).",
  "SkyLaser.scan": "1 turns each rig's beam into a scanning laser: a flickering triangular plane of light, like club lasers. Wire a Setup node's level here to switch it with the song.",
  "SkyLaser.scanSpread": "How wide each plane of light opens, degrees.",
  "SkyLaser.scanLines": "How finely each plane follows the rock that cuts it off (only its two edge lines are drawn).",
  "SkyLaser.scanBright": "Brightness of the plane itself (the lines and the marks on the rock follow intensity).",
  "SkyLaser.flicker": "How much the scanners flicker.",
  "SkyLaser.sweep": "Scanning planes: how far each swings to either side, degrees.",
  "SkyLaser.sweepSpeed": "Scanning planes: swings per second when the music is calm.",
  "SkyLaser.peak": "Wire a Peaks node's peak here: the swings get wider on peaks in the song.",
  "SkyLaser.peakPush": "Wire a Peaks node's push here: the swings speed up on peaks, smoothly.",
  "SkyLaser.peakWider": "At a full peak the swings are this many times wider again (2 = three times as wide).",
  "SkyLaser.peakFaster": "Extra swings per second at a full peak.",
  "Peaks.window": "Seconds of recent music a peak is measured against: a hit counts if it rises above that.",
  "Peaks.threshold": "How far above the recent level counts as a peak.",
  "Peaks.attack": "Seconds for the peak signal to rise.",
  "Peaks.release": "Seconds for it to fall back.",
  "Laser.originU": "Where the fixture sits, left-right (0-1 across the picture).",
  "Laser.originV": "Where the fixture sits, up-down (0-1 down the picture).",
  "Laser.originDepth": "0 = on the rock at that spot; more = floating in the air that far out.",
  "Laser.azimuth": "Aim left-right, degrees (0 = into the scene).",
  "Laser.elevation": "Aim up-down, degrees.",
  "Laser.roll": "Tilts the fan of beams around its aim.",
  "Laser.spread": "How wide the fan of beams opens, degrees.",
  "Laser.sheet": "Fills between the beams with a plane of light (0-1).",

  // Scan
  "Scan.position": "Where the scan slice is (0-1 across the scene).",
  "Scan.spacing": "Gap between trailing slices.",
  "Scan.thickness": "Slice thickness, feet.",
  "Scan.trail": "Glow left behind the moving slice, feet.",

  // Sky, tone, sync
  "Sky.mix": "0 = the photo's own sky, 1 = the colour gradient that follows the sun.",
  "Sky.brightness": "How bright the sky is.",
  "Sky.clouds": "How much of the photo's clouds shows as texture on the gradient.",
  "Sky.glow": "Warm glow around the sun in the sky.",
  "Sky.span": "How tall the gradient is, in degrees above the lowest open sky.",
  "Tone.baseDim": "Brightness of the dark night scene before any lights.",
  "Tone.baked": "How much of the original photo's daylight shows through (0 = none).",
  "Tone.cap": "Brightness ceiling: bright areas roll off softly toward it instead of clipping. Raise it to let the brightest lights (the sun) stand out more.",
  "Sync.lead": "Shifts the visuals against the audio, in milliseconds. Positive = visuals earlier. Use if lights land a little late or early.",
};

/** Help for one input of a node type: the table's, else its own doc, else the shared entry. */
export function inputHelp(type: string, input: string): string {
  const own = NODE_TYPES[type]?.inputs.find((i) => i.name === input)?.doc;
  return HELP[`${type}.${input}`] ?? own ?? HELP[input] ?? "";
}

/** The first sentence of a node type's description. */
export function nodeHelp(type: string): string {
  const doc = NODE_TYPES[type]?.doc ?? "";
  const end = doc.search(/[.:](\s|$)/);
  return end > 0 ? doc.slice(0, end + 1) : doc;
}
