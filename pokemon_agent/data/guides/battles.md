---
guide: battles
title: Battle reference — leaders, counters and Gen 1 quirks
source: derived from Pokemon Red/Blue game data
about: A lookup rather than a route. Leader and Elite Four rosters, the type interactions that actually decide those fights, and the Gen 1 mechanics that make Red behave unlike every later game. Cross-check rosters against the extracted trainer data when it is available.
---

## Gym leaders at a glance
<!-- slug: gym-leaders -->
<!-- summary: All eight rosters with the one move that beats each. -->

Badge order and rosters, with the counter that matters most:

1. **Brock** (Pewter, Boulder) — Geodude Lv12, Onix Lv14. Rock/Ground. Water and Grass do quadruple damage. Onix has huge Defense and terrible Special, so any special-class move (in Gen 1 that is Water, Grass, Fire, Ice, Electric, Psychic, Dragon) bypasses its Defense entirely. Watch Bide.
2. **Misty** (Cerulean, Cascade) — Staryu Lv18, Starmie Lv21. Pure Water. Electric and Grass. Starmie is fast and its Bubblebeam drops Speed.
3. **Lt. Surge** (Vermilion, Thunder) — Voltorb Lv21, Pikachu Lv18, Raichu Lv24. Pure Electric. Ground is a straight immunity: Dig or Earthquake on a Diglett, Dugtrio, Sandslash or Nidoking wins without taking a hit.
4. **Erika** (Celadon, Rainbow) — Victreebel Lv29, Tangela Lv24, Vileplume Lv29. Grass, two of them Poison. Fire, Ice, Flying, Psychic. Her powders (Sleep Powder, Stun Spore) cost more fights than her damage does.
5. **Koga** (Fuchsia, Soul) — Koffing Lv37, Muk Lv39, Koffing Lv37, Weezing Lv43. Pure Poison. Psychic and Ground. Selfdestruct and Toxic are the threats.
6. **Sabrina** (Saffron, Marsh) — Kadabra Lv38, Mr. Mime Lv37, Venomoth Lv38, Alakazam Lv43. Psychic. Almost nothing counters Psychic in Gen 1; Bug is the only real weakness and no good Bug move exists. Out-level her or out-speed her with strong Normal moves.
7. **Blaine** (Cinnabar, Volcano) — Growlithe Lv42, Ponyta Lv40, Rapidash Lv42, Arcanine Lv47. Pure Fire. Water, Rock, Ground.
8. **Giovanni** (Viridian, Earth) — Rhyhorn Lv45, Dugtrio Lv42, Nidoqueen Lv44, Nidoking Lv45, Rhydon Lv50. Ground, with Rock and Poison mixed in. Water, Grass and Ice all hit for double or quadruple. Surf plus Ice Beam clears the room.

Rhyhorn and Rhydon are Ground/Rock, which makes them quadruple weak to both Water and Grass. That is the single biggest type multiplier available against any leader.

## Elite Four and the Champion
<!-- slug: elite-four -->
<!-- summary: Five back-to-back fights, their rosters, and what beats each. -->

No healing between rooms, no Center, no shop. Whatever the team carries in is what finishes.

**Lorelei** — Dewgong Lv54, Cloyster Lv53, Slowbro Lv54, Jynx Lv56, Lapras Lv56. Ice with Water attached. Electric hits everything except Jynx. Fighting and Rock hit the Ice halves. Cloyster's Defense is enormous but its Special is ordinary, and it is quadruple weak to nothing — Thunderbolt is the practical answer. Slowbro is Water/Psychic and takes double from Electric, Grass and Bug.

**Bruno** — Onix Lv53, Hitmonchan Lv55, Hitmonlee Lv55, Onix Lv56, Machamp Lv58. Fighting and Rock. Psychic sweeps the entire room: it is super effective on all three Fighting types and neutral on the Onix. Water or Grass one-shots both Onix.

**Agatha** — Gengar Lv56, Golbat Lv56, Haunter Lv55, Arbok Lv58, Gengar Lv60. Ghost/Poison and Poison/Flying. Psychic hits her Poison types for double, and because of the Gen 1 Ghost bug (see the mechanics section) it also lands on Gengar and Haunter. Her real weapons are Hypnosis, Confuse Ray and Toxic — bring Full Heals.

**Lance** — Gyarados Lv58, Dragonair Lv56, Dragonair Lv56, Aerodactyl Lv60, Dragonite Lv62. Ice hits every single member: the Dragonair line for double, Aerodactyl for double via Flying, Dragonite for quadruple via Dragon/Flying. Blizzard or Ice Beam alone can carry this fight. Electric covers Gyarados and Aerodactyl.

**Champion (the rival)** — Pidgeot Lv61, Alakazam Lv59, Rhydon Lv61, plus a rotating trio (Gyarados/Arcanine/Exeggutor depending on his starter) and his starter at Lv63. No type theme, six Pokémon, and the only fight in the game where broad coverage matters more than one super-effective move.

## Gen 1 type chart quirks
<!-- slug: type-quirks -->
<!-- summary: Where Red's type chart differs from every later game. -->

Red and Blue predate two whole types and several rebalances. The differences change which counters actually work:

- **No Dark and no Steel.** Psychic therefore has exactly one weakness — Bug — and the only Bug attacks in the game are Twineedle, Pin Missile, Leech Life and Megahorn's absence. Psychic is the strongest type in Gen 1 by a wide margin.
- **The Ghost bug.** Ghost is coded as having no effect on Psychic, which is the reverse of the intended design. Psychic attacks still hit Ghosts normally. Combined with the point above, nothing meaningfully checks a Psychic type.
- **Bug beats Poison, and Poison beats Bug.** In Gen 1, Bug moves are super effective against Poison. Both directions are super effective. That makes Beedrill's Twineedle notable and makes Poison types poor Bug counters.
- **Ice is not weak to Steel** (no Steel), so Ice types only fear Fire, Fighting and Rock. Ice offence is close to unresisted apart from Water and Fire.
- **Special is one stat.** There is no Special Attack / Special Defense split. A Pokémon with high Special is both a strong special attacker and a strong special wall. Alakazam, Mewtwo and Starmie benefit enormously.
- **Move class is fixed by type, not by move.** Every Normal, Fighting, Flying, Ground, Rock, Bug, Ghost and Poison move is physical; every Fire, Water, Grass, Electric, Psychic, Ice and Dragon move is special. So Hyper Beam is physical and Fire Blast is special, always.

## Damage, crits and stat stages
<!-- slug: mechanics -->
<!-- summary: Crit formula, badge boosts, and how X items really work. -->

The mechanics that make Gen 1 strategies look strange:

- **Critical hits scale with base Speed.** Crit rate is base Speed divided by 512 for a normal move, roughly 0.4% to 25%. Fast Pokémon crit far more. High-crit moves (Slash, Razor Leaf, Crabhammer, Karate Chop) multiply that by 8, so a fast user of Slash crits almost always.
- **Critical hits ignore stat stages.** A crit recalculates damage as if no Attack boosts, no Defense drops and no Reflect existed. That means Screech and Swords Dance are undone by a crit, in both directions.
- **Focus Energy is broken.** It divides crit rate by 4 instead of multiplying by 4. Never use it.
- **Badge boosts.** Each badge raises a stat by roughly 12.5% in battle, and the boost is reapplied every time a stat-modifying move lands, so stacking effects compound in ways the displayed stats do not show.
- **X items are worth more than they look.** X Accuracy sets accuracy such that one-hit-KO moves (Horn Drill, Fissure, Guillotine) and every other move stop missing. That is why the speedrun route is built on X Accuracy and Horn Drill rather than on damage.
- **1/256 miss.** Every move has a floor miss chance of about 1/256 even at 100% accuracy. Plans that require a hit should account for it.
- **Thrash and Petal Dance lock the user in** for 3-4 turns and then confuse it. Confusion self-hits are typed Normal and ignore the type chart.
- **Partial trapping** (Wrap, Bind, Fire Spin, Clamp) locks the target out of acting for the duration in Gen 1, which makes it far stronger than in later games.

## Team building for a normal run
<!-- slug: team-building -->
<!-- summary: What coverage a six-slot team needs to finish Red comfortably. -->

A team that finishes Red without grinding needs four things: a Water move, an Ice move, a Ground move and something that outspeeds Psychic types.

**Coverage that matters.** Surf and Ice Beam between them are super effective against six of the eight gym leaders and against Lance's entire roster. Earthquake or Dig answers Surge and helps against Koga and Blaine. A Psychic move answers Bruno, Agatha and Koga. Beyond those four, extra coverage is mostly redundant.

**Reliable picks, by role.**
- Starter: any of the three works. Squirtle's line ends with Surf and Ice Beam and covers the most gyms.
- Water/Ice: Lapras (free from a scientist in Silph Co.) learns Surf, Ice Beam and Body Slam and has bulk.
- Ground: Dugtrio (Diglett's Cave) is extremely fast and one-shots Surge; Nidoking learns Earthquake, Thunderbolt, Ice Beam and Horn Drill by TM and covers four types alone.
- Psychic: Alakazam (trade a Kadabra, or level one) is the strongest thing in the game and beats Bruno and Agatha single-handed.
- Electric: Jolteon (Eevee from Celadon, Thunder Stone) or a Raichu; useful for Misty, Lorelei and Lance's Gyarados.
- Flexible: Snorlax (Route 12 or 16, needs the Poké Flute) with Body Slam and Earthquake is a wall that also hits hard.

**HM slots.** Cut, Fly, Surf, Strength and Flash are all required somewhere. Surf and Strength belong on real team members; Cut and Flash can go on a spare. Fly is worth a real slot for the travel time it saves.

**Levels.** Around Lv50 leaving Victory Road is comfortable. Below Lv45 the Elite Four becomes a Full Restore war.
