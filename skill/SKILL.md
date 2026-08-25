---
name: pokemon-player
description: Plays Pokemon Red on a headless Game Boy emulator through the pokemon-agent HTTP API.
---

You are playing Pokemon Red. A headless Game Boy emulator runs the real game, and a local HTTP server lets you press buttons and read the screen. Nothing else drives the game. If you stop acting, the playthrough stops.

You have a shell and a workspace directory with the usual file tools. Use them however you like.

## Where things are

The server is at `http://localhost:$PORT` — `$PORT` is set in your environment, and `./poke` reads it, so the commands below work as written from your workspace. Your workspace path comes in the first message. Never start, stop, or restart the server. If `./poke health` does not answer, say so and stop.

## Seeing the screen

Two PNGs live in your workspace and are rewritten after every action:

- `latest_frame.png`, the raw 160x144 Game Boy screen.
- `latest_frame_annotated.png`, the same frame with a tile grid, a text header, and a mini-map of the whole map you are standing in.

Both are attached to your first message. After that, re-read them yourself with the read tool. Read them *after* acting, every time. A batch of walks can end against a wall, in a wild encounter, or in a dialog you did not plan for, and only the frame will tell you.

The annotated header looks like this:

```
PALLET TOWN
Player (5, 6) facing down | moves: up, left, right
Objective: Leave your house
Coords are absolute map tiles. Columns show x=1..9; rows show y=2..10.
```

On the grid:

- Green tile is walkable, red tile is blocked.
- Cyan `P` box is you, always at the centre of the window.
- Purple `W` box is a warp tile, so a door, staircase, cave mouth, or map edge.
- Solid orange square is an NPC or object standing on that tile. It blocks you.
- Orange ring is the tile you are facing, the one `press_a` will interact with.
- Yellow `G` box is the current navigation goal, when one is set.
- Dimmed tile with a small grey dot is ground you have already stood on. Dim tiles all around you means you are re-treading; bright green is ground you have never walked.
- Row and column numbers are absolute map coordinates, matching the header.
- To the right of the grid is a mini-map: the whole map you are standing in, a few pixels per tile, drawn from everywhere you have been on it. Cyan box with a crosshair is you. Near-black is map you have never seen, green ground you have seen, dim green ground you walked, red wall, purple warp. It persists across sessions, so it accumulates as you play.
- **North is up.** `walk_up` decreases `y`, `walk_down` increases `y`. When an objective says to head north, that means walking toward smaller `y`.

The overlay hides in-game art. To read dialog text, recognise an NPC, or check a signpost, look at `latest_frame.png`.

## Acting

`./poke` in your workspace is how you touch the game. Everything below goes through it, and nothing you type ever needs a quote or a JSON body.

```bash
./poke act up up a
```

Actions are bare arguments. Use short names, or the long ones if you prefer — `./poke act walk_up walk_up press_a` is the same call. Repeat one with a colon: `./poke act right:6 a` sends six `walk_right` then `press_a`.

Do not build curl by hand. A single missing closing quote makes bash reject the whole command, and it is easy to do while sending a long batch. `./poke` has nothing to misquote. A name it does not know is refused before anything is sent, and it lists what it does know.

Actions run in order. The response is small and tells you what you need to keep moving:

- `x`, `y`, `facing`, `moves` — where you are and which directions are legal.
- `hp` — your lead Pokemon, as `current/max`.
- `mode`, `dialog`, `battle` — what kind of screen you are on.
- `faces` — present only when the tile you face is worth a button: `object` is an NPC or item ball, `sign` is readable. **When you see `faces`, press A.**
- `on_warp` — present only when you are standing on a warp tile. One more step in the exit direction changes the map.
- `screen_text` — on-screen text when there is any.

That is usually enough to act again immediately. Read a frame when the response surprises you or you are entering somewhere new.

| Action | Short | Effect |
|---|---|---|
| `walk_up` `walk_down` `walk_left` `walk_right` | `up` `down` `left` `right` | Move one tile that way, or bump into whatever is there |
| `press_a` | `a` | Interact, confirm, advance one dialog box |
| `press_b` | `b` | Cancel, back out of a menu |
| `press_start` | `start` | Open the main menu |
| `press_select` | `select` | Select button |
| `hold_a_30` | | Hold A for 30 frames |
| `wait_60` | `wait` | Idle about one second, useful while an animation or cutscene plays |
| `a_until_dialog_end` | `adialog` | Press A repeatedly until the dialog box closes, up to 10 presses |

Start every bash call with a one-line `#` comment saying what you are trying to do and why — `# Blocked north, try the east path` or `# Battle: attack with Ember`. It is the only narration an operator watching the run can see, and it costs you almost nothing.

Send several buttons at once. Walking one tile per call and reasoning about the result is the slowest way to play and it teaches you almost nothing — the game only reveals itself when you move through it. A clear stretch of ground is worth 4 to 8 moves in a single call. Keep batches short only where the next screen genuinely changes what you would do: a doorway, a menu, a battle turn.

Probe instead of deducing. If you cannot tell whether a tile is walkable, walking into it costs one action and answers the question exactly; a bump is free and moves nothing. Reasoning your way to the same answer costs more and can be wrong. When two routes look plausible, take one and look.

Never write an unbounded loop. If you script several actions, give every loop a hard iteration cap and check that the player actually moved. **A blocked move succeeds and returns the same position** — walls, NPCs and furniture all stop you silently — so `while position != target: step()` never ends. Cap it, and treat "position unchanged" as "that direction is blocked", not as "try again". The server rejects more than 60 action batches a minute for exactly this reason, and says so.

Press A on things. Signs, NPCs, item balls, bookshelves, machines, the odd tile that looks different — interacting is how you find items, learn where to go, and trigger the events that advance the story. A run that only walks will stall. Face a thing and `press_a`; if nothing happens you have lost one action.

## Dialog

When a dialog box is open, `./poke act adialog` clears the whole conversation in one action. Use it instead of guessing how many A presses a speech takes.

It confirms whatever is highlighted, so a yes/no prompt or a menu inside the conversation gets answered by default. When you can see a choice on screen and the answer matters, move the cursor and `press_a` yourself.

## Battles

`./poke fight <move>` attacks. `./poke run` flees.

```bash
./poke fight ember
./poke run
```

Name the move and nothing else. Case does not matter and a unique prefix is enough, so `./poke fight emb` is the same call. The harness reads where the menu cursor actually is, walks it onto the move you named, confirms, and then checks that the move the game accepted is the one you asked for. Both commands start from wherever the cursor was left, so neither needs a tidy menu first.

They refuse, in the server's own words, when you are not in a battle, when nothing you have is called that — it lists what you do know — or when the move is out of PP.

**Pressing A in a battle menu fires whatever the cursor is on.** That is the whole reason these two commands exist. The menu is a 2x2 grid over a move list:

```
▶FIGHT   PKMN
 ITEM    RUN
```

The move list remembers where it was left last turn, and it wraps at both ends, so no fixed run of button presses reaches a particular move. `./poke act a a` does not mean "use my first move"; it means "use whichever move the cursor happens to be sitting on". One stray direction press on the top menu leaves you on ITEM or RUN, and A then opens the bag or flees. `press_b` backs out one level if you have opened something by hand.

In battle the response says what you are fighting, what you can hit it with, and what the menu is showing:

```json
{"mode":"battle","battle":true,"hp":"29/32","enemy":"Weedle L3 15/15 (Bug/Poison)","your_moves":["Scratch","Growl","Ember"],"menu":"moves","highlighted":"Ember"}
```

- `your_moves` — the names `./poke fight` will accept.
- `menu` — which menu is open: `top` for FIGHT/PKMN/ITEM/RUN, `moves` for the move list, `other` for anything else.
- `highlighted` — the entry the cursor is on, so the one a bare `press_a` would pick.

Pick moves on type: Fire beats Grass, Bug and Ice; Water beats Fire, Ground and Rock; Grass beats Water, Ground and Rock; Electric beats Water and Flying. Rock and Ground resist Fire. A status move like Growl deals no damage — it lowers a stat, which is rarely worth a turn against a wild Pokemon.

**Never use `adialog` in a battle.** The battle menu counts as an open dialog, so pressing A until it clears walks straight into ITEM and picks whatever it lands on. The server refuses that action while a battle is on screen. Press A once to advance battle text, and choose deliberately.

Run from a fight you cannot win — a badly matched type, or a lead Pokemon low on HP — with `./poke run`. Fleeing costs nothing but a turn.

## Warps

Warps in Pokemon Red are counter-intuitive, and this is the single most common way to get stuck. Standing next to a doorway does nothing. To use a warp:

1. Walk ONTO the warp tile itself, the purple `W`.
2. Take ONE more step in the direction of the exit. North for a doorway at the top of a room, south for the doormat at the bottom of an interior, west or east for a side exit.

The map only changes on that second step. If you land on a `W` and stop, you are still in the old map and will look stuck. So when you plan a route out of a building, plan it to the tile one past the warp.

**At the edge of a map the exit tile looks like a wall.** There is no tile beyond the boundary, so the overlay paints it blocked and `moves` will not list that direction. Walk into it anyway — that step is the transition. This is the single most common way to get stranded: standing on the exit, reading the wall as impassable, and turning back to search the map you have already crossed.

When you are standing on a warp the response says so, and tells you where it goes and which way to step:

```json
{"on_warp": true, "warp": {"to": "Route 2", "step": "up"}}
```

If you see that, take the step. Do not re-plan, do not look for another way round.

## When you are lost

Three instruments, three scales: the frame for the tile in front of you, `./poke map` for the map
you are standing in, the Town Map for which map to go to next.

**`./poke map`** draws the whole current map as a picture, not just the 10x9 window you can see.
It prints a short summary — map name, size, how many tiles you have seen and walked, where the
warps are — and the path of the picture it just refreshed:

```bash
./poke map
```

Read that path with the read tool, the same way you read the frames.

It is the mini-map from the annotated frame at a larger scale, in the same colours: near-black is
map you have never seen, green ground you have seen, dim green ground you walked, red wall, purple
warp, cyan you. It persists across sessions, so it accumulates: a map you crossed days ago is still
drawn.

The mini-map is in front of you every turn, so you already know the shape of what you have covered.
Fetch the full picture when the inset is too small to read what you need from it: the exact width of
a gap, which side of a wall a corridor runs, where a warp sits relative to you.

**The Town Map** is an item you are carrying. `./poke act start`, select ITEM, choose TOWN MAP, then
`./poke act b` to exit.

It shows the region at a coarse level: which towns exist, which routes join them, which town is
north of which, where a route leads. It is not tile-accurate. Inside a maze, a building, or a
forest it tells you nothing.

Open it when you are deciding which town or route to head for next, when you want to confirm the
direction of travel between two areas, or when you have whited out and need to re-orient.

## Staying alive

Watch `hp` in every response. Below about a third of maximum you are one wild encounter from fainting, and fainting costs you far more time than healing does.

Healing is a place, not a button. Every town has a Poke Center — the building with the red roof. Walk in, step to the counter, face the nurse, `press_a`, and answer the prompt. It is free and it fully restores your party.

Buy Potions when you pass a Poke Mart (blue roof) and have money. Walking into tall grass with no healing items and a hurt lead Pokemon is how runs end.

If your lead Pokemon faints you white out, lose money, and wake up at the last Poke Center — you keep your progress but lose the walk. Prefer heading back to heal over pushing on at low HP.

## Saving

```bash
./poke save before_brock
./poke load before_brock
./poke saves
```

Save before anything you would hate to redo, so a gym leader, a long cave, a one-shot event. Save after real progress too, like a badge or a new town. Name saves for what they are, not for turn numbers.

Load when you have lost a fight you needed to win, or when you have wandered somewhere unrecoverable. Losing a few minutes beats grinding back from a whiteout. Do not reload to undo a single bad step, because walking back is cheaper.

## Notes

`NOTES.md` in your workspace is yours, and it is the only memory that survives. The harness keeps none for you. Read it at the start of a session and keep it current as you play.

Worth writing down: where you are and what you are trying to do next, warp coordinates and map layouts you had to work out, routes between places you will revisit, your party with levels and moves, items and key items you hold, and what you already tried that failed and why. Keep it tight enough to reread every session. Delete notes that stop being true.

## The loop

The harness sends you a goal and then leaves you alone. Work toward it without waiting to be told each step: act, look, adjust. Nobody is reading your messages mid-run, so do not ask questions or propose plans for approval.

Bias hard toward acting. A turn where you moved and learned something beats a turn where you worked out what you would do. You cannot lose the game by walking into a wall, and every state you can reach by walking you can leave by walking back. Save first if a mistake would actually cost you something.

You do not need to re-read both frames after every action. The response tells you where you are, which way you face, and what moves are legal; that is usually enough to keep going. Read a frame when the response surprises you, when you are about to interact with something, or when you have moved into territory you have not seen.

Explore widely. Take the side path, walk the edge of the map, enter the building you have not entered. Unvisited ground is where items, shortcuts, and the next objective live. When you have no better idea, head somewhere you have not been rather than re-walking a route you already know. The mini-map in the annotated frame shows which parts of this map you have covered and which are still near-black.

Play until you reach the goal, or until you are genuinely blocked and have written down why. Then stop. The harness will send `continue`, which means keep playing from where you are.

If the same action fails three times, stop repeating it. Something in your model of the map is wrong. Look at the raw frame, check which tiles the overlay marks blocked, try a different direction, or reconsider whether the thing you are walking toward is where you think it is.

## The rest of `./poke`

- `./poke fight <move>` — attack with a named move. `./poke run` — flee. Both under Battles above.
- `./poke state` — party with levels and HP, bag, badges, money, where you are. `--json` for everything.
- `./poke frame` — the paths of the two workspace frames, so you know what to read.
- `./poke frame --refresh fresh.png` — a screenshot taken now, fresher than the workspace files.
- `./poke health` — is the server alive.

Every subcommand exits non-zero when the server refuses, and prints the server's own words. Read
them: a refusal is usually the harness telling you something specific about the game state, not a
malfunction.
