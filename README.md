# traffic-worm-effect

A minimal Python demo of the traffic “worm/accordion” effect at a green light and a coordinated mitigation using a simple CACC platoon controller.

The script simulates four vehicles starting from rest at a stop line and produces three GIF animations:
- `baseline_idm.gif` — human driving with reaction delay (IDM-based), showing the worm effect.
- `coordinated_cacc.gif` — centrally coordinated platoon (CACC) with smoother spacing and launch.
- `stacked_comparison_en.gif` — a vertical side-by-side: top = CACC, bottom = worm effect.

## What this shows
- **Worm effect:** Reaction delays propagate through the queue and create stop-and-go waves that slow discharge.
- **CACC mitigation:** A simple spacing policy with feed-forward of the leader’s acceleration reduces amplification and shortens clearance time.

## How it works
- **Baseline:** Intelligent Driver Model with an explicit reaction delay. The leader reacts after a short delay; followers react to delayed states of the vehicle ahead.
- **Coordinated:** A cooperative adaptive cruise controller uses a time-headway spacing target and PD terms on gap and relative speed, plus feed-forward of the leader’s acceleration.

## Run
```bash
python traffic_platoon_sim.py
