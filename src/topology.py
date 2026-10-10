import matplotlib.pyplot as plt

pos = {"Sink": (0, 0), 1: (-3, 2), 2: (0, 2.5), 3: (3, 2), 4: (0, 5), 5: (3, 5)}
edges = [(1, "Sink"), (2, "Sink"), (3, "Sink"), (4, 2), (5, 3)]
dead = {2, 3}   # relays that die ~day 5.9
cut_off = {4, 5}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, title, state in [(axes[0], "Day 1: all nodes alive", False),
                         (axes[1], "Day 6+: relays 2 and 3 dead", True)]:
    for a, b in edges:
        broken = state and (a in dead or b in dead)
        ax.plot(*zip(pos[a], pos[b]), color="red" if broken else "gray",
                ls="--" if broken else "-", lw=2, zorder=1)
    for n, (x, y) in pos.items():
        if n == "Sink":
            col = "gold"
        elif state and n in dead:
            col = "red"
        elif state and n in cut_off:
            col = "orange"
        else:
            col = "lightgreen"
        ax.scatter(x, y, s=1400, c=col, edgecolors="black", zorder=2)
        ax.text(x, y, "Sink" if n == "Sink" else f"N{n}", ha="center", va="center", fontweight="bold")
    ax.set_title(title); ax.set_xlim(-4.5, 4.5); ax.set_ylim(-1, 6); ax.axis("off")
axes[1].text(0, -0.8, "Red = dead relay | Orange = alive but cut off", ha="center", fontsize=9)
plt.tight_layout(); plt.savefig("topology.png", dpi=200); plt.show()