import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids display issues when saving headlessly

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

np.random.seed(7)

# Parameters
N = 20
P_ER = 0.2
NAME = "SHRIRAM"
DT = 0.01
STEPS_PER_LETTER = 500
HOLD_STEPS = 80
FORMATION_GAIN = 1.0
TRANSLATION_GAIN = 1.5
LETTER_SCALE = 1.0
LETTER_WIDTH = 2.0
LETTER_GAP = 1.0
START_X = 0.0
START_HOLD = 2

FRAME_STRIDE = 3  # take every 3rd simulation step as an animation frame


def connected_graph(n, p):
    while True:
        graph = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(graph):
            return graph


G = connected_graph(N, P_ER)
L = nx.laplacian_matrix(G).toarray().astype(float)

# Initial agent positions
x0 = np.random.uniform(-1, 1, (N, 2)) * 3
x0[:, 0] -= 5

TL, TM, TR = (0, 4), (1, 4), (2, 4)
ML, MM, MR = (0, 2), (1, 2), (2, 2)
BL, BM, BR = (0, 0), (1, 0), (2, 0)

LETTER_SEGMENTS = {
    "S": [(TR, TM), (TM, TL), (TL, ML), (ML, MR), (MR, BR), (BR, BM), (BM, BL)],
    "H": [(TL, BL), (TR, BR), (ML, MR)],
    "R": [(TL, BL), (TL, TR), (TR, MR), (ML, MR), (MM, BR)],
    "I": [(TL, TR), (TM, BM), (BL, BR)],
    "A": [(BL, TM), (TM, BR), (ML, MR)],
    "M": [(BL, TL), (TL, MM), (MM, TR), (TR, BR)]
}


def sample_segments(segments, n):
    segments = [(np.array(a, float), np.array(b, float)) for a, b in segments]
    lengths = np.array([np.linalg.norm(b - a) for a, b in segments])
    cumulative = np.r_[0, np.cumsum(lengths)]
    points = np.zeros((n, 2))

    for i, t in enumerate(np.linspace(0, lengths.sum(), n)):
        k = min(np.searchsorted(cumulative, t, side="right") - 1, len(segments) - 1)
        a, b = segments[k]
        length = lengths[k]
        ratio = 0 if length == 0 else (t - cumulative[k]) / length
        points[i] = a + ratio * (b - a)

    return points


def letter_points(letter, x_offset=0, scale=LETTER_SCALE):
    points = sample_segments(LETTER_SEGMENTS[letter], N) * scale
    points -= points.mean(axis=0)
    points[:, 0] += x_offset
    return points


def assign_targets(current, targets):
    cost = cdist(current, targets)
    _, cols = linear_sum_assignment(cost)
    return targets[cols]


def move_to_formation(start, target):
    x = start.copy()
    trajectory = []
    desired_center = target.mean(axis=0)

    for _ in range(STEPS_PER_LETTER):
        formation_error = x - target
        u_formation = -FORMATION_GAIN * (L @ formation_error)

        center_error = x.mean(axis=0) - desired_center
        u_translation = -TRANSLATION_GAIN * center_error

        x += DT * (u_formation + u_translation)
        trajectory.append(x.copy())

    return x, np.array(trajectory)


# Create the full trajectory
positions = x0.copy()
all_traj = []
final_positions = {}

x_offset = START_X

# Block 0: initial hold, shown before any letter starts forming
all_traj.append(
    np.repeat(positions[None, :, :], int(START_HOLD / DT), axis=0)
)

for i, letter in enumerate(NAME):
    print(f"Forming {letter} ({i + 1}/{len(NAME)})")

    target = letter_points(letter, x_offset)
    assigned_target = assign_targets(positions, target)

    positions, movement = move_to_formation(positions, assigned_target)
    all_traj.append(movement)          # movement block for this letter
    final_positions[i] = positions.copy()

    error = np.linalg.norm(positions - assigned_target)
    print(f"Finished '{letter}' at x={x_offset:.2f}, error={error:.4f}")

    hold = np.repeat(positions[None, :, :], HOLD_STEPS, axis=0)
    all_traj.append(hold)              # hold block for this letter

    x_offset += LETTER_WIDTH + LETTER_GAP

full_traj = np.concatenate(all_traj)
print("Total animation frames (raw sim steps):", len(full_traj))
print("Total blocks in all_traj:", len(all_traj))  # should be 1 + 2*len(NAME)


def build_frames():
    """Correctly walk all_traj: block 0 is the initial hold, then for each
    letter i the movement block is at index (1 + 2*i) and its hold block
    is at index (2 + 2*i)."""
    frames = []

    # initial hold, labeled as "about to form" the first letter
    for points in all_traj[0][::FRAME_STRIDE]:
        frames.append((points, 0, NAME[0]))

    for i, letter in enumerate(NAME):
        movement_block = all_traj[1 + 2 * i]
        hold_block = all_traj[2 + 2 * i]

        for points in movement_block[::FRAME_STRIDE]:
            frames.append((points, i, letter))
        for points in hold_block[::FRAME_STRIDE]:
            frames.append((points, i, letter))

    return frames


def animate_sequence():
    fig, ax = plt.subplots(figsize=(14, 5))

    agents = ax.scatter([], [], c="crimson", s=45)
    completed = ax.scatter([], [], c="lightgray", s=20)

    lines = [ax.plot([], [], color="gray", lw=0.25)[0] for _ in G.edges()]

    total_width = len(NAME) * (LETTER_WIDTH + LETTER_GAP)
    ax.set_xlim(-6, total_width + 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")

    title = ax.set_title("")

    frames = build_frames()
    print("Total rendered frames:", len(frames))

    def update(data):
        points, i, letter = data
        agents.set_offsets(points)

        for line, (a, b) in zip(lines, G.edges()):
            line.set_data(
                [points[a, 0], points[b, 0]],
                [points[a, 1], points[b, 1]]
            )

        if i:
            done = np.vstack([final_positions[j] for j in range(i)])
            completed.set_offsets(done)
        else:
            completed.set_offsets(np.empty((0, 2)))

        title.set_text(f"Forming '{letter}' ({i + 1}/{len(NAME)})")
        return [agents, completed, title] + lines

    anim = FuncAnimation(
        fig, update, frames=frames, interval=25, blit=False, repeat=False
    )
    return fig, anim


fig, anim = animate_sequence()

writer = PillowWriter(fps=40)  # fps here controls playback speed of saved GIF

def _progress(current_frame, total_frames):
    if current_frame % 50 == 0 or current_frame == total_frames - 1:
        print(f"Saving frame {current_frame + 1}/{total_frames}")

anim.save(
    "SHRIRAM_formation.gif",
    writer=writer,
    dpi=100,
    progress_callback=_progress,
)

print("Saved SHRIRAM_formation.gif")

plt.close(fig)  # release the figure/memory now that saving is done