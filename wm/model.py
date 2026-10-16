"""
The world model: how the box moves in the next 0.1 s, given where the pusher is and where it goes.
"""
import numpy as np
import torch
from torch import nn

MEMBERS = 5
HIDDEN = 128
IN_SCALE = (100, 100, 1000, 1000)      # inputs seen by the networks in cm, cm, mm, mm
FLOOR_IN_SCALE = (100, 100, 1, 1, 100, 100, 1000, 1000)
OUT_SCALE = (1000, 1000, 180 / np.pi)  # outputs learned in mm, mm, degrees (1 mm counts as much as 1 degree)


def wrap(a):
    """An angle brought into [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def turn(vec, angle):
    """Vectors (... x 2) turned by angle (radians). Works on numpy arrays and torch tensors."""
    lib = torch if torch.is_tensor(vec) else np
    c, s = lib.cos(angle), lib.sin(angle)
    return lib.stack([c * vec[..., 0] - s * vec[..., 1], s * vec[..., 0] + c * vec[..., 1]], -1)


def steps(df, floor=False):
    """Every 0.1 s step of every push as model inputs X (N x 4) and outputs Y (N x 3), in the
    box's frame at the start of the step, plus the push each step comes from.
    floor=True (the test without the box frame): raw floor positions instead, X is N x 8."""
    X, Y, push = [], [], []
    for e, g in df.groupby("episode", sort=False):
        tip = g[["tip_x", "tip_y"]].to_numpy()
        box = g[["box_x", "box_y", "box_yaw"]].to_numpy()
        yaw = box[:-1, 2]
        if floor:
            X.append(np.hstack([box[:-1, :2], np.cos(yaw)[:, None], np.sin(yaw)[:, None], tip[:-1], np.diff(tip, axis=0)]))
            Y.append(np.hstack([np.diff(box[:, :2], axis=0), wrap(np.diff(box[:, 2]))[:, None]]))
        else:
            X.append(np.hstack([turn(tip[:-1] - box[:-1, :2], -yaw), turn(np.diff(tip, axis=0), -yaw)]))
            Y.append(np.hstack([turn(np.diff(box[:, :2], axis=0), -yaw), wrap(np.diff(box[:, 2]))[:, None]]))
        push.append(np.full(len(g) - 1, e))
    return np.vstack(X), np.vstack(Y), np.concatenate(push)


def symmetries(X, Y):
    """The 8 turned and mirrored versions of each example (the box looks the same in all of them)."""
    Xs, Ys = [], []
    for mirror in (False, True):
        x, y = X.copy(), Y.copy()
        if mirror:                        # flip across the box's x axis: y changes sign, turns reverse
            x[:, [1, 3]] *= -1
            y[:, [1, 2]] *= -1
        for _ in range(4):                # quarter turns: (a, b) -> (-b, a); the box's turn is unchanged
            Xs.append(x)
            Ys.append(y)
            x = np.column_stack([-x[:, 1], x[:, 0], -x[:, 3], x[:, 2]])
            y = np.column_stack([-y[:, 1], y[:, 0], y[:, 2]])
    return np.vstack(Xs), np.vstack(Ys)


def floor_symmetries(X, Y):
    """The same 8 versions for the raw floor inputs (a test): the finger, its move and the box's slide are turned
    and mirrored about the box's own centre and axes, and the box stays where it is (a square looks the same)."""
    box, yaw = X[:, :2], np.arctan2(X[:, 3], X[:, 2])
    xs, ys = symmetries(np.hstack([turn(X[:, 4:6] - box, -yaw), turn(X[:, 6:8], -yaw)]),
                        np.column_stack([turn(Y[:, :2], -yaw), Y[:, 2]]))
    yaw = np.tile(yaw, 8)                 # symmetries() stacks the 8 versions one block after the other
    return (np.hstack([np.tile(X[:, :4], (8, 1)), np.tile(box, (8, 1)) + turn(xs[:, :2], yaw), turn(xs[:, 2:], yaw)]),
            np.column_stack([turn(ys[:, :2], yaw), ys[:, 2]]))


class WorldModel(nn.Module):
    def __init__(self, members=MEMBERS, hidden=HIDDEN, floor=False):
        super().__init__()
        self.floor = floor                # True: the test without the box frame (see steps)
        self.nets = nn.ModuleList(
            nn.Sequential(nn.Linear(8 if floor else 4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, 3))
            for _ in range(members))
        self.register_buffer("in_scale", torch.tensor(FLOOR_IN_SCALE if floor else IN_SCALE, dtype=torch.float32))
        self.register_buffer("out_scale", torch.tensor(OUT_SCALE, dtype=torch.float32))

    def forward(self, x):
        """x: N x 4 (the same inputs for every network) or members x N x 4 (one set each), in metres.
        Returns members x N x 3: the box's slide (metres) and turn (radians), in its own frame.
        (Without the box frame: 8 inputs, and the slide is on the floor.)"""
        if x.dim() == 2:
            x = x.expand(len(self.nets), *x.shape)
        return torch.stack([net(xi * self.in_scale) for net, xi in zip(self.nets, x)]) / self.out_scale


def fit(model, X, Y, push, train_steps=6000, batch=512, lr=2e-3, seed=0, copies=True, log=print):
    """Train each network on its own resample of the pushes (drawn with replacement).
    copies=False (a test): each example shown once, without its 7 turned and mirrored versions."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    pushes = np.unique(push)
    rows = {p: np.flatnonzero(push == p) for p in pushes}
    for m, net in enumerate(model.nets):
        pick = np.concatenate([rows[p] for p in rng.choice(pushes, len(pushes))])
        copy = floor_symmetries if model.floor else symmetries
        x, y = copy(X[pick], Y[pick]) if copies else (X[pick], Y[pick])
        x = torch.tensor(x, dtype=torch.float32) * model.in_scale
        y = torch.tensor(y, dtype=torch.float32) * model.out_scale
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_steps)
        for i in range(train_steps):
            idx = torch.randint(len(x), (batch,))
            loss = ((net(x[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        log(f"network {m + 1}/{len(model.nets)}: {len(pick)} steps x {8 if copies else 1} versions, "
            f"final loss {loss.item():.3f} (mm or degrees, squared)")
    return model


@torch.no_grad()
def rollout(model, box0, tip):
    """Predict where the box goes while the pusher follows a given path, 0.1 s at a time.

    box0: B x 3 starting box poses (x, y, yaw), or members x B x 3 to carry on from each
    network's own previous prediction; tip: B x T x 2 pusher positions (table frame).
    Each network rolls out on its own, feeding its predictions back in, so their spread at
    the end says how much they disagree about the whole push.
    Returns members x B x T x 3 box poses (the first is box0).
    """
    box0 = torch.as_tensor(box0, dtype=torch.float32)
    tip = torch.as_tensor(tip, dtype=torch.float32)
    box = box0 if box0.dim() == 3 else box0.expand(len(model.nets), *box0.shape)
    path = [box]
    for k in range(tip.shape[1] - 1):
        yaw = box[..., 2]
        here, move = tip[:, k].expand_as(box[..., :2]), (tip[:, k + 1] - tip[:, k]).expand_as(box[..., :2])
        if model.floor:
            box = box + model(torch.cat([box[..., :2], torch.cos(yaw)[..., None], torch.sin(yaw)[..., None], here, move], -1))
        else:
            out = model(torch.cat([turn(here - box[..., :2], -yaw), turn(move, -yaw)], -1))
            box = box + torch.cat([turn(out[..., :2], yaw), out[..., 2:]], -1)
        path.append(box)
    return torch.stack(path, 2)


def save(model, path, **info):
    torch.save(dict(state=model.state_dict(), members=len(model.nets),
                    hidden=model.nets[0][0].out_features, info=info), path)


def load_model(path):
    saved = torch.load(path, weights_only=False)
    model = WorldModel(saved["members"], saved["hidden"])
    model.load_state_dict(saved["state"])
    return model.eval(), saved["info"]
