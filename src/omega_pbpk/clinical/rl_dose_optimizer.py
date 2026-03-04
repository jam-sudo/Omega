"""Phase 44: RL Dose Optimizer — tabular Q-learning for adaptive multi-step dosing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "TherapeuticWindow",
    "PKParams",
    "RLState",
    "RLAction",
    "RLDosingPolicy",
    "RLTrainingResult",
    "PKEnvironment",
    "QLearningOptimizer",
    "optimize_dose_rl",
]

_N_TROUGH_BINS: int = 10
_N_PEAK_BINS: int = 5


@dataclass(frozen=True)
class TherapeuticWindow:
    cmin_mg_L: float
    cmax_mg_L: float
    target_trough_mg_L: float | None = None
    auc_target_mg_h_L: float | None = None


@dataclass(frozen=True)
class PKParams:
    cl_L_per_h: float
    vd_L: float
    ka_per_h: float = 1.0
    f_bioavail: float = 0.8
    route: str = "oral"
    interval_h: float = 24.0


@dataclass(frozen=True)
class RLState:
    step: int
    conc_trough_mg_L: float
    conc_peak_mg_L: float
    time_in_window: bool
    below_cmin: bool
    above_cmax: bool
    doses_given: int


@dataclass(frozen=True)
class RLAction:
    dose_mg: float
    action_idx: int


@dataclass
class RLDosingPolicy:
    optimal_dose_sequence: list[float]
    dose_grid_mg: list[float]
    n_steps: int
    interval_h: float
    pk_params: PKParams
    window: TherapeuticWindow
    final_trough_mg_L: float
    final_peak_mg_L: float
    time_in_window_fraction: float
    mean_reward_per_episode: float
    converged: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class RLTrainingResult:
    episode_rewards: list[float]
    episode_window_fractions: list[float]
    q_table_shape: tuple[int, int]
    n_episodes: int
    epsilon_final: float
    policy: RLDosingPolicy


class PKEnvironment:
    def __init__(
        self,
        pk: PKParams,
        window: TherapeuticWindow,
        dose_grid_mg: list[float],
        n_steps: int,
        reward_weights: dict[str, float] | None = None,
        seed: int = 42,
    ):
        self._pk = pk
        self._window = window
        self._dose_grid = np.array(dose_grid_mg, dtype=np.float64)
        self._n_steps = n_steps
        self._rng = np.random.default_rng(seed)
        # reward weights
        rw = reward_weights or {}
        self._w_win = rw.get("w_window", 2.0)
        self._w_sub = rw.get("w_sub", 3.0)
        self._w_tox = rw.get("w_tox", 5.0)
        self._w_target = rw.get("w_target", 1.0)
        self._w_dose = rw.get("w_dose_pen", 0.1)
        # state
        self._dose_history: list[tuple[float, float]] = []  # (t_dose, dose_mg)
        self._current_step: int = 0
        # precompute trough bins (log-spaced from cmin/10 to cmax*3)
        lo = max(window.cmin_mg_L / 10.0, 1e-6)
        hi = window.cmax_mg_L * 3.0
        self._trough_edges = np.concatenate(
            [
                [0.0],
                np.logspace(np.log10(lo), np.log10(hi), _N_TROUGH_BINS - 1),
                [np.inf],
            ]
        )
        self._peak_edges = np.array(
            [
                0.0,
                window.cmax_mg_L / 2.0,
                window.cmax_mg_L,
                window.cmax_mg_L * 2.0,
                window.cmax_mg_L * 10.0,
                np.inf,
            ]
        )
        self._n_states = _N_TROUGH_BINS * _N_PEAK_BINS * (n_steps + 1)
        self._n_actions = len(dose_grid_mg)

    def _single_dose_conc(self, t_elapsed: float, dose_mg: float) -> float:
        ke = self._pk.cl_L_per_h / self._pk.vd_L
        if self._pk.route == "iv":
            return float((dose_mg / self._pk.vd_L) * np.exp(-ke * t_elapsed))
        else:
            ka = self._pk.ka_per_h
            if abs(ka - ke) < 1e-6:
                ke = ke + 1e-6
            factor = (self._pk.f_bioavail * dose_mg / self._pk.vd_L) * (ka / (ka - ke))
            return float(factor * (np.exp(-ke * t_elapsed) - np.exp(-ka * t_elapsed)))

    def _superpose_conc(self, t_eval: float) -> float:
        total = 0.0
        for t_dose, dose in self._dose_history:
            if t_dose <= t_eval:
                total += self._single_dose_conc(t_eval - t_dose, dose)
        return total

    def _trough_peak(self, step: int) -> tuple[float, float]:
        if not self._dose_history:
            return (0.0, 0.0)
        interval_h = self._pk.interval_h
        trough = self._superpose_conc(step * interval_h)
        # Find peak from the last dose given
        t_last, dose_last = self._dose_history[-1]
        if self._pk.route == "iv":
            peak = dose_last / self._pk.vd_L
        else:
            ke = self._pk.cl_L_per_h / self._pk.vd_L
            ka = self._pk.ka_per_h
            if abs(ka - ke) < 1e-6:
                ke = ke + 1e-6
            t_max = np.log(ka / ke) / (ka - ke)
            t_max = max(0.0, float(t_max))
            peak = self._superpose_conc(t_last + t_max)
        return (float(trough), float(peak))

    def _compute_reward(self, trough: float, peak: float, step: int) -> float:
        reward = 0.0
        cmin = self._window.cmin_mg_L
        cmax = self._window.cmax_mg_L
        if trough >= cmin and peak <= cmax:
            reward += self._w_win
        if trough < cmin:
            reward -= self._w_sub * (cmin - trough) / cmin
        if peak > cmax:
            reward -= self._w_tox * (peak - cmax) / cmax
        if self._window.target_trough_mg_L is not None:
            target = self._window.target_trough_mg_L
            reward += self._w_target * np.exp(-abs(trough - target) / max(cmin, 0.01))
        # Small dose conservatism penalty
        if len(self._dose_history) > 0:
            last_dose = self._dose_history[-1][1]
            reward -= self._w_dose * last_dose / self._dose_grid.max()
        return float(reward)

    def reset(self) -> RLState:
        self._dose_history = []
        self._current_step = 0
        return RLState(
            step=0,
            conc_trough_mg_L=0.0,
            conc_peak_mg_L=0.0,
            time_in_window=False,
            below_cmin=True,
            above_cmax=False,
            doses_given=0,
        )

    def step(self, action_idx: int) -> tuple[RLState, float, bool]:
        dose = float(self._dose_grid[action_idx])
        t_dose = self._current_step * self._pk.interval_h
        self._dose_history.append((t_dose, dose))
        self._current_step += 1
        trough, peak = self._trough_peak(self._current_step)
        reward = self._compute_reward(trough, peak, self._current_step)
        done = self._current_step >= self._n_steps
        cmin = self._window.cmin_mg_L
        cmax = self._window.cmax_mg_L
        window_hit = cmin <= trough <= cmax
        state = RLState(
            step=self._current_step,
            conc_trough_mg_L=trough,
            conc_peak_mg_L=peak,
            time_in_window=window_hit,
            below_cmin=trough < cmin,
            above_cmax=peak > cmax,
            doses_given=len(self._dose_history),
        )
        return state, reward, done

    def encode_state(self, state: RLState) -> int:
        t_bin = int(np.searchsorted(self._trough_edges, state.conc_trough_mg_L, side="right")) - 1
        t_bin = int(np.clip(t_bin, 0, _N_TROUGH_BINS - 1))
        p_bin = int(np.searchsorted(self._peak_edges, state.conc_peak_mg_L, side="right")) - 1
        p_bin = int(np.clip(p_bin, 0, _N_PEAK_BINS - 1))
        s_bin = min(state.step, self._n_steps)
        return int(s_bin * _N_TROUGH_BINS * _N_PEAK_BINS + t_bin * _N_PEAK_BINS + p_bin)


class QLearningOptimizer:
    def __init__(
        self,
        env: PKEnvironment,
        n_episodes: int = 2000,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        self._env = env
        self._n_episodes = n_episodes
        self._alpha = alpha
        self._gamma = gamma
        self._eps = epsilon_start
        self._eps_end = epsilon_end
        self._eps_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)
        self.q_table: NDArray[np.float64] = np.zeros(
            (env._n_states, env._n_actions), dtype=np.float64
        )

    def train(self) -> RLTrainingResult:
        episode_rewards: list[float] = []
        episode_window_fracs: list[float] = []
        for _ep in range(self._n_episodes):
            state = self._env.reset()
            s_idx = self._env.encode_state(state)
            total_reward = 0.0
            window_steps = 0
            done = False
            while not done:
                if self._rng.uniform() < self._eps:
                    a_idx = int(self._rng.integers(0, self._env._n_actions))
                else:
                    a_idx = int(np.argmax(self.q_table[s_idx]))
                next_state, reward, done = self._env.step(a_idx)
                ns_idx = self._env.encode_state(next_state)
                target = reward + self._gamma * float(np.max(self.q_table[ns_idx])) * (
                    0.0 if done else 1.0
                )
                self.q_table[s_idx, a_idx] += self._alpha * (target - self.q_table[s_idx, a_idx])
                s_idx = ns_idx
                total_reward += reward
                if next_state.time_in_window:
                    window_steps += 1
            episode_rewards.append(total_reward)
            episode_window_fracs.append(window_steps / self._env._n_steps)
            self._eps = max(self._eps_end, self._eps * self._eps_decay)

        policy = self._extract_greedy_policy()
        converged = self._check_convergence(episode_rewards)
        mean_r = (
            float(np.mean(episode_rewards[-100:]))
            if len(episode_rewards) >= 100
            else float(np.mean(episode_rewards))
        )
        policy.mean_reward_per_episode = mean_r
        policy.converged = converged
        return RLTrainingResult(
            episode_rewards=episode_rewards,
            episode_window_fractions=episode_window_fracs,
            q_table_shape=(self.q_table.shape[0], self.q_table.shape[1]),
            n_episodes=self._n_episodes,
            epsilon_final=self._eps,
            policy=policy,
        )

    def _extract_greedy_policy(self) -> RLDosingPolicy:
        state = self._env.reset()
        s_idx = self._env.encode_state(state)
        dose_sequence: list[float] = []
        window_steps = 0
        final_trough = 0.0
        final_peak = 0.0
        done = False
        while not done:
            a_idx = int(np.argmax(self.q_table[s_idx]))
            next_state, _reward, done = self._env.step(a_idx)
            dose_sequence.append(float(self._env._dose_grid[a_idx]))
            if next_state.time_in_window:
                window_steps += 1
            final_trough = next_state.conc_trough_mg_L
            final_peak = next_state.conc_peak_mg_L
            s_idx = self._env.encode_state(next_state)

        time_in_window_fraction = (
            window_steps / self._env._n_steps if self._env._n_steps > 0 else 0.0
        )
        return RLDosingPolicy(
            optimal_dose_sequence=dose_sequence,
            dose_grid_mg=self._env._dose_grid.tolist(),
            n_steps=self._env._n_steps,
            interval_h=self._env._pk.interval_h,
            pk_params=self._env._pk,
            window=self._env._window,
            final_trough_mg_L=final_trough,
            final_peak_mg_L=final_peak,
            time_in_window_fraction=time_in_window_fraction,
            mean_reward_per_episode=0.0,  # filled in by train()
            converged=False,  # filled in by train()
        )

    def _check_convergence(self, rewards: list[float]) -> bool:
        if len(rewards) < 200:
            return False
        last = rewards[-200:]
        mean = abs(np.mean(last))
        if mean < 1e-9:
            return True
        return float(np.std(last)) / mean < 0.05


def optimize_dose_rl(
    pk_params: PKParams,
    window: TherapeuticWindow,
    n_steps: int = 10,
    dose_grid_mg: list[float] | None = None,
    n_episodes: int = 2000,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    seed: int = 42,
    reward_weights: dict[str, float] | None = None,
) -> RLTrainingResult:
    if window.cmin_mg_L >= window.cmax_mg_L:
        raise ValueError("cmin_mg_L must be < cmax_mg_L")
    if dose_grid_mg is None:
        # Auto-compute: doses that achieve ~cmin to ~cmax
        d_lo = max(
            1.0,
            window.cmin_mg_L * pk_params.cl_L_per_h / max(pk_params.f_bioavail, 0.01) * 0.5,
        )
        d_hi = min(
            5000.0,
            window.cmax_mg_L * pk_params.cl_L_per_h / max(pk_params.f_bioavail, 0.01) * 3.0,
        )
        d_lo = min(d_lo, d_hi / 2.0)
        dose_grid_mg = list(np.logspace(np.log10(d_lo), np.log10(d_hi), 10))
    env = PKEnvironment(pk_params, window, dose_grid_mg, n_steps, reward_weights, seed)
    opt = QLearningOptimizer(
        env, n_episodes, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay, seed
    )
    return opt.train()
